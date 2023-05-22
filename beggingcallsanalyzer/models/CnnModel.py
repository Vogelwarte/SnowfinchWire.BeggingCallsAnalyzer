from __future__ import annotations

from pathlib import Path
from typing import Union
import warnings

import numpy as np
import pandas as pd
from opensoundscape import CNN
from opensoundscape.torch.models.cnn import load_model, use_resample_loss
from opensoundscape.metrics import predict_multi_target_labels
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from ..training.postprocessing import post_process

from typing import Literal


class CnnModel:
    def __init__(self, win_length, batch_size=128, num_workers=14, epochs=7) -> None:
        self.win_length = win_length
        self.batch_size = batch_size
        self.num_workers=num_workers
        self.epochs = epochs
        self._model = CNN('resnet18', classes=['feeding', 'contact'], sample_duration=win_length)
        use_resample_loss(self._model)

    def fit(self, train_path: Union[str, Path], onehot_labels_file = 'one-hot_labels.csv') -> CnnModel:
        """
        Trains the model on all files in a directory. 
        :param train_path: path to a directory containing training data
        :param show_progressbar: show a progressbar when processing input files
        :param extension: audio file extensions
        :return: fitted model
        """
        labels = pd.read_csv(f'{train_path}/{onehot_labels_file}')

        if not labels.columns.str.contains('feeding'):
            warnings.warn("")
        if not labels.columns.str.contains('contact'):
            warnings.warn("")
        if not labels.columns.str.contains('filename'):
            warnings.warn("")

        redundant_columns = [c for c in labels.columns.str if c not in {'filename', 'feeding', 'contact'}]
        labels = labels.drop(columns=redundant_columns)

        train_df = labels.set_index('filename')
        self._model.train(
            train_df=train_df,
            save_path=None, 
            epochs=self.epochs,
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
        )
        
        return self

    def predict(self, recordings, predict_path, threshold=0.8, merge_window = 10, cut_length = 2.2, extension = 'flac'):
        """
        Splits every audio files in the given directory into windows of specified length and predicts the occurenced of an
        event on each of them.
        :param predict_path: path to a directory containg audio data
        :param merge_window: (in seconds) detections within this interval will be merged during postprocessing
        :param cut_length: (in seconds) detections shorter than this will be removed during postprocessing
        :param show_progressbar:show a progress bar when processing input file
        :param extension: audio file extensions
        :return: a dictionary containing predicted values for every window for every audio file in predict_path
        """
        num_workers = max(os.cpu_count() / 2, 20)
        model_results = self._model.predict(recordings, activation_layer='sigmoid', batch_size=256, num_workers=num_workers)
        df = predict_multi_target_labels(model_results, threshold)
        df['contact'] = (df['contact'] ^ df['feeding']) & df['contact']
        results = {}
        for file, new_df in df.groupby(level=0):
            
            feeding_df = self.__group_classes(new_df.droplevel(0), 'feeding', merge_window, cut_length)
            contact_df = self.__group_classes(new_df.droplevel(0), 'contact', merge_window, cut_length)
            file_results = pd.concat([feeding_df, contact_df]).sort_values('start_time')
            results[file] = {
                'duration': new_df.reset_index().iloc[-1]['end_time'],
                'predictions': file_results.reset_index(drop=True)
            }

        df = pd.concat([v['predictions'] for v in results.values()], keys=results.keys(), names=['filename', 'idx']).drop(columns=['start_time', 'end_time'])
        pattern = f'(.*)[\\/](?P<brood_id>[^\\/]+)[\\/](?:[^\\/]+)[\\/](?P<datetime>.*)\.{extension}' #'(?P<BROOD_ID>.*)-BA[0-9]+_BS[0-9]+-.*_(?P<DATE>[0-9]{8})_(?P<TIME>[0-9]{6}).flac'
        data = df.index.get_level_values(0).str.extract(pattern)
        data.index.name='idx'
        df = df.reset_index().join(data)
        df = pd.get_dummies(df, columns=['class'])
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d_%H%M%S')
        df = df.groupby(['brood_id', 'datetime']).sum().reset_index()
        df = df.drop(columns=['filename', 'idx', 0])
        summary_path = f'{predict_path}/summary.csv'
        df.to_csv(summary_path, index=None, mode='a', header=not os.path.exists(summary_path))
        # df = df.set_index(['brood_id', 'datetime'])
        # df = df.unstack(level=[0]).resample('1h').first().stack(level=[1], dropna=False).swaplevel(1, 0).sort_index()
        # fig, ax = plt.subplots()
        # for date, new_df in df.groupby(level=0):
        #     ax.plot(new_df.index.get_level_values(1), new_df['class_feeding'], label=date, marker='o')
        # ax.set_ylim(bottom=0)
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        # plt.legend()
        # plt.xticks(rotation=30)
        # plt.ylabel("Number of feedings")
        # plt.xlabel("Date and time")
        # plt.savefig(f'{predict_path}/feeding_plot.png')
        
        return results

    def evaluate(self, test_path, merge_window = 10, cut_length = 2.2, show_progressbar = True, extension = 'flac') -> \
    tuple[float, np.ndarray]:
        """
        Evaluates performance of the model on all files in a directory.
        :param test_path: path to a directory containg audio data
        :param merge_window: (in seconds) detections within this interval will be merged during postprocessing
        :param cut_length: (in seconds) detections shorter than this will be removed during postprocessing
        :param show_progressbar:show a progress bar when processing input file
        :param extension: audio file extensions
        :return: a tuple containing accuracy and confusion matrix
        """
        raise NotImplementedError()


    @staticmethod
    def from_file(path: Path | str, *, win_length = None, hop_length = None, window_type = None,
                  percentage_overlap = None) -> CnnModel:
        """
        Creates SvmModel from file. Parameters will be read from the file name. If they are not present or need to be
        overriden, they can be passed as arguments to this method.
        :param path: path to the saved model
        :param win_length: analysis window length (in seconds)
        :param hop_length: window hop length (in seconds)
        :param window_type: type of windowing function to be used ('hamming', 'hann', 'boxcar')
        :param percentage_overlap: Minimum percentage overlap of analysis window and input label that should be achieved
        for an event to be detected
        :return: constructed model
        """
        oss_model: CNN = load_model(path)
        
        model = CnnModel(0)
        model._model = oss_model
        model.win_length = oss_model.preprocessor.sample_duration
        return model
    
    def save(self, path):
        self._model.save(path)

    def __group_classes(self, df: pd.DataFrame, class_name: Literal['contact', 'feeding'], contact_merge_window, contact_cut_length):
        df_export = df[class_name].to_frame()
        if class_name == 'contact':
            df_export[class_name] = post_process(df_export.to_numpy().flatten(), self.win_length, self.win_length, contact_merge_window, contact_cut_length)
        df_export = df_export[df_export[class_name]==1]
        df_export = df_export.reset_index()
        df_export = df_export.rename(columns={class_name: 'class'})
        df_export['group'] = ((df_export["start_time"].round(1) > df_export['end_time'].shift().round(1)) | (df_export['class'] != df_export['class'].shift())).cumsum()
        result = df_export.groupby(["group", "class"]).agg({'start_time': 'min', 'end_time': 'max'})
        result = result.reset_index(names=['group', 'class']).drop(columns=['group'])[["start_time", "end_time", "class"]]
        result['class'] = class_name
        return result
