from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
from opensoundscape import CNN
from opensoundscape.metrics import predict_multi_target_labels
from opensoundscape.torch.models.cnn import load_model, use_resample_loss

from beggingcallsanalyzer.training.postprocessing import post_process


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

    def predict(self, recordings: pd.DataFrame, threshold=0.85, merge_window = 10, cut_length = 2.2, contact_merge_window = 10, contact_cut_length = 3, 
                batch_size=256, contact_threshold=0.887):
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
        num_workers = int(min(os.cpu_count() / 2, 20))
        model_results = self._model.predict(recordings, activation_layer='sigmoid', batch_size=batch_size, num_workers=num_workers)
        df = predict_multi_target_labels(model_results, [threshold, contact_threshold, 1])
        df['contact'] = (df['contact'] ^ df['feeding']) & df['contact']
        results = {}
        for file, new_df in df.groupby(level=0):
            feeding_df = self.__group_classes(new_df.droplevel(0), 'feeding', merge_window, cut_length)
            contact_df = self.__group_classes(new_df.droplevel(0), 'contact', contact_merge_window, contact_cut_length)
            file_results = pd.concat([feeding_df, contact_df]).sort_values('start_time')
            results[file] = {
                'duration': new_df.reset_index().iloc[-1]['end_time'],
                'predictions': file_results.reset_index(drop=True)
            }
        
        return results

    @staticmethod
    def from_file(path: Path | str) -> CnnModel:
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
