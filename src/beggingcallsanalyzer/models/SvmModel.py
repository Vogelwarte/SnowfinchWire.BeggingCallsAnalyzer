from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tqdm.autonotebook import tqdm

from ..common.preprocessing.io import load_recording_data
from ..training.evaluation import evaluate_model
from ..training.persistence import load_model
from ..training.postprocessing import post_process
from ..training.preprocessing import extract_features, process_features_classes


class SvmModel:
    def __init__(self, win_length, hop_length, window_type, percentage_overlap) -> None:
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_type = window_type
        self.percentage_overlap = percentage_overlap
        self._pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('svc', SVC(C = 20, cache_size = 2000))
        ])

    def fit(self, train_path: Union[str, Path], show_progressbar = True, extension = 'flac') -> SvmModel:
        """
        Train the model
        :param train_path: path to a directory containing training data
        :param show_progressbar: show a progressbar when processing input files
        :param extension: audio file extensions
        :return: fitted model
        """
        files = Path(train_path).glob(f'**/*.{extension}')
        data = []

        for recording in tqdm(files, disable = not show_progressbar):
            file = load_recording_data(recording)
            duration = len(file.audio_data) / float(file.audio_sample_rate)
            df = extract_features(file.audio_data, file.audio_sample_rate, self.win_length, self.hop_length,
                                  self.window_type)
            df = process_features_classes(df, file.labels, self.percentage_overlap, duration, self.win_length,
                                          self.hop_length)
            data.append(df)
        train = pd.concat(data, ignore_index = True)
        x_train, y_train = train.drop(columns = ['y']), train['y']
        self._pipeline.fit(x_train, y_train)
        return self

    def predict(self, predict_path, merge_window = 10, cut_length = 2.2, show_progressbar = True, extension = 'flac'):
        """
        Processes all audio files in the given directory to
        :param predict_path: path to a directory containg audio data
        :param merge_window: (in seconds) detections within this interval will be merged during postprocessing
        :param cut_length: (in seconds) detections shorter than this will be removed during postprocessing
        :param show_progressbar:show a progress bar when processing input file
        :param extension: audio file extensions
        :return: a dictionary containing predicted values for every window for every audio file in predict_path
        """
        files = Path(predict_path).glob(f'**/*.{extension}')
        flac: Path
        results = {}
        for flac in tqdm(files, disable = not show_progressbar):
            audio_data, sample_rate = sf.read(flac)
            df = extract_features(audio_data, sample_rate, self.win_length, self.hop_length)
            y_pred = self._pipeline.predict(df)
            y_processed = post_process(y_pred, self.win_length, self.hop_length, merge_window, cut_length)
            results[flac] = {
                'duration': len(audio_data) / float(sample_rate),
                'predictions': y_processed
            }
        return results

    def evaluate(self, test_path, merge_window = 10, cut_length = 2.2, show_progressbar = True, extension = 'flac') -> \
    tuple[float, np.ndarray]:
        """
        Evaluates performance of the model
        :param test_path: path to a directory containg audio data
        :param merge_window: (in seconds) detections within this interval will be merged during postprocessing
        :param cut_length: (in seconds) detections shorter than this will be removed during postprocessing
        :param show_progressbar:show a progress bar when processing input file
        :param extension: audio file extensions
        :return: a tuple containing accuracy and confusion matrix
        """
        files = Path(test_path).glob(f'**/*.{extension}')
        data = []

        for recording in tqdm(files, disable = not show_progressbar):
            file = load_recording_data(recording)
            duration = len(file.audio_data) / float(file.audio_sample_rate)
            df = extract_features(file.audio_data, file.audio_sample_rate, self.win_length, self.hop_length,
                                  self.window_type)
            df = process_features_classes(df, file.labels, self.percentage_overlap, duration, self.win_length,
                                          self.hop_length)
            data.append(df)
        test = pd.concat(data, ignore_index = True)
        x_test, y_test = test.drop(columns = ['y']), test['y']
        accuracy, cm = evaluate_model(self._pipeline, x_test, y_test, self.win_length, hop_length = self.hop_length,
                                      merge_window = merge_window, cut_length = cut_length)
        return accuracy, cm

    @staticmethod
    def from_file(path: Path | str, *, win_length = None, hop_length = None, window_type = None,
                  percentage_overlap = None) -> SvmModel:
        pipeline, matched_win_length, matched_percentage_overlap, matched_window_type = load_model(path)
        if matched_win_length is None and win_length is None:
            raise ValueError('Could not find window length in filename and none was provided')
        if matched_percentage_overlap is None and percentage_overlap is None:
            raise ValueError('Could not find overlap percentage in filename and none was provided')
        if matched_window_type is None and window_type is None:
            raise ValueError('Could not find window type in filename and none was provided')

        model = SvmModel(
            win_length = matched_win_length if win_length is None else win_length,
            hop_length = matched_win_length if hop_length is None else hop_length,
            window_type = matched_window_type if window_type is None else window_type,
            percentage_overlap = matched_percentage_overlap if percentage_overlap is None else percentage_overlap)

        model._pipeline = pipeline
        return model
