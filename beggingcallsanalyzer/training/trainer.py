from dataclasses import dataclass
from os import getenv
from pathlib import Path
import pandas as pd

from beggingcallsanalyzer.models.SvmModel import SvmModel
from beggingcallsanalyzer.training.evaluation import evaluate_model
from beggingcallsanalyzer.training.persistence import save_model
from beggingcallsanalyzer.training.postprocessing import to_audacity_labels
from beggingcallsanalyzer.utilities.exceptions import ArgumentError

from pathlib import Path
from shutil import rmtree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .preprocessing import extract_features, process_features_classes
from ..common.preprocessing.io import load_recording_data


from typing import Union

from ..utilities.exceptions import ArgumentError

@dataclass
class TrainingReport:
    win_length: float
    window_type: str
    accuracy: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

class Trainer:
    def load_and_prepare_data(path: Path, window, step, percentage_overlap, test_size = 0.1, extension = 'flac', window_type = 'hamming', show_progressbar = True) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if not path.exists():
            raise ArgumentError("Provided path is incorrect, it does not exist")
        elif path.is_file():
            raise ArgumentError("Provided path is incorrect, it should be a directory")
        files = list(path.glob(f'**/*.{extension}'))

        train_paths, test_paths = train_test_split(files, test_size = test_size, random_state = 2)
        data = []
        with tqdm(total = len(files), disable=not show_progressbar) as pbar:
            for recording in train_paths:
                file = load_recording_data(recording)
                duration = len(file.audio_data) / float(file.audio_sample_rate)
                df = extract_features(file.audio_data, file.audio_sample_rate, window, step, window_type)
                df = process_features_classes(df, file.labels, percentage_overlap, duration, window, step)
                data.append(df)
                pbar.update()
            train = pd.concat(data, ignore_index = True)

            data.clear()
            for recording in test_paths:
                file = load_recording_data(recording)
                duration = len(file.audio_data) / float(file.audio_sample_rate)
                df = extract_features(file.audio_data, file.audio_sample_rate, window, step, window_type)
                df = process_features_classes(df, file.labels, percentage_overlap, duration, window, step)
                data.append(df)
                pbar.update()
            test = pd.concat(data, ignore_index = True)

        return train.drop(columns = ['y']), train['y'], test.drop(columns = ['y']), test['y']

    def predict(self, model_path, files_directory, merge_window = 10, cut_length = 2.2, win_length = None,
                hop_length = None, window_type = None, overlap_percentage = None, extension = 'flac'):
        model = SvmModel.from_file(model_path, win_length = win_length, hop_length = hop_length,
                                   window_type = window_type,
                                   percentage_overlap = overlap_percentage)

        # try:
        predictions = model.predict(files_directory, merge_window = merge_window, cut_length = cut_length,
                                    extension = extension)

        for filename, data in predictions.items():
            labels = to_audacity_labels(data['predictions'], data['duration'], model.win_length, model.hop_length)
            labels.to_csv(f'{filename.parent}/predicted_{filename.stem}.txt', header = None, index = None, sep = '\t')


    def train_evaluate(self, path = None, show_progressbar = False, merge_window = 10, cut_length = 2.2,
                       output_path: str = '.out'):
        data_path = Path(path if path else getenv('SNOWFINCH_TRAINING_DATA_PATH'))

        output_path = Path(output_path)
        output_path.mkdir(parents = True, exist_ok = True)
        self.clean_output_directory(output_path)

        overlap_percentage = 0.7
        reports = []
        for win_length in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for window_type in ['hamming', 'hann', 'boxcar']:
                print(
                    f'Training with {win_length}s long {window_type} window and overlap percentage {overlap_percentage}')
                print('Preparing data...')
                try:
                    x_train, y_train, x_test, y_test = self.load_and_prepare_data(data_path, win_length, win_length,
                                                                             overlap_percentage,
                                                                             window_type = window_type,
                                                                             show_progressbar = show_progressbar)
                except ValueError as e:
                    print(e)
                    print('Quitting...')
                    return

                print('Done')
                print('Training model...')
                model = self.fit_model(x_train, y_train)
                save_model(model,
                           output_path / f'svm_win_len_{win_length}_overlap_{overlap_percentage}_win_type_{window_type}.skops')
                accuracy, cm = evaluate_model(model, x_test, y_test, win_length, hop_length = win_length,
                                              merge_window = merge_window, cut_length = cut_length)
                reports.append(
                    TrainingReport(win_length, window_type, accuracy, cm[1][1], cm[0][0], cm[0][1], cm[1][0]))
                print('Done')

        print('Final results:')
        results_df = pd.DataFrame(reports)
        print(results_df.to_string(index = False))


    def train(self, training_data_path, win_length, window_type, overlap_percentage, output_path: str = '.out',
              show_progressbar = True):
        model = SvmModel(win_length, hop_length = win_length, window_type = window_type,
                         percentage_overlap = overlap_percentage)
        try:
            model.fit(training_data_path, show_progressbar)
        except ArgumentError as e:
            print(e)
            print('Quitting...')
            return

        output_path = Path(output_path)
        if not output_path.is_file():
            output_path.mkdir(parents = True, exist_ok = True)
            output_path = output_path / f'svm_win_len_{win_length}_overlap_{overlap_percentage}_win_type_{window_type}.skops'

        save_model(model, output_path)


    def fit_model(x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series]) -> Pipeline:
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('svc', SVC(C = 20, cache_size = 2000, probability=True))
        ])
        pipe.fit(x_train, y_train)
        return pipe

    def clean_output_directory(path: Path):
        for path in path.glob("**/*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)
