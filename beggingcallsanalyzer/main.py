from dataclasses import dataclass
from os import getenv
from pathlib import Path

import fire
import pandas as pd

from beggingcallsanalyzer.models.SvmModel import SvmModel
from beggingcallsanalyzer.training.evaluation import evaluate_model
from beggingcallsanalyzer.training.persistence import save_model
from beggingcallsanalyzer.training.postprocessing import to_audacity_labels
from beggingcallsanalyzer.training.train import clean_output_directory, load_and_prepare_data, fit_model
from beggingcallsanalyzer.utilities.exceptions import ArgumentError


@dataclass
class TrainingReport:
    win_length: float
    window_type: str
    accuracy: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class Cli:
    def predict(self, model_path, files_directory, merge_window = 10, cut_length = 2.2, win_length = None,
                hop_length = None, window_type = None, overlap_percentage = None, extension = 'flac',
                batch_size = 100):
        model = SvmModel.from_file(model_path, win_length = win_length, hop_length = hop_length,
                                   window_type = window_type,
                                   percentage_overlap = overlap_percentage)

        recordings = list(Path(files_directory).rglob(f'*.{extension}'))
        for i in range(0, len(recordings), batch_size):
            to_idx = min(i + batch_size, len(recordings))
            # try:
            predictions = model.predict(recordings[i:to_idx], merge_window = merge_window, cut_length = cut_length)

            try:
                for filename, data in predictions.items():
                    labels = to_audacity_labels(data['predictions'], data['duration'], model.win_length, model.hop_length)
                    labels.to_csv(f'{filename.parent}/predicted_{filename.stem}.txt', header = None, index = None, sep = '\t')
            except ArgumentError as e:
                print(e)
                print('Quitting...')
                return

    def train_evaluate(self, path = None, show_progressbar = False, merge_window = 10, cut_length = 2.2,
                       output_path: str = '.out'):
        data_path = Path(path if path else getenv('SNOWFINCH_TRAINING_DATA_PATH'))

        output_path = Path(output_path)
        output_path.mkdir(parents = True, exist_ok = True)
        clean_output_directory(output_path)

        overlap_percentage = 0.7
        reports = []
        for win_length in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for window_type in ['hamming', 'hann', 'boxcar']:
                print(
                    f'Training with {win_length}s long {window_type} window and overlap percentage {overlap_percentage}')
                print('Preparing data...')
                try:
                    x_train, y_train, x_test, y_test = load_and_prepare_data(data_path, win_length, win_length,
                                                                             overlap_percentage,
                                                                             window_type = window_type,
                                                                             show_progressbar = show_progressbar)
                except ValueError as e:
                    print(e)
                    print('Quitting...')
                    return

                print('Done')
                print('Training model...')
                model = fit_model(x_train, y_train)
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


def run():
    fire.Fire(Cli)
