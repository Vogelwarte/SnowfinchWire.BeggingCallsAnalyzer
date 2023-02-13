import fire

from src.beggingcallsanalyzer.prediction.prediction import predict
from src.beggingcallsanalyzer.training.persistence import load_model, save_model

from pathlib import Path
from os import getenv
from src.beggingcallsanalyzer.training.train import clean_output_directory, load_and_prepare_data, fit_model
from src.beggingcallsanalyzer.training.evaluation import evaluate_model
import pandas as pd
from dataclasses import dataclass

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
    def predict(self, model_path, files_directory, window, step, merge_window, cut_length):
        model = load_model(model_path)
        predict(model, files_directory, window, step, merge_window=merge_window, cut_length=cut_length)

    def train(self, path = None, show_progressbar = False, merge_window = 10, cut_length=2.2):
        base_path = Path(path if path else getenv('SNOWFINCH_TRAINING_DATA_PATH'))
        clean_output_directory('.out')
        overlap_percentage = 0.7
        reports = []
        for win_length in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                for window_type in ['hamming', 'hann', 'boxcar']:
                    print(f'Training with {win_length}s long {window_type} window and overlap percentage {overlap_percentage}')
                    print('Preparing data...')
                    x_train, y_train, x_test, y_test = load_and_prepare_data(base_path, win_length, win_length,
                                                                             overlap_percentage, window_type=window_type, 
                                                                             show_progressbar = show_progressbar)
                    print('Done')
                    print('Training model...')
                    model = fit_model(x_train, y_train)
                    save_model(model, f'.out/svm_win_len_{win_length}_overlap_{overlap_percentage}_win_type_{window_type}.skops')
                    accuracy, cm = evaluate_model(model, x_test, y_test, win_length, hop_length=win_length, 
                                                  merge_window=merge_window, cut_length=cut_length)
                    reports.append(TrainingReport(win_length, window_type, accuracy, cm[1][1], cm[0][0], cm[0][1], cm[1][0]))
                    print('Done')

        print('Final results:')
        results_df = pd.DataFrame(reports)
        print(results_df.to_string(index=False))


if __name__ == '__main__':
    fire.Fire(Cli)
