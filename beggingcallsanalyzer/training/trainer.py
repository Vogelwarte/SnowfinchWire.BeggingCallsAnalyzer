from dataclasses import dataclass
from os import getenv
import os
from pathlib import Path
from shutil import rmtree

import pandas as pd
import rich.progress
from more_itertools import chunked
from sklearn.model_selection import train_test_split
from beggingcallsanalyzer.plotting.plotting import (
    plot_feeding_count_daily, plot_feeding_count_hourly,
    plot_feeding_duration_daily, plot_feeding_duration_hourly)

from beggingcallsanalyzer.common.preprocessing.io import load_recording_data
from beggingcallsanalyzer.models.SvmModel import SvmModel
from beggingcallsanalyzer.plotting.summary import create_summary_csv
from beggingcallsanalyzer.training.evaluation import evaluate_model
from beggingcallsanalyzer.training.persistence import save_model
from beggingcallsanalyzer.training.preprocessing import (
    extract_features, process_features_classes)
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

class Trainer:
    def load_and_prepare_data(path: Path, window, step, percentage_overlap, test_size = 0.1, extension = 'flac', window_type = 'hamming', show_progressbar = True) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if not path.exists():
            raise ArgumentError("Provided path is incorrect, it does not exist")
        elif path.is_file():
            raise ArgumentError("Provided path is incorrect, it should be a directory")
        files = list(path.glob(f'**/*.{extension}'))

        train_paths, test_paths = train_test_split(files, test_size = test_size, random_state = 2)
        data = []
        with rich.progress.Progress(total = len(files), disable=not show_progressbar) as progress:
            task1 = progress.add_task("Processing training files...", total=len(train_paths))
            for recording in train_paths:
                file = load_recording_data(recording)
                duration = len(file.audio_data) / float(file.audio_sample_rate)
                df = extract_features(file.audio_data, file.audio_sample_rate, window, step, window_type)
                df = process_features_classes(df, file.labels, percentage_overlap, duration, window, step)
                data.append(df)
                progress.update(task1)
            train = pd.concat(data, ignore_index = True)

            data.clear()
            task2 = progress.add_task("Processing test files...", total=len(test_paths))
            for recording in test_paths:
                file = load_recording_data(recording)
                duration = len(file.audio_data) / float(file.audio_sample_rate)
                df = extract_features(file.audio_data, file.audio_sample_rate, window, step, window_type)
                df = process_features_classes(df, file.labels, percentage_overlap, duration, window, step)
                data.append(df)
                progress.update(task2)
            test = pd.concat(data, ignore_index = True)

        return train.drop(columns = ['y']), train['y'], test.drop(columns = ['y']), test['y']

    def predict(self, model_path, input_directory, output_directory, merge_window = 3, cut_length = 2.2, win_length = None,
                hop_length = None, window_type = None, extension = 'flac', processing_batch_size=100, create_plots=True):
        model = SvmModel.from_file(model_path, win_length = win_length, hop_length = hop_length,
                                   window_type = window_type)
        input_directory_path = Path(input_directory)
        recordings = list(input_directory_path.rglob(f'*.{extension}'))
        df_summary = pd.DataFrame()
        output_directory_path = Path(output_directory)
        output_directory_path.mkdir(parents=True, exist_ok=True)
        incorrect_folder_structure = False
        for recordings_chunk in chunked(recordings, processing_batch_size):
            predictions = model.predict(recordings_chunk, merge_window = merge_window, cut_length = cut_length)
            
            if not incorrect_folder_structure and (bad_files := [f.name for f in filter(lambda k: k.parent.parent.parent != input_directory_path, predictions.keys())]):
                print(f'Files {", ".join(bad_files)} are in an unexpected place in the directory structure. All predictions will be places in the root of the output directory, summary and plots will not be generated')
                incorrect_folder_structure = True
                output_path = output_directory_path

            if not incorrect_folder_structure:
                summary = create_summary_csv(predictions, output_directory, extension)
                df_summary = pd.concat([df_summary, summary])
            
            for filename, data in predictions.items():
                if not incorrect_folder_structure:
                    output_path = Path(f'{output_directory}/{filename.parent.parent.name}/{filename.parent.name}/')
                output_path.mkdir(parents=True, exist_ok=True)
                data['predictions'].to_csv(output_path/f'{filename.stem}.txt', header = None, index = None, sep = '\t')

        if create_plots and not incorrect_folder_structure:
            summary_path = f'{output_directory}/summary.csv'
            df_summary.to_csv(summary_path, index=None, header=not os.path.exists(summary_path))
            try:                
                plot_feeding_count_hourly(df_summary, output_directory)
                plot_feeding_duration_hourly(df_summary, output_directory)
                plot_feeding_count_daily(df_summary, output_directory)
                plot_feeding_duration_daily(df_summary, output_directory)
            except:
                print("Could not create some of the plots, are you sure that the files are in the required directory structure?")

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


    def train(self, training_data_path, win_length, hop_length, window_type, overlap_percentage, output_path: str = '.out',
              show_progressbar = True):
        model = SvmModel(win_length, hop_length = hop_length, window_type = window_type,
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


    def clean_output_directory(path: Path):
        for path in path.glob("**/*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)
