from os import getenv
from pathlib import Path
from shutil import rmtree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

from src.beggingcallsanalyzer.training.evaluation import evaluate_model
from src.beggingcallsanalyzer.training.persistence import save_model
from .preprocessing import extract_features, process_features_classes
from ..common.preprocessing.io import load_recording_data


def fit_model(x_train: pd.DataFrame, y_train: pd.DataFrame) -> SVC:
    svm = SVC(C = 20, cache_size = 2000)
    svm.fit(x_train, y_train)
    return svm


def load_and_prepare_data(path: Path, window, step, percentage_overlap, test_size = 0.1, extension = 'flac'):
    flacs = list(path.glob(f'**/*.{extension}'))

    train_paths, test_paths = train_test_split(flacs, test_size = test_size, random_state = 1)
    data = []
    with tqdm(total = len(flacs)) as pbar:
        for flac in train_paths:
            file = load_recording_data(flac)
            duration = len(file.audio_data) / float(file.audio_sample_rate)
            df = extract_features(file.audio_data, file.audio_sample_rate, window, step)
            df = process_features_classes(df, file.labels, percentage_overlap, duration, window, step)
            data.append(df)
            pbar.update()
        train = pd.concat(data, ignore_index = True)

        data.clear()
        for flac in test_paths:
            file = load_recording_data(flac)
            duration = len(file.audio_data) / float(file.audio_sample_rate)
            df = extract_features(file.audio_data, file.audio_sample_rate, window, step)
            df = process_features_classes(df, file.labels, percentage_overlap, duration, window, step)
            data.append(df)
            pbar.update()
        test = pd.concat(data, ignore_index = True)

    return train.drop(columns = ['y']), train['y'], test.drop(columns = ['y']), test['y']


def clean_output_directory(path):
    for path in Path(path).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)


if __name__ == '__main__':
    base_path = Path(getenv('SNOWFINCH_TRAINING_DATA_PATH'))
    clean_output_directory('.out')
    for win_length in [0.5, 0.6]:
        for overlap_percentage in [0.5, 0.7]:
            print(f'Training window length {win_length} and overlap percentage {overlap_percentage}')
            print('Preparing data...')
            x_train, y_train, x_test, y_test = load_and_prepare_data(base_path, win_length, win_length,
                                                                     overlap_percentage)
            print('Done')
            print('Training model...')
            model = fit_model(x_train, y_train)
            save_model(model, f'.out/svm_win{win_length}_ov{overlap_percentage}.skops')
            evaluate_model(model, x_test, y_test)
            print('Done')
