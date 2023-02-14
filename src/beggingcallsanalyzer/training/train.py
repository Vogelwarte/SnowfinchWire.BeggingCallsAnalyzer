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


def fit_model(x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series]) -> Pipeline:
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svc', SVC(C = 20, cache_size = 2000))
    ])
    pipe.fit(x_train, y_train)
    return pipe


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


def clean_output_directory(path: Path):
    for path in path.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)
