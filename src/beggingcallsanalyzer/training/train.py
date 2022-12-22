from os import getenv
from pathlib import Path

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .data_processing import extract_features
from ..common.preprocessing.io import load_recording_data


def train_model(x_train: pd.DataFrame, y_train: pd.DataFrame, scaler: TransformerMixin = None) -> SVC:
	if scaler is not None:
		x_train = scaler.fit_transform(x_train)
	svm = SVC(C = 20)
	svm.fit(x_train, y_train)
	return svm


def load_and_prepare_date(path: Path, window, step, test_size = 0.1):
	flacs = list(path.glob('**/*.flac'))

	train_paths, test_paths = train_test_split(flacs, test_size = test_size, random_state = 1)
	data = []

	for flac in train_paths:
		file = load_recording_data(flac)
		df = extract_features(file.audio_data, file.audio_sample_rate, window, step, file.labels)
		data.append(df.drop(columns = ['start', 'stop', 'time']))

	train = pd.concat(data, ignore_index = True)
	data.clear()
	for flac in test_paths:
		file = load_recording_data(flac)
		df = extract_features(file.audio_data, file.audio_sample_rate, window, step, file.labels)
		data.append(df.drop(columns = ['start', 'stop', 'time']))
	test = pd.concat(data, ignore_index = True)

	return train.drop(columns = ['y']), train['y'], test.drop(columns = ['y']), test['y']


if __name__ == '__main__':
	base_path = Path(getenv('SNOWFINCH_TRAINING_DATA_PATH'))
	print('Preparing data...')
	x_train, y_train, x_test, y_test = load_and_prepare_date(base_path)
	print('Done')
	print('Training model...')
	train_model(x_train, y_train)
	print('Done')
