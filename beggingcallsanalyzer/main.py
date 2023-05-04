from dataclasses import dataclass
from os import getenv
from pathlib import Path

import fire
import pandas as pd

from beggingcallsanalyzer.models.SvmModel import SvmModel
from beggingcallsanalyzer.models.CnnModel import CnnModel
from beggingcallsanalyzer.training.evaluation import evaluate_model
from beggingcallsanalyzer.training.persistence import save_model
from beggingcallsanalyzer.training.postprocessing import to_audacity_labels
from beggingcallsanalyzer.training.train import clean_output_directory, load_and_prepare_data, fit_model
from beggingcallsanalyzer.utilities.exceptions import ArgumentError

class Cli:
    def predict(self, model_path, files_directory, merge_window = 10, cut_length = 2.2, win_length = None,
                hop_length = None, window_type = None, overlap_percentage = None, extension = 'flac', threshold=0.8):
        model = CnnModel.from_file(model_path, win_length = win_length, hop_length = hop_length,
                                   window_type = window_type,
                                   percentage_overlap = overlap_percentage)

        # try:
        predictions = model.predict(files_directory, merge_window = merge_window, cut_length = cut_length,
                                    extension = extension)
        try:
            for filename, data in predictions.items():
                data['predictions'].to_csv(f'{filename.parent}/predicted_{filename.stem}.txt', header = None, index = None, sep = '\t')
        except ArgumentError as e:
            print(e)
            print('Quitting...')
            return

    def train_evaluate(self, path = None, show_progressbar = False, merge_window = 10, cut_length = 2.2,
                       output_path: str = '.out'):
        pass


    def train(self, training_data_path, win_length, window_type, overlap_percentage, output_path: str = '.out',
              show_progressbar = True):
        pass


def run():
    fire.Fire(Cli)
