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



