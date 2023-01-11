import numpy as np
import pandas as pd
from pyAudioAnalysis import ShortTermFeatures as aF
from sklearn.preprocessing import MinMaxScaler
from spafe.features.bfcc import bfcc
from spafe.utils.preprocessing import SlidingWindow


def percentage_overlap(win: pd.Interval, interval: pd.Interval, percentage) -> bool:
    if percentage < 0 or percentage > 1:
        raise ValueError(f"Percentage should be between 0 and 1, found {percentage}")

    win_size = win.length * percentage
    return win.overlaps(interval) and \
        (win.left > interval.left and (win.left + win_size) < interval.right or
         (win.right - win_size) > interval.left and win.right < interval.right)


def process_features_classes(df: pd.DataFrame, labels: pd.DataFrame, overlap_percentage, duration, window, step) -> pd.DataFrame:
    time = np.arange(0, duration, step)
    time_interval = pd.arrays.IntervalArray.from_arrays(time, time + window)
    df['time'] = time_interval[:len(df)]

    labels['time'] = pd.arrays.IntervalArray.from_arrays(labels['start'].astype('float32'),
                                                         labels['end'].astype('float32'))
    feeding_intervals = labels.loc[labels['label'] == 'feeding', 'time']
    df['y'] = 0
    for i in range(df.shape[0]):
        for interv in feeding_intervals:
            window_interv: pd.Interval = df.loc[i, 'time']
            if percentage_overlap(window_interv, interv, overlap_percentage):
                df.loc[i, 'y'] = 1

    return df.drop(columns = ['time'])


def extract_features(audio: np.ndarray, sample_rate: int, window: float, step: float) -> pd.DataFrame:
    bfccs = bfcc(audio,
                 fs = sample_rate,
                 window = SlidingWindow(window, step, "hamming")
                 )
    scaled_bfccs = MinMaxScaler().fit_transform(bfccs)
    bfccs_df = pd.DataFrame(scaled_bfccs, columns = [f'bfcc_{i}' for i in range(bfccs.shape[1])])

    sp_features, feature_names = aF.feature_extraction(audio, sample_rate,
                                                       int(sample_rate * window),
                                                       int(sample_rate * step))
    scaled_sp_features = MinMaxScaler().fit_transform(sp_features.T)

    sp_ft_df = pd.DataFrame(data = scaled_sp_features, columns = feature_names)

    ret_df = bfccs_df.join(sp_ft_df.loc[:,
                  ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy',
                   'spectral_flux', 'spectral_rolloff']])
    return ret_df
