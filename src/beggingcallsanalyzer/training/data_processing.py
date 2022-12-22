import numpy as np
import pandas as pd
from pyAudioAnalysis import ShortTermFeatures as aF


def majority_overlap(win: pd.Interval, interval: pd.Interval) -> bool:
    return win.overlaps(interval) and \
        (win.left > interval.left and win.mid < interval.right or
         win.mid > interval.left and win.right < interval.right)


def process_features_classes(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels['time'] = pd.arrays.IntervalArray.from_arrays(labels['start'].astype('float32'),
                                                         labels['end'].astype('float32'))
    feeding_intervals = labels.loc[labels['label'] == 'feeding', 'time']
    df['y'] = 0
    for i in range(df.shape[0]):
        for interv in feeding_intervals:
            window_interv: pd.Interval = df.loc[i, 'time']
            if majority_overlap(window_interv, interv):
                df.loc[i, 'y'] = 1

    return df


def extract_features(audio: np.ndarray, sample_rate: int, window: float, step: float,
                     labels: pd.DataFrame) -> pd.DataFrame:
    duration = len(audio) / float(sample_rate)

    features, feature_names = aF.feature_extraction(audio, sample_rate, 
    int(sample_rate * window), 
    int(sample_rate * step))

    df = pd.DataFrame(data = features.T, columns = feature_names)
    time = np.arange(0, duration - step, step)
    df['time'] = pd.arrays.IntervalArray.from_arrays(time, time + window)

    ret_df = process_features_classes(df, labels)

    return ret_df
