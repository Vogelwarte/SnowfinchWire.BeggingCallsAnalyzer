import numpy as np
import pandas as pd
from librosa.feature import spectral_centroid, spectral_rolloff, zero_crossing_rate, rms
from pyACA import FeatureSpectralSpread, FeatureSpectralFlux
from librosa.feature import melspectrogram, mfcc
from librosa.core.spectrum import power_to_db, _spectrogram

from ..utilities.exceptions import ArgumentError


def percentage_overlap(win: pd.Interval, interval: pd.Interval, percentage) -> bool:
    if percentage < 0 or percentage > 1:
        raise ArgumentError(f"Overlap percentage should be between 0 and 1, found {percentage}")

    win_size = win.length * percentage
    return win.overlaps(interval) and \
        (win.left > interval.left and (win.left + win_size) < interval.right or
         (win.right - win_size) > interval.left and win.right < interval.right)


def process_features_classes(df: pd.DataFrame, labels: pd.DataFrame, overlap_percentage, duration, win_length, hop_length) -> pd.DataFrame:
    if win_length <= 0:
        raise ArgumentError('Window length has to be positive.')
    if hop_length <= 0:
        raise ArgumentError('Hop length has to be positive')
    if duration <= 0:
        raise ArgumentError('Duration has to be positive')

    time = np.arange(0, duration + hop_length, hop_length)
    time_interval = pd.arrays.IntervalArray.from_arrays(time, time + win_length)
    df['time'] = time_interval[:len(df)]

    labels['time'] = pd.arrays.IntervalArray.from_arrays(labels['start'].astype('float32'),
                                                         labels['end'].astype('float32'))
    feeding_intervals = labels.loc[labels['label'] == 'feeding', 'time']
    df['y'] = 0
    for i in range(df.shape[0]):
        for interv in feeding_intervals:
            window_interv = df.loc[i, 'time']
            if percentage_overlap(window_interv, interv, overlap_percentage):
                df.loc[i, 'y'] = 1

    return df.drop(columns = ['time'])


def extract_features(data: np.ndarray, sample_rate: int, win_length: float, hop_length: float, window_type='hamming') -> pd.DataFrame:
    if win_length <= 0:
        raise ArgumentError('Window length has to be positive.')
    if hop_length <= 0:
        raise ArgumentError('Hop length has to be positive')

    n_mfcc=13

    win_length = round(win_length*sample_rate)
    hop_length = round(hop_length*sample_rate)
    spec, _ = _spectrogram(y=data, n_fft=win_length, hop_length=hop_length, window=window_type)
    mel_spec = melspectrogram(S=spec, sr=sample_rate)
    mfccs = mfcc(S=power_to_db(mel_spec), n_mfcc=n_mfcc).T

    centroid = spectral_centroid(S=spec, sr=sample_rate).flatten()
    roloff = spectral_rolloff(S=spec, sr=sample_rate).flatten()
    zcr = zero_crossing_rate(data, frame_length=win_length, hop_length=hop_length).flatten()
    energy = rms(S=spec, frame_length=win_length, hop_length=hop_length).flatten()
    spread = FeatureSpectralSpread(spec, sample_rate)
    flux = FeatureSpectralFlux(spec, sample_rate)
    ret_df = pd.DataFrame([*mfccs.T, zcr, energy, centroid, spread, flux , roloff]).T
    ret_df.columns = [*[f'mfcc_{i+1}' for i in range(n_mfcc)], 'zcr', 'energy','spectral_centroid', 'spectral_spread', 'spectral_flux', 'spectral_rolloff']

    return ret_df
