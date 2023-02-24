import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest
from beggingcallsanalyzer.common.preprocessing.io import SnowfinchNestRecording
from beggingcallsanalyzer.training.preprocessing import percentage_overlap, process_features_classes, extract_features
from beggingcallsanalyzer.utilities.exceptions import ArgumentError


def generate_audio(sample_rate: int, length_sec: int) -> np.ndarray:
    return np.random.random(sample_rate * length_sec) * 2.0 - 1.0


def generate_labels(start: float, end: float, count: int, labels: list[str]) -> pd.DataFrame:
    max_label_length_sec = (end - start) / count

    label_starts = np.linspace(start, end - max_label_length_sec, num = count)
    label_ends = label_starts + np.random.random(count) * max_label_length_sec
    label_indices = np.random.randint(0, len(labels), count)

    return pd.DataFrame(
        data = {
            'start': label_starts,
            'end': label_ends,
            'label': np.array(labels)[label_indices]
        }
    )


def generate_nest_recoring(
        sample_rate: int, length_sec: int, label_count: int,
        brood_age: int, brood_size: int, labels: list[str]
) -> SnowfinchNestRecording:
    audio = generate_audio(sample_rate, length_sec)
    labels = generate_labels(0.0, length_sec, label_count, labels)
    return SnowfinchNestRecording(audio, sample_rate, labels, brood_age, brood_size)


class TestOverlappingIntervals:
    @pytest.mark.parametrize("percentage", [0, 0.1, 0.3, 0.5, 0.7, 1])
    def test_outer_interval_contains_window_returns_true(self, percentage):
        outer = pd.Interval(0, 10)
        window = pd.Interval(2, 8)
        assert percentage_overlap(window, outer, percentage) is True

    @pytest.mark.parametrize("percentage", [0, 0.1, 0.3, 0.5, 0.8])
    def test_outer_interval_left_major_overlap_returns_true(self, percentage):
        outer = pd.Interval(0, 7)
        window = pd.Interval(2, 8)
        assert percentage_overlap(window, outer, percentage) is True

    @pytest.mark.parametrize("percentage", [0, 0.1, 0.3, 0.5, 0.8])
    def test_outer_interval_right_major_overlap_returns_true(self, percentage):
        outer = pd.Interval(3, 10)
        window = pd.Interval(2, 8)
        assert percentage_overlap(window, outer, percentage) is True

    @pytest.mark.parametrize("percentage", [0.4, 0.7, 0.8, 0.9])
    def test_outer_interval_right_minor_overlap_returns_false(self, percentage):
        outer = pd.Interval(7, 10)
        window = pd.Interval(2, 8)
        assert percentage_overlap(window, outer, percentage) is False

    @pytest.mark.parametrize("percentage", [0.2, 0.4, 0.7, 0.8, 0.9])
    def test_outer_interval_left_minor_overlap_returns_false(self, percentage):
        outer = pd.Interval(0, 3)
        window = pd.Interval(2, 8)
        assert percentage_overlap(window, outer, percentage) is False

    @pytest.mark.parametrize("percentage", [0, 0.1, 0.3, 0.5, 0.7, 1])
    def test_no_overlap_returns_false(self, percentage):
        outer = pd.Interval(0, 2)
        window = pd.Interval(4, 6)
        assert percentage_overlap(window, outer, percentage) is False

    def test_incorrect_percentage(self):
        outer = pd.Interval(0, 2)
        window = pd.Interval(4, 6)
        with pytest.raises(ArgumentError, match = "Overlap percentage should be between 0 and 1, found 15"):
            percentage_overlap(window, outer, 15)


@pytest.fixture
def feature_classes_processing_input():
    labels = pd.DataFrame({
        'label': ['feeding', 'feeding'],
        'start': [1.0, 5.0],
        'end': [3.2, 6.1]
    })
    duration = 11
    window = 1
    time = np.arange(0, duration - window, window)
    overlap_percentage = 0.7
    df = pd.DataFrame({
        'time': pd.arrays.IntervalArray.from_arrays(time, time + window)
    })
    return labels, duration, window, overlap_percentage, df


@pytest.fixture
def extract_features_input():
    data = generate_nest_recoring(
        sample_rate = 48000, length_sec = 60, label_count = 7,
        brood_size = 3, brood_age = 10, labels = ['contact', 'feeding']
    )
    window = 0.5
    hop = 0.25
    return data, window, hop


class TestProcessingClasses:
    def test_processed_features_contain_class_column(self, feature_classes_processing_input):
        labels, duration, window, overlap_percentage, df = feature_classes_processing_input

        df = process_features_classes(df, labels, overlap_percentage, duration, window, window)
        assert 'y' in df.columns

    def test_processed_features_negative_duration_should_throw(self, feature_classes_processing_input):
        labels, _, window, overlap_percentage, df = feature_classes_processing_input
        duration = -2

        with pytest.raises(ArgumentError, match = 'Duration has to be positive'):
            process_features_classes(df, labels, overlap_percentage, duration, window, window)

    def test_processed_features_negative_window_should_throw(self, feature_classes_processing_input):
        labels, duration, hop, overlap_percentage, df = feature_classes_processing_input
        window = -2

        with pytest.raises(ArgumentError, match = 'Window.* has to be positive'):
            process_features_classes(df, labels, overlap_percentage, duration, window, hop)

    def test_processed_features_negative_hop_should_throw(self, feature_classes_processing_input):
        labels, duration, window, overlap_percentage, df = feature_classes_processing_input
        hop = -2

        with pytest.raises(ArgumentError, match = 'Hop.* has to be positive'):
            process_features_classes(df, labels, overlap_percentage, duration, window, hop)

    def test_processed_features_column_contains_valid_values(self, feature_classes_processing_input):
        labels, duration, window, overlap_percentage, df = feature_classes_processing_input
        df = process_features_classes(df, labels, overlap_percentage, duration, window, window)
        assert ptypes.is_integer_dtype(df['y'].dtype)

    def test_extracted_features_negative_window_should_throw(self, extract_features_input):
        data, _, step = extract_features_input
        window = -2

        with pytest.raises(ArgumentError, match = 'Window.* has to be positive'):
            extract_features(data.audio_data, data.audio_sample_rate, window, step)

    def test_extracted_features_negative_hop_should_throw(self, extract_features_input):
        data, window, _ = extract_features_input
        step = -2

        with pytest.raises(ArgumentError, match = 'Hop.* has to be positive'):
            extract_features(data.audio_data, data.audio_sample_rate, window, step)

    def test_extracted_features_appropriate_total_length(self, extract_features_input):
        data, window, step = extract_features_input
        duration = len(data.audio_data) / float(data.audio_sample_rate)
        features = extract_features(data.audio_data, data.audio_sample_rate, window, step)

        assert features.shape[0] == duration / step + 1

    def test_extracted_features_contain_all_columns(self, extract_features_input):
        data, window, step = extract_features_input
        features = extract_features(data.audio_data, data.audio_sample_rate, window, step)
        assert set(features.columns) == {*[f'mfcc_{i + 1}' for i in range(13)], 'zcr', 'energy',
                                         'spectral_centroid', 'spectral_spread', 'spectral_flux',
                                         'spectral_rolloff'}
