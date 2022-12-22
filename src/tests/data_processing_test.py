import pandas as pd
from beggingcallsanalyzer.training.data_processing import majority_overlap, process_features_classes, extract_features
from beggingcallsanalyzer.common.preprocessing.io import SnowfinchNestRecording
import numpy as np
import pandas.api.types as ptypes

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
    def test_outer_interval_contains_window_returns_true(self):
        outer = pd.Interval(0, 10)
        window = pd.Interval(2, 8)
        assert majority_overlap(window, outer) == True

    def test_outer_interval_left_major_overlap_returns_true(self):
        outer = pd.Interval(0, 7)
        window = pd.Interval(2, 8)
        assert majority_overlap(window, outer) == True

    def test_outer_interval_right_major_overlap_returns_true(self):
        outer = pd.Interval(3, 10)
        window = pd.Interval(2, 8)
        assert majority_overlap(window, outer) == True

    def test_outer_interval_right_minor_overlap_returns_false(self):
        outer = pd.Interval(7, 10)
        window = pd.Interval(2, 8)
        assert majority_overlap(window, outer) == False

    def test_outer_interval_left_minor_overlap_returns_false(self):
        outer = pd.Interval(0, 3)
        window = pd.Interval(2, 8)
        assert majority_overlap(window, outer) == False

    def test_no_overlap_returns_false(self):
        outer = pd.Interval(0, 2)
        window = pd.Interval(4, 6)
        assert majority_overlap(window, outer) == False


class TestProcessingClasses:
    def test_processed_features_contain_class_column(self):
        labels = pd.DataFrame({
            'label': ['feeding', 'feeding'],
            'start': [1.0, 5.0],
            'end': [3.2, 6.1]
        })
        time = np.arange(0, 10, 1)
        window = 1
        df = pd.DataFrame({
            'time': pd.arrays.IntervalArray.from_arrays(time, time+window)
        })
        df = process_features_classes(df, labels)
        assert 'y' in df.columns


    def test_class_column_contains_valid_values(self):
        labels = pd.DataFrame({
            'label': ['feeding', 'feeding'],
            'start': [1.0, 5.0],
            'end': [3.2, 6.1]
        })
        time = np.arange(0, 10, 1)
        window = 1
        df = pd.DataFrame({
            'time': pd.arrays.IntervalArray.from_arrays(time, time+window)
        })
        df = process_features_classes(df, labels)
        assert ptypes.is_integer_dtype(df['y'].dtype)

    def test_class_column_contains_correct_values(self):
        labels = pd.DataFrame({
            'label': ['feeding', 'feeding'],
            'start': [1.0, 5.0],
            'end': [3.2, 6.1]
        })
        time = np.arange(0, 10, 1)
        window = 1
        df = pd.DataFrame({
            'time': pd.arrays.IntervalArray.from_arrays(time, time+window)
        })
        
        result = process_features_classes(df, labels)
        y1 = result[result['y']==1]
        intervals = list(y1['time'])
        assert len(y1) == 3
        assert pd.Interval(1, 2) in intervals
        assert pd.Interval(2, 3) in intervals
        assert pd.Interval(5, 6) in intervals

    def test_extracted_features_appropriate_length(self):
        data = generate_nest_recoring(
			sample_rate = 48000, length_sec = 60, label_count = 7,
			brood_size = 3, brood_age = 10, labels = ['contact', 'feeding']
		)
        window = 0.5
        step = 0.25
        duration = len(data.audio_data) / float(data.audio_sample_rate)
        features = extract_features(data.audio_data, data.audio_sample_rate, window, step, data.labels)

        assert features.shape[0] == duration / step - 1
