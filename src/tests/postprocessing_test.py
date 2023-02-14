import numpy as np
import pytest

from beggingcallsanalyzer.training.postprocessing import post_process, to_audacity_labels
from beggingcallsanalyzer.utilities.exceptions import ArgumentError


@pytest.fixture
def to_audacity_labels_sample_data():
    y_pred = np.array(
        [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
    )
    duration = 5.5
    step = 0.5
    window = 0.5
    return y_pred, duration, step, window


@pytest.fixture
def post_process_sample_data():
    y_pred = np.array(
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    )
    win_length = 1
    hop_length = 1
    merge_window = 1
    cut_length = 1
    return y_pred, win_length, hop_length, merge_window, cut_length


class TestPostprocessing:
    def test_post_process_anomaly_removal(self):
        y_pred = np.array(
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
        )
        win_length = 1
        hop_length = 1
        merge_window = 2
        cut_length = 1
        y = post_process(y_pred, win_length, hop_length, merge_window, cut_length)
        assert np.all(y == np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0]))

    def test_post_process_neighbor_joining(self):
        y_pred = np.array(
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0]
        )
        win_length = 1
        hop_length = 1
        merge_window = 2
        cut_length = 1
        y = post_process(y_pred, win_length, hop_length, merge_window, cut_length)
        assert np.all(y == np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0]))

    def test_post_process_removal_and_joining_order(self):
        y_pred = np.array(
            [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0]
        )
        win_length = 1
        hop_length = 1
        merge_window = 2
        cut_length = 1
        y = post_process(y_pred, win_length, hop_length, merge_window, cut_length)
        assert np.all(y == np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

    def test_post_process_negative_window_should_throw(self, post_process_sample_data):
        y_pred, _, hop_length, merge_window, cut_length = post_process_sample_data
        win_length = -2
        with pytest.raises(ArgumentError, match = 'Window length has to be positive'):
            post_process(y_pred, win_length, hop_length, merge_window, cut_length)

    def test_post_process_negative_hop_should_throw(self, post_process_sample_data):
        y_pred, win_length, _, merge_window, cut_length = post_process_sample_data
        hop_length = -2
        with pytest.raises(ArgumentError, match = 'Hop length has to be positive'):
            post_process(y_pred, win_length, hop_length, merge_window, cut_length)

    def test_post_process_negative_merge_window_should_throw(self, post_process_sample_data):
        y_pred, win_length, hop_length, _, cut_length = post_process_sample_data
        merge_window = -2
        with pytest.raises(ArgumentError, match = 'Maximum merge distance has to be positive'):
            post_process(y_pred, win_length, hop_length, merge_window, cut_length)

    def test_post_process_negative_cut_length_should_throw(self, post_process_sample_data):
        y_pred, win_length, hop_length, merge_window, _ = post_process_sample_data
        cut_length = -2
        with pytest.raises(ArgumentError, match = 'Maximum cut length has to be positive'):
            post_process(y_pred, win_length, hop_length, merge_window, cut_length)

    def test_audacity_labels_columns(self, to_audacity_labels_sample_data):
        y_pred, duration, step, window = to_audacity_labels_sample_data
        labels = to_audacity_labels(y_pred, duration, window, step)
        assert labels.columns[0] == 'start'
        assert labels.columns[1] == 'end'
        assert labels.columns[2] == 'class'

    def test_audacity_correct_labels(self, to_audacity_labels_sample_data):
        y_pred, duration, step, window = to_audacity_labels_sample_data
        labels = to_audacity_labels(y_pred, duration, window, step)
        assert np.all(labels.loc[:, 'class'] == 'feeding')

    def test_audacity_correct_times(self, to_audacity_labels_sample_data):
        y_pred, duration, step, window = to_audacity_labels_sample_data
        labels = to_audacity_labels(y_pred, duration, window, step)
        assert len(labels) == 3
        assert labels.loc[0, 'start'] == 1
        assert labels.loc[0, 'end'] == 1.5
        assert labels.loc[1, 'start'] == 2
        assert labels.loc[1, 'end'] == 3
        assert labels.loc[2, 'start'] == 4
        assert labels.loc[2, 'end'] == 4.5

    def test_to_audacity_labels_negative_hop_should_throw(self, to_audacity_labels_sample_data):
        y_pred, duration, _, window = to_audacity_labels_sample_data
        hop = -2
        with pytest.raises(ArgumentError, match = 'Hop length has to be positive'):
            to_audacity_labels(y_pred, duration, window, hop)

    def test_to_audacity_labels_negative_window_should_throw(self, to_audacity_labels_sample_data):
        y_pred, duration, hop, _ = to_audacity_labels_sample_data
        window = -2
        with pytest.raises(ArgumentError, match = 'Window length has to be positive'):
            to_audacity_labels(y_pred, duration, window, hop)
