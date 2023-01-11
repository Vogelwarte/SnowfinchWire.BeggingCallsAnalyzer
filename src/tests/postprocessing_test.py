import numpy as np

from beggingcallsanalyzer.training.postprocessing import post_process, to_audacity_labels


class TestPostprocessing:
    def test_anomaly_removal(self):
        y_pred = np.array(
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
        )
        y = post_process(y_pred)
        assert np.all(y == np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0]))

    def test_neighbor_joining(self):
        y_pred = np.array(
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]
        )
        y = post_process(y_pred)
        assert np.all(y == np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0]))

    def test_removal_and_joining_order(self):
        y_pred = np.array(
            [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
        )
        y = post_process(y_pred)
        assert np.all(y == np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

    def test_audacity_labels_columns(self):
        y_pred = np.array(
            [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
        )
        duration = 5.5
        step = 0.5
        window = 0.5
        labels = to_audacity_labels(y_pred, duration, window, step)
        assert labels.columns[0] == 'start'
        assert labels.columns[1] == 'end'
        assert labels.columns[2] == 'class'

    def test_audacity_correct_labels(self):
        y_pred = np.array(
            [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
        )
        duration = 5.5
        step = 0.5
        window = 0.5
        labels = to_audacity_labels(y_pred, duration, window, step)
        assert np.all(labels.loc[:, 'class'] == 'feeding')

    def test_audacity_correct_times(self):
        y_pred = np.array(
            [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
        )
        duration = 5.5
        step = 0.5
        window = 0.5
        labels = to_audacity_labels(y_pred, duration, window, step)
        assert len(labels) == 3
        assert labels.loc[0, 'start'] == 1
        assert labels.loc[0, 'end'] == 1.5
        assert labels.loc[1, 'start'] == 2
        assert labels.loc[1, 'end'] == 3
        assert labels.loc[2, 'start'] == 4
        assert labels.loc[2, 'end'] == 4.5