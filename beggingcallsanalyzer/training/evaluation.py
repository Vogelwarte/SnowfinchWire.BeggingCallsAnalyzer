import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from beggingcallsanalyzer.training.postprocessing import post_process


def evaluate_model(model: Pipeline, x_test, y_true, win_length, hop_length, merge_window, cut_length) -> tuple[float, np.ndarray]:
    y_pred = model.predict(x_test)
    y_pred = post_process(y_pred, win_length, hop_length, merge_window, cut_length)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, cm