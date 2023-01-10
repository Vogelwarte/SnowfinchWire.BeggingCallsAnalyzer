from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.svm import SVC

from src.beggingcallsanalyzer.training.postprocessing import post_process


def evaluate_model(model: SVC, x_test, y_true):
    y_pred = model.predict(x_test)
    y_pred = post_process(y_pred)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
