from sklearn.svm import SVC
from skops.io import dump, load


def load_model(path: str):
    trusted_types = ['numpy.float64']
    model = load(path, trusted=trusted_types)
    return model

def save_model(model: SVC, path: str) -> None:
    dump(model, path)
