from pathlib import Path

from sklearn.svm import SVC
from skops.io import dump, load


def load_model(path: str):
    trusted_types = ['numpy.float64']
    model = load(path, trusted=trusted_types)
    return model

def save_model(model: SVC, path: str) -> None:
    Path(path).parent.mkdir(exist_ok = True, parents = True)
    dump(model, path)
