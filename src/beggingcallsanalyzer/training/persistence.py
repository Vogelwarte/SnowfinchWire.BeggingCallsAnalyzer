from pathlib import Path

from skops.io import dump, load
from typing import Any, Union


def load_model(path: Union[str, Path]):
    trusted_types = ['numpy.float64']
    model = load(path, trusted=trusted_types)
    return model

def save_model(model: Any, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(exist_ok = True, parents = True)
    dump(model, path)
