import re
from pathlib import Path
from typing import Any, Union

from skops.io import dump, load


def load_model(path: Union[str, Path]) -> tuple[Any, float, float, str]:
    trusted_types = ['numpy.float64']
    typed_path = path if isinstance(path, Path) else Path(path)
    m = re.match(
        'svm_win_len_(?P<win_len>1\.0|0.\d+)_overlap_(?P<overlap>1.0|0\.\d+)_win_type_(?P<win_type>.*?)\..*',
        typed_path.name)
    win_length = float(m.group('win_len'))
    overlap_percentage = float(m.group('overlap'))
    win_type = m.group('win_type')

    model = load(path, trusted = trusted_types)
    return model, win_length, overlap_percentage, win_type


def save_model(model: Any, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(exist_ok = True, parents = True)
    dump(model, path)
