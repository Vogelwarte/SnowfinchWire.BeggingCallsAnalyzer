import soundfile as sf
from soundfile import SoundFileError
from typing import Union
from pathlib import Path

def is_valid_audio(path: Union[str, Path]) -> bool:
    try:
        sf.read(path)
        return True
    except SoundFileError:
        return False
