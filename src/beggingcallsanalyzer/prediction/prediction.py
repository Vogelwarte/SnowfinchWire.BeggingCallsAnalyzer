from pathlib import Path

import soundfile as sf
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.beggingcallsanalyzer.training.postprocessing import post_process, to_audacity_labels
from src.beggingcallsanalyzer.training.preprocessing import extract_features


def predict(model: Pipeline, directory, window, step, extension = 'flac', merge_window=10, cut_length=2.2):
    files = Path(directory).glob(f'**/*.{extension}')
    flac: Path
    for flac in tqdm(files):
        audio_data, sample_rate = sf.read(flac)
        duration = len(audio_data) / float(sample_rate)
        df = extract_features(audio_data, sample_rate, window, step)
        y_pred = model.predict(df)
        y_processed = post_process(y_pred, window, step, merge_window, cut_length)
        labels = to_audacity_labels(y_processed, duration, window, step)
        labels.to_csv(f'{flac.parent}/{flac.stem}.txt', header=None, index=None, sep='\t')
