import os
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from more_itertools import chunked
from rich.progress import Progress, SpinnerColumn, TextColumn
from typer_config import yaml_conf_callback
from typing_extensions import Annotated

from beggingcallsanalyzer.console import console
from beggingcallsanalyzer.models.CnnModel import CnnModel
from beggingcallsanalyzer.plotting.plotting import (
    plot_feeding_count_daily, plot_feeding_count_hourly,
    plot_feeding_duration_daily, plot_feeding_duration_hourly)
from beggingcallsanalyzer.plotting.summary import create_summary_csv
from beggingcallsanalyzer.training.trainer import Trainer
from beggingcallsanalyzer.utilities.exceptions import ArgumentError
from beggingcallsanalyzer.utilities.validators import is_valid_audio

app = typer.Typer()

@app.command()
def oss(
            model_path: Annotated[str, typer.Option(help="Path to the .model file")],
            input_directory: Annotated[str, typer.Option(help="Path containing all of the input files")],
            output_directory: Annotated[str, typer.Option(help="Path to the output directory")],
            config: Annotated[str, typer.Option(help="Path to a yaml file containing desired options set", callback=yaml_conf_callback, is_eager=True)] = None,
            merge_window: Annotated[float, typer.Option(help="Feeding observations at most this close together will be merged into one observation")] = 3, 
            cut_length: Annotated[float, typer.Option(help="Feeding observations at most this long will be discarded")] = 2,
            contact_merge_window: Annotated[float, typer.Option(help="Contact observations at most this close together will be merged into one observation")] = 10, 
            contact_cut_length: Annotated[float, typer.Option(help="Contact observations at most this long will be discarded")] = 2,
            extension: Annotated[str, typer.Option(help="Extension of the files that are to be evaluated (wihtout .)")] = 'flac', 
            threshold: Annotated[float, typer.Option(help="Decision threshold for feeding classification")] = 0.85,
            contact_threshold: Annotated[float, typer.Option(help="Decision threshold for contact classification")] = 0.887,
            processing_batch_size: Annotated[int, typer.Option(help="The number of files that will be predicted simultaneously (the higher the number, the higher the RAM usage)")] = 100,
            inference_batch_size: Annotated[int, typer.Option(help="The number of input samples used  in inference (the higher the number, the higher the VRAM usage)")] = 100,
            create_plots: Annotated[bool, typer.Option(help="Whether or not to create plots")] = False
        ):    
    model = CnnModel.from_file(model_path)

    warnings.filterwarnings('ignore', ".*keyword argument 'filename' has been renamed to 'path'.*")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
        output_directory: Path = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        progress.add_task(description="Running prediction...", total=None)
        input_directory_path = Path(input_directory)
        recordings = list(input_directory_path.rglob(f'*.{extension}'))

        df_summary = pd.DataFrame()
        incorrect_folder_structure = False
        for recordings_chunk in chunked(recordings, processing_batch_size):
            valid_recordings = [path for path in recordings_chunk if is_valid_audio(path)]
            predictions = model.predict(valid_recordings, merge_window = merge_window, cut_length = cut_length,
                                            threshold=threshold, batch_size=inference_batch_size, contact_cut_length=contact_cut_length, contact_merge_window=contact_merge_window,
                                            contact_threshold=contact_threshold)
            
            if not incorrect_folder_structure and (bad_files := [f.name for f in filter(lambda k: k.parent.parent.parent != input_directory_path, predictions.keys())]):
                progress.print(f'[orange1][Warning]:[/] Files {", ".join(bad_files)} are in an unexpected place in the directory structure. All predictions will be places in the root of the output directory, summary and plots will not be generated')
                incorrect_folder_structure = True
                output_path = output_directory

            if not incorrect_folder_structure:
                summary = create_summary_csv(predictions, output_directory, extension)
                df_summary = pd.concat([df_summary, summary])

            for filename, data in predictions.items():
                if not incorrect_folder_structure:
                    output_path = Path(f'{output_directory}/{filename.parent.parent.name}/{filename.parent.name}/')
                output_path.mkdir(parents=True, exist_ok=True)
                data['predictions'].to_csv(output_path/f'{filename.stem}.txt', header = None, index = None, sep = '\t')

    if create_plots and not incorrect_folder_structure:
        summary_path = f'{output_directory}/summary.csv'
        df_summary.to_csv(summary_path, index=None, header=not os.path.exists(summary_path))
        plot_feeding_count_hourly(df_summary, output_directory)
        plot_feeding_duration_hourly(df_summary, output_directory)
        plot_feeding_count_daily(df_summary, output_directory)
        plot_feeding_duration_daily(df_summary, output_directory)
    

@app.command()
def fe(            
            model_path: Annotated[str, typer.Option(help="Path to the .skops file")],
            input_directory: Annotated[str, typer.Option(help="Path containing all of the input files")],
            output_directory: Annotated[str, typer.Option(help="Path to the output directory")],
            config: Annotated[str, typer.Option(help="", callback=yaml_conf_callback, is_eager=True)] = None,
            merge_window: Annotated[float, typer.Option(help="Frames at most this close together will be merged into one observation")] = 3, 
            cut_length: Annotated[float, typer.Option(help="Observations at most this long will be discarded")] = 2, 
            win_length: Annotated[Optional[float], typer.Option(help="Feature extraction frame length (will prioritize the one saved in the model)")] = 0.5,
            hop_length: Annotated[Optional[float], typer.Option(help="Feature extraction hop length (will prioritize the one saved in the model)")] = 0.5, 
            window_type: Annotated[Optional[str], typer.Option(help="Windowing function type (will prioritize the one saved in the model)")] = "hamming",
            extension: Annotated[str, typer.Option(help="Extension of the files that are to be evaluated (wihtout .)")] = 'flac',
            create_plots: Annotated[bool, typer.Option(help="Whether or not to create plots")] = False,
            processing_batch_size: Annotated[int, typer.Option(help="The number of files that will be predicted simultaneously (the higher the number, the higher the RAM usage)")] = 100
        ):
    trainer = Trainer()
    warnings.filterwarnings('ignore', ".*Trying to unpickle estimator.*")
    try:
        trainer.predict(model_path, input_directory, output_directory, merge_window, cut_length, win_length, hop_length,
                    window_type, extension, create_plots=create_plots, processing_batch_size=processing_batch_size)
    except ArgumentError as e:
        print(e)


if __name__ == "__main__":
    app()
