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

app = typer.Typer()

@app.command()
def predict_oss(
            
            model_path: Annotated[str, typer.Option(help="Path to the .model file")],
            input_directory: Annotated[str, typer.Option(help="Path containing all of the input files")],
            output_directory: Annotated[str, typer.Option(help="Path to the directory")],
            config: Annotated[str, typer.Option(help="Path to a yaml file containing desired options set", callback=yaml_conf_callback, is_eager=True)] = None,
            merge_window: Annotated[float, typer.Option(help="Feeding observations at most this close together will be merged into one observation")] = 10, 
            cut_length: Annotated[float, typer.Option(help="Feeding observations at most this long will be discarded")] = 2.2,
            contact_merge_window: Annotated[float, typer.Option(help="Contact observations at most this close together will be merged into one observation")] = 10, 
            contact_cut_length: Annotated[float, typer.Option(help="Contact observations at most this long will be discarded")] = 2.2,
            extension: Annotated[str, typer.Option(help="Extension of the files that are to be evaluated (wihtout .)")] = 'flac', 
            threshold: Annotated[float, typer.Option(help="Decision threshold for feeding classification")] = 0.85,
            contact_threshold: Annotated[float, typer.Option(help="Decision threshold for contact classification")] = 0.887,
            processing_batch_size: Annotated[int, typer.Option(help="The number of files that will be predicted simultaneously (the higher the number, the higher the RAM usage)")] = 100,
            inference_batch_size: Annotated[int, typer.Option(help="The number of input samples used  in inference (the higher the number, the higher the VRAM usage)")] = 100,
            create_plots: Annotated[bool, typer.Option(help="Whether or not to create plots")] = False):
    
    model = CnnModel.from_file(model_path)

    warnings.filterwarnings('ignore', ".*keyword argument 'filename' has been renamed to 'path'.*")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
        output_directory: Path = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        progress.add_task(description="Running prediction...", total=None)
        recordings = list(Path(input_directory).rglob(f'*.{extension}'))

        unexpected_structure = 0
        for rec in recordings:
            if rec.parent.parent.parent != input_directory:
                unexpected_structure += 1
        
        if unexpected_structure:
            progress.print(f"[orange1][Warning]:[/] {unexpected_structure} files were outside of the expected file structure, they will not be included in the summary file")

        df_summary = pd.DataFrame()
        for recordings_chunk in chunked(recordings, processing_batch_size):
            predictions = model.predict(recordings_chunk, merge_window = merge_window, cut_length = cut_length,
                                            threshold=threshold, batch_size=inference_batch_size, contact_cut_length=contact_cut_length, contact_merge_window=contact_merge_window)
            
            summary = create_summary_csv(predictions, output_directory, extension)
            df_summary = pd.concat([df_summary, summary])

            for filename, data in predictions.items():
                output_path = Path(f'{output_directory}/{filename.parent.parent.name}/{filename.parent.name}/')
                output_path.mkdir(parents=True, exist_ok=True)
                data['predictions'].to_csv(output_path/f'{filename.stem}.txt', header = None, index = None, sep = '\t')

    if create_plots:
        plot_feeding_count_hourly(df_summary, output_directory)
        plot_feeding_duration_hourly(df_summary, output_directory)
        plot_feeding_count_daily(df_summary, output_directory)
        plot_feeding_duration_daily(df_summary, output_directory)
    
@app.command()
def predict_fe(            
            model_path: Annotated[str, typer.Option(help="")],
            input_directory: Annotated[str, typer.Option(help="")],
            output_directory: Annotated[str, typer.Option(help="")],
            config: Annotated[str, typer.Option(help="", callback=yaml_conf_callback, is_eager=True)] = None,
            merge_window: Annotated[float, typer.Option(help="")] = 10, 
            cut_length: Annotated[float, typer.Option(help="")] = 2.2, 
            win_length: Annotated[float, typer.Option(help="")] = None,
            hop_length: Annotated[float, typer.Option(help="")] = None, 
            window_type: Annotated[str, typer.Option(help="")] = None, 
            overlap_percentage: Annotated[float, typer.Option(help="")] = None, 
            extension: Annotated[str, typer.Option(help="")] = 'flac',
            create_plots: Annotated[bool, typer.Option(help="Whether or not to create plots")] = False,
            processing_batch_size: Annotated[int, typer.Option(help="The number of files that will be predicted simultaneously (the higher the number, the higher the RAM usage)")] = 100):
    trainer = Trainer()
    try:
        trainer.predict(model_path, input_directory, output_directory, merge_window, cut_length, win_length, hop_length,
                    window_type, overlap_percentage, extension, create_plots=create_plots, processing_batch_size=processing_batch_size)
    except ArgumentError as e:
        print(e)

@app.command()
def train_oss(win_length, batch_size, num_workers, epochs, training_data_path, output_path: str = '.out'):
    model = CnnModel(win_length, batch_size, num_workers, epochs)
    model.fit(training_data_path)
    model.save(output_path)
    

@app.command()
def train_fe(training_data_path, win_length, window_type, overlap_percentage, output_path: str = '.out', show_progressbar = True):
    trainer = Trainer()
    try:
        trainer.train(training_data_path, win_length, window_type, overlap_percentage, output_path, show_progressbar)
    except ArgumentError as e:
        print(e)

def run():
    app()

if __name__ == "__main__":
    app()