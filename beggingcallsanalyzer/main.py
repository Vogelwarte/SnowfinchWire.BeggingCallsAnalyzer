import typer
from typing import Literal
from typing_extensions import Annotated

from beggingcallsanalyzer.models.CnnModel import CnnModel
from beggingcallsanalyzer.utilities.exceptions import ArgumentError
from beggingcallsanalyzer.training.trainer import Trainer
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import warnings


app = typer.Typer()

@app.command()
def predict_oss(model_path: Annotated[str, typer.Option(help="")],
            input_directory: Annotated[str, typer.Option(help="")],
            output_directory: Annotated[str, typer.Option(help="")],
            merge_window: Annotated[float, typer.Option(help="")] = 10, 
            cut_length: Annotated[float, typer.Option(help="")] = 2.2, 
            win_length: Annotated[float, typer.Option(help="")] = None,
            hop_length: Annotated[float, typer.Option(help="")] = None, 
            window_type: Annotated[str, typer.Option(help="")] = None, 
            overlap_percentage: Annotated[float, typer.Option(help="")] = None, 
            extension: Annotated[str, typer.Option(help="")] = 'flac', 
            threshold: Annotated[float, typer.Option(help="")] = 0.8,
            processing_batch_size: Annotated[int, typer.Option(help="The number of files that will be predicted simultaneously (the higher the number, the higher the RAM usage)")] = 100,
            inference_batch_size: Annotated[int, typer.Option(help="The number of input samples used  in inference (the higher the number, the higher the VRAM usage)")] = 100):
    model = CnnModel.from_file(model_path, win_length = win_length, hop_length = hop_length,
                                window_type = window_type,
                                percentage_overlap = overlap_percentage)
    warnings.filterwarnings('ignore', ".*keyword argument 'filename' has been renamed to 'path'.*")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        progress.add_task(description="Running prediction...", total=None)
        recordings = list(Path(input_directory).rglob(f'*.{extension}'))
        for i in range(0, len(recordings), processing_batch_size):
            to_idx = min(i + processing_batch_size, len(recordings))
            predictions = model.predict(recordings[i:to_idx], input_directory, merge_window = merge_window, cut_length = cut_length,
                                            extension = extension, threshold=threshold, batch_size=inference_batch_size)
            
            for filename, data in predictions.items():
                data['predictions'].to_csv(f'{output_directory}/{filename.stem}.txt', header = None, index = None, sep = '\t')
    
@app.command()
def predict_fe(model_path: Annotated[str, typer.Option(help="")],
            input_directory: Annotated[str, typer.Option(help="")],
            output_directory: Annotated[str, typer.Option(help="")],
            merge_window: Annotated[float, typer.Option(help="")] = 10, 
            cut_length: Annotated[float, typer.Option(help="")] = 2.2, 
            win_length: Annotated[float, typer.Option(help="")] = None,
            hop_length: Annotated[float, typer.Option(help="")] = None, 
            window_type: Annotated[str, typer.Option(help="")] = None, 
            overlap_percentage: Annotated[float, typer.Option(help="")] = None, 
            extension: Annotated[str, typer.Option(help="")] = 'flac'):
    trainer = Trainer()
    try:
        trainer.predict(model_path, input_directory, output_directory, merge_window, cut_length, win_length, hop_length,
                    window_type, overlap_percentage, extension)
    except ArgumentError as e:
        print(e)


# @app.command()
# def train_evaluate(path = None, show_progressbar = False, merge_window = 10, cut_length = 2.2,
#                     output_path: str = '.out'):
#     pass

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