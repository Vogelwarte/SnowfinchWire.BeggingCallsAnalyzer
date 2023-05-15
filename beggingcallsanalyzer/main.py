import typer
from typing import Literal
from typing_extensions import Annotated

from beggingcallsanalyzer.models.CnnModel import CnnModel
from beggingcallsanalyzer.utilities.exceptions import ArgumentError
from beggingcallsanalyzer.training.trainer import Trainer
from rich.progress import Progress, SpinnerColumn, TextColumn


app = typer.Typer()

@app.command()
def predict_oss(model_path: Annotated[str, typer.Option(help="")],
            files_directory: Annotated[str, typer.Option(help="")],
            merge_window: Annotated[float, typer.Option(help="")] = 10, 
            cut_length: Annotated[float, typer.Option(help="")] = 2.2, 
            win_length: Annotated[float, typer.Option(help="")] = None,
            hop_length: Annotated[float, typer.Option(help="")] = None, 
            window_type: Annotated[Literal['hann', 'hamming'], typer.Option(help="")] = None, 
            overlap_percentage: Annotated[float, typer.Option(help="")] = None, 
            extension: Annotated[str, typer.Option(help="")] = 'flac', 
            threshold: Annotated[float, typer.Option(help="")]=0.8):
    model = CnnModel.from_file(model_path, win_length = win_length, hop_length = hop_length,
                                window_type = window_type,
                                percentage_overlap = overlap_percentage)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="Running prediction...", total=None)
        predictions = model.predict(files_directory, merge_window = merge_window, cut_length = cut_length,
                                    extension = extension, threshold=threshold)
    print("Done!")
    for filename, data in predictions.items():
        data['predictions'].to_csv(f'{filename.parent}/predicted_{filename.stem}.txt', header = None, index = None, sep = '\t')
    
@app.command()
def predict_fe(model_path: Annotated[str, typer.Option(help="")],
            files_directory: Annotated[str, typer.Option(help="")],
            merge_window: Annotated[float, typer.Option(help="")] = 10, 
            cut_length: Annotated[float, typer.Option(help="")] = 2.2, 
            win_length: Annotated[float, typer.Option(help="")] = None,
            hop_length: Annotated[float, typer.Option(help="")] = None, 
            window_type: Annotated[Literal['hann', 'hamming'], typer.Option(help="")] = None, 
            overlap_percentage: Annotated[float, typer.Option(help="")] = None, 
            extension: Annotated[str, typer.Option(help="")] = 'flac'):
    trainer = Trainer()
    try:
        trainer.predict(model_path, files_directory, merge_window, cut_length, win_length, hop_length,
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