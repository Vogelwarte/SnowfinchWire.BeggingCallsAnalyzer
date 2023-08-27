import typer
from typer_config import yaml_conf_callback
from typing_extensions import Annotated

from beggingcallsanalyzer.models.CnnModel import CnnModel
from beggingcallsanalyzer.training.trainer import Trainer
from beggingcallsanalyzer.utilities.exceptions import ArgumentError

app = typer.Typer()

@app.command()
def oss(
            win_length: Annotated[float, typer.Option(help="Window length used for FFT")], 
            batch_size: Annotated[int, typer.Option(help="Batch side used in mini-batch neural network learning")], 
            num_workers: Annotated[int, typer.Option(help="How many subprocesses to use for data loading")], 
            epochs: Annotated[int, typer.Option(help="Maximum number of epochs used for learning")], 
            training_data_path: Annotated[str, typer.Option(help="Path to the prepared audio chunks")], 
            output_path: Annotated[str, typer.Option(help="The path for the final model")] = '.out',
            config: Annotated[str, typer.Option(help="Path to a yaml file containing desired options set", callback=yaml_conf_callback, is_eager=True)] = None
        ):
    model = CnnModel(win_length, batch_size, num_workers, epochs)
    model.fit(training_data_path)
    model.save(output_path)
    

@app.command()
def fe(
            training_data_path: Annotated[str, typer.Option(help="Path to the prepared audio chunks")], 
            win_length: Annotated[float, typer.Option(help="Window length used for FFT")],  
            hop_length: Annotated[float, typer.Option(help="Hop length used for FFT")],
            window_type: Annotated[str, typer.Option(help="Window type used for FFT")],
            overlap_percentage: Annotated[float, typer.Option(help="Window length used for FFT")],
            output_path: Annotated[str, typer.Option(help="The path for the final model")] = '.out',
            config: Annotated[str, typer.Option(help="Path to a yaml file containing desired options set", callback=yaml_conf_callback, is_eager=True)] = None,
            show_progressbar = True
        ):
    trainer = Trainer()
    try:
        trainer.train(training_data_path, win_length, hop_length, window_type, overlap_percentage, output_path, show_progressbar)
    except ArgumentError as e:
        print(e)

if __name__ == "__main__":
    app()
