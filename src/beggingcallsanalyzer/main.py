import fire

from src.beggingcallsanalyzer.prediction.prediction import predict
from src.beggingcallsanalyzer.training.persistence import load_model


class Cli:
    def predict(self, model_path, files_directory, window, step):
        model = load_model(model_path)
        predict(model, files_directory, window, step)


    def train(self):
        pass

if __name__ == '__main__':
    fire.Fire(Cli)
