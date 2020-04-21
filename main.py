import click

from helpers.dataset_creator import DataSetCreator
from helpers.dataset_downloader import DataSetDownloader
from helpers.move_prediction import RockPaperScissorsPredictor


predictor = RockPaperScissorsPredictor()


def download_dataset():
    dd = DataSetDownloader()
    dd.download_dataset()


def create_dataset():
    dc = DataSetCreator()
    dc.create_dataset()


def train_model():
    predictor.train()


def detect_hand_from_webcam():
    predictor.detect_move_from_webcam()


COMMANDS = {
    1: {"label": "Download dataset", "func": download_dataset},
    2: {"label": "Create/update dataset", "func": create_dataset},
    3: {"label": "Train Model", "func": train_model},
    4: {"label": "Detect move from webcam", "func": detect_hand_from_webcam},
    99: {"label": "Exit"},
}


@click.command()
def cli():
    while True:
        click.echo("Available commands")
        for id_, command_info in COMMANDS.items():
            click.echo(f"{id_}. {command_info['label']}")
        value = click.prompt("Which command do you want to run?", type=int)
        if value == 99:
            break
        command = COMMANDS.get(value)
        if not command:
            click.echo("Input a valid command")
            continue
        command["func"]()


if __name__ == "__main__":
    cli()
