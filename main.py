import click

from helpers.dataset_creator import DataSetCreator
from helpers.dataset_downloader import DataSetDownloader
from helpers.hand_detector import HandDetector
from helpers.model_trainer import ModelTrainer


def download_dataset():
    dd = DataSetDownloader()
    dd.download_dataset()


def create_dataset():
    dc = DataSetCreator(webcam_index=1)
    dc.create_dataset()


def train_model():
    mt = ModelTrainer()
    mt.train()


def detect_hand_from_webcam():
    hd = HandDetector()
    hd.detect_from_webcam(webcam_index=1)


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
