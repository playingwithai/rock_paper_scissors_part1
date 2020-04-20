import os

import click
from imageai.Prediction.Custom import ModelTraining

from helpers.models import ModelTypeEnum, ModelMixin

execution_path = os.getcwd()


class ModelTrainer(ModelMixin):
    def __init__(
        self,
        dataset_name="rock_paper_scissors",
        model_type=ModelTypeEnum.RESNET,
        num_experiments=100,
        enhance_data=True,
        batch_size=16,
        show_network_summary=True,
    ):
        if not os.path.exists(dataset_name):
            raise Exception('Dataset not found. Run "download_dataset" command')

        self.trainer = ModelTraining()
        self._set_proper_model_type(model_type, self.trainer)
        self.trainer.setDataDirectory(os.path.join(execution_path, dataset_name))
        self.num_experiments = num_experiments
        self.enhance_data = enhance_data
        self.batch_size = batch_size
        self.show_network_summary = show_network_summary

    def train(self):
        click.echo("training")
        self.trainer.trainModel(
            num_objects=3,  # We have 3 different objects: "rock", "paper", "scissors"
            num_experiments=self.num_experiments,
            enhance_data=self.enhance_data,
            batch_size=self.batch_size,
            show_network_summary=self.show_network_summary,
        )
