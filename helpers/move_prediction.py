import logging
import os
from enum import Enum

import click
import cv2
from imageai.Prediction.Custom import CustomImagePrediction, ModelTraining

from helpers.opencv import opencv_video_capture

# Show only errors in console
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class ModelTypeEnum(Enum):
    """
    An helper enum to help for model type choice
    """

    RESNET = 0
    SQEEZENET = 1
    INCEPTIONV3 = 2
    DENSENET = 3


class RockPaperScissorsPredictor:
    """
    This class contains the required code for model training and move prediction using a
    webcam
    """

    MODEL_TYPE_SET_LOOKUP = {
        ModelTypeEnum.RESNET: lambda x: x.setModelTypeAsResNet(),
        ModelTypeEnum.SQEEZENET: lambda x: x.setModelTypeAsSqueezeNet(),
        ModelTypeEnum.INCEPTIONV3: lambda x: x.setModelTypeAsInceptionV3(),
        ModelTypeEnum.DENSENET: lambda x: x.setModelTypeAsDenseNet(),
    }

    def __init__(
        self,
        dataset_name="rock_paper_scissors",
        model_type=ModelTypeEnum.RESNET,
        class_number=3,  # We have 3 different objects: "rock", "paper", "scissors"
    ):
        if not os.path.exists(dataset_name):
            raise Exception("Dataset not found")

        self.dataset_name = dataset_name
        self.model_type = model_type
        self.class_number = class_number
        self.base_path = os.getcwd()

    def _set_proper_model_type(self, model_type, trainer_or_detector):
        self.MODEL_TYPE_SET_LOOKUP[model_type](trainer_or_detector)

    def train(
        self, epochs=100, enhance_data=True, batch_size=16, show_network_summary=True,
    ):
        click.echo("Start training...")
        # Instantiate a ModelTraining object that will be used for model training
        trainer = ModelTraining()
        # Set the model type of the neural network (it must be the same of the
        # prediction)
        self._set_proper_model_type(self.model_type, trainer)
        # Set the path to the data directory
        trainer.setDataDirectory(os.path.join(self.base_path, self.dataset_name))
        # Train the model
        trainer.trainModel(
            num_objects=self.class_number,
            num_experiments=epochs,
            enhance_data=enhance_data,
            batch_size=batch_size,
            show_network_summary=show_network_summary,
        )

    def _get_model_file_name(self, dataset_name):
        # Compose the path to the dataset folder
        models_path = os.path.join(self.base_path, dataset_name, "models")
        # Get the list of the model files from the models folder
        available_models = {
            idx: model_file_name
            for idx, model_file_name in enumerate(os.listdir(models_path))
        }
        chosen_model = None
        while not chosen_model:
            # Print in console available models
            click.echo("Available models:")
            for idx, model_file_name in available_models.items():
                click.echo(f"{idx}. {model_file_name}")
            # ask for a model file to be used for predictions
            model_idx = click.prompt("Which model do you want to use?", type=int)
            chosen_model = available_models.get(model_idx)
        return chosen_model

    def detect_move_from_webcam(
        self,
        model_file_name=None,
        config_file_name="model_class.json",
        webcam_index=0,
        sensibility=80,
    ):
        # Instantiate the CustomImagePrediction object that will predict our moves
        predictor = CustomImagePrediction()
        # Set the model type of the neural network (it must be the same of the training)
        self._set_proper_model_type(self.model_type, predictor)
        if not model_file_name:
            # If model file name is not set, ask user to choose which model must be used
            model_file_name = self._get_model_file_name(self.dataset_name)
        # Set path to the trained model file
        predictor.setModelPath(
            os.path.join(self.base_path, self.dataset_name, "models", model_file_name)
        )
        # Set path to the json file that contains our classes and their labels
        predictor.setJsonPath(
            os.path.join(self.base_path, self.dataset_name, "json", config_file_name)
        )
        # Load the trained model and set it to use "class_number" classes
        predictor.loadModel(num_objects=self.class_number)
        # Context manager that help us to activate/deactivate our webcam
        with opencv_video_capture(webcam_index) as cap:
            # Run until user press "q" key
            while True:
                # Acquire a frame from webcam
                _, frame = cap.read()
                # Do a prediction on that frame
                predictions, probabilities = predictor.predictImage(
                    frame, result_count=3, input_type="array"
                )
                # Get a tuple (class_predicted, probability) that contains the best
                # prediction
                best_prediction = max(
                    zip(predictions, probabilities), key=lambda x: x[1]
                )
                # If probability of the best prediction is >= of the min sensibility
                # required, it writes a label with predicted class name and probability
                # over the frame
                if best_prediction[1] >= sensibility:
                    cv2.putText(
                        frame,
                        f"{best_prediction[0]} ({round(best_prediction[1], 2)})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                # Display the acquired frame on a dedicated window
                cv2.imshow("Move predictor", frame)
                # Break cycle if user press "q" key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
