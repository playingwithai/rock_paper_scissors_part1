import os
import time

import click
import cv2
from imageai.Prediction.Custom import CustomImagePrediction

from helpers.models import ModelMixin, ModelTypeEnum
from helpers.opencv import opencv_video_capture


execution_path = os.getcwd()


class HandDetector(ModelMixin):
    def __init__(
        self,
        dataset_name="rock_paper_scissors",
        model_file_name=None,
        config_file_name="model_class.json",
        model_type=ModelTypeEnum.RESNET,
        num_objects=3,
        frame_sleep=0.2,
    ):
        self.frame_sleep = frame_sleep
        self.detector = CustomImagePrediction()
        self._set_proper_model_type(model_type, self.detector)
        if not model_file_name:
            model_file_name = self._get_model_file_name(dataset_name)
        self.detector.setModelPath(
            os.path.join(execution_path, dataset_name, "models", model_file_name)
        )
        self.detector.setJsonPath(
            os.path.join(execution_path, dataset_name, "json", config_file_name)
        )
        self.detector.loadModel(num_objects=num_objects)

    def _get_model_file_name(self, dataset_name):
        models_path = os.path.join(execution_path, dataset_name, "models")
        available_models = {
            idx: model_file_name
            for idx, model_file_name in enumerate(os.listdir(models_path))
        }
        chosen_model = None
        while not chosen_model:
            click.echo("Available models:")
            for idx, model_file_name in available_models.items():
                click.echo(f"{idx}. {model_file_name}")

            model_idx = click.prompt("Which model do you want to use?", type=int)
            chosen_model = available_models.get(model_idx)
        return chosen_model

    def detect_from_webcam(self, webcam_index=0, sensibility=80):
        with opencv_video_capture(webcam_index) as cap:
            while True:
                ret, frame = cap.read()
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame = frame[210:-210, 30:-150]
                predictions, probabilities = self.detector.predictImage(
                    frame, result_count=3, input_type="array"
                )
                best_prediction = max(
                    zip(predictions, probabilities), key=lambda x: x[1]
                )
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
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(self.frame_sleep)
