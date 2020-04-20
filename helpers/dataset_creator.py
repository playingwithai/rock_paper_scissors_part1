import os
import random
import re
import shutil
import time

import click
import cv2

from helpers.opencv import opencv_video_capture

execution_path = os.getcwd()


class DataSetCreator:
    def __init__(
        self, dataset_name="rock_paper_scissors", train_test_split=0.7, webcam_index=0
    ):
        self.dataset_name = dataset_name
        self.train_test_split = train_test_split
        self.webcam_index = webcam_index

    def _create_dataset_dir(self, path):
        if os.path.exists(path):
            if not click.confirm(f"Do you want to delete the content of {path} dir?"):
                return
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError as e:
                click.echo(f"Error: {path} : { e.strerror}", err=True)
        os.mkdir(path)

    def _get_image_last_index(self, path):
        indexes = [int(re.findall(r"\d+", item)[0]) for item in os.listdir(path)]
        return indexes and max(indexes) + 1 or 0

    def _acquire_image(self, move):
        path = os.path.join(execution_path, self.dataset_name, move)
        counter = self._get_image_last_index(path)
        print(counter)
        with opencv_video_capture(self.webcam_index) as cap:
            while True:
                _, frame = cap.read()
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    cv2.imwrite(
                        os.path.join(path, f"img_{counter}.jpg"), frame,
                    )
                    click.echo(f"Image acquired ({counter})")
                    counter += 1

                elif key == ord("q"):
                    break
                time.sleep(0.2)

    def _create_move_dataset(self):
        click.echo("Every moves should have at least 200 images")
        for move in ("rock", "paper", "scissors"):
            self._create_dataset_dir(
                os.path.join(execution_path, self.dataset_name, move)
            )
            click.echo(f"Take a picture of {move} move by different angles")
            self._acquire_image(move)

    def _split_train_test_dataset(self):
        train_path = os.path.join(execution_path, self.dataset_name, "train")
        test_path = os.path.join(execution_path, self.dataset_name, "test")
        self._create_dataset_dir(train_path)
        self._create_dataset_dir(test_path)

        click.echo("Splitting train and test dataset...")
        for move in ("rock", "paper", "scissors"):
            self._create_dataset_dir(os.path.join(train_path, move))
            self._create_dataset_dir(os.path.join(test_path, move))
            move_path = os.path.join(self.dataset_name, move)
            image_list = os.listdir(move_path)
            train_image_set = set(
                random.sample(
                    image_list, k=int(len(image_list) * self.train_test_split)
                )
            )
            for image in train_image_set:
                shutil.copyfile(
                    os.path.join(move_path, image),
                    os.path.join(train_path, move, image),
                )
            for image in set(image_list) - train_image_set:
                shutil.copyfile(
                    os.path.join(move_path, image), os.path.join(test_path, move, image)
                )

    def _convert_to_grayscale(self):
        for dataset in ("train", "test"):
            base_path = os.path.join(execution_path, self.dataset_name, dataset)
            for move in ["rock", "paper", "scissors"]:
                move_path = os.path.join(base_path, move)
                with click.progressbar(
                    os.listdir(move_path), label=f"Converting {move_path}"
                ) as items:
                    for item in items:
                        item_path = os.path.join(move_path, item)
                        if not os.path.isfile(item_path):
                            continue
                        image = cv2.imread(item_path)
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(item_path, gray_image)

    def _canny_edge_conversion(self, threshold1=100, threshold2=50):
        for dataset in ("train", "test"):
            base_path = os.path.join(execution_path, self.dataset_name, dataset)
            for move in ["rock", "paper", "scissors"]:
                move_path = os.path.join(base_path, move)
                with click.progressbar(
                    os.listdir(move_path), label=f"Canny edge conversion {move_path}"
                ) as items:
                    for item in items:
                        item_path = os.path.join(move_path, item)
                        if not os.path.isfile(item_path):
                            continue
                        image = cv2.imread(item_path)
                        gray_image = cv2.Canny(image, threshold1, threshold2)
                        cv2.imwrite(item_path, gray_image)

    def create_dataset(self):
        # Create dataset directory
        self._create_dataset_dir(self.dataset_name)
        # Create rock, paper, scissors dataset
        self._create_move_dataset()
        # Split train and test dataset
        self._split_train_test_dataset()
        # Optional gray conversion
        # self._convert_to_grayscale()
        # # Optional canny edge conversion
        # self._canny_edge_conversion()
        click.echo("Dataset created")
