import os
import time
import zipfile

import click
import requests

execution_path = os.getcwd()


class DataSetDownloader:
    """
    Rock paper scissors dataset is offered
    by Laurence Moroney (lmoroney@gmail.com) at
    http://www.laurencemoroney.com/rock-paper-scissors-dataset/
    """

    def __init__(
        self,
        dataset_name="rock_paper_scissors",
        train_dataset_url="https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip",
        test_dataset_url="https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip",
        chunk_size=4096,
    ):

        if not (train_dataset_url and train_dataset_url):
            raise Exception("Train and test dataset urls are required")

        self.dataset_name = dataset_name
        self.train_dataset_url = train_dataset_url
        self.test_dataset_url = test_dataset_url
        self.train_dataset_name = os.path.basename(self.train_dataset_url)
        self.test_dataset_name = os.path.basename(self.test_dataset_url)
        self.chunk_size = chunk_size

    def _get_dataset_file(self, url, filename):
        try:
            with open(filename, "wb") as f:
                res = requests.get(url, stream=True)
                res.raise_for_status()
                file_size = res.headers.get("content-length")

                if file_size is None:
                    f.write(res.content)
                else:
                    with click.progressbar(
                        length=int(file_size), label=f"Downloading {filename}"
                    ) as bar:
                        for data in res.iter_content(chunk_size=self.chunk_size):
                            f.write(data)
                            bar.update(len(data))
        except FileExistsError:
            click.echo(f'Error: a file named "{filename}" already exists')

    def _get_dataset(self):
        self._get_dataset_file(self.train_dataset_url, self.train_dataset_name)
        click.echo("Train dataset downloaded")
        self._get_dataset_file(self.test_dataset_url, self.test_dataset_name)
        click.echo("Test dataset downloaded")

    def _unzip_archive(self, zip_filename):
        with zipfile.ZipFile(zip_filename, "r") as z:
            with click.progressbar(
                z.infolist(), label=f"Extracting {zip_filename}"
            ) as files:
                time.sleep(1)  # A little hacks to see the progress bar for little sets
                for f in files:
                    z.extract(f, os.path.join(execution_path, self.dataset_name))
        click.echo("Dataset extracted")

    def _unzip_archives(self):
        self._unzip_archive(self.train_dataset_name)
        self._unzip_archive(self.test_dataset_name)

    def _create_dataset_dir(self):
        if not os.path.exists(self.dataset_name):
            os.mkdir(self.dataset_name)
            click.echo("Dataset dir created")

    def _rename_dataset_dirs(self):
        try:
            os.rename(
                os.path.join(
                    execution_path,
                    self.dataset_name,
                    os.path.splitext(self.train_dataset_name)[0],
                ),
                os.path.join(execution_path, self.dataset_name, "train"),
            )
            click.echo("Renamed train dir")
        except FileExistsError:
            click.echo("train dir already exists")
        try:
            os.rename(
                os.path.join(
                    execution_path,
                    self.dataset_name,
                    os.path.splitext(self.test_dataset_name)[0],
                ),
                os.path.join(execution_path, self.dataset_name, "test"),
            )
            click.echo("Renamed test dir")
        except FileExistsError:
            click.echo("test dir already exists")

    def _delete_archives(self):
        try:
            os.remove(os.path.join(execution_path, self.train_dataset_name))
        except FileNotFoundError:
            pass
        try:
            os.remove(os.path.join(execution_path, self.test_dataset_name))
        except FileNotFoundError:
            pass
        click.echo("Archives removed")

    def download_dataset(self):
        # Create dataset directory
        self._create_dataset_dir()
        # Download dataset archives
        self._get_dataset()
        # Unzip archives
        self._unzip_archives()
        # Rename dataset dirs to "train" and "test"
        self._rename_dataset_dirs()
        # Delete archives
        self._delete_archives()
        click.echo("Done")
