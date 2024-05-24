import os
import zipfile
import gdown
from src.logging import logger
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """
    Class for downloading and extracting data.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize DataIngestion instance.

        Args:
            config (DataIngestionConfig): Configuration for data ingestion.
        """
        self.config = config


    def download_file(self) -> None:
        """
        Download data from the specified URL.
        
        Raises:
            Exception: If downloading fails.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
        except Exception as e:
            logger.error(f"Error occurred while downloading data: {e}")
            raise e


    def extract_zip_file(self) -> None:
        """
        Extract the downloaded zip file.
        
        Raises:
            Exception: If extraction fails.
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
        except Exception as e:
            logger.error(f"Error occurred while extracting zip file: {e}")
            raise e
