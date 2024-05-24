import os
from pathlib import Path
from constants.path_conf import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import DataIngestionConfig

class ConfigurationManager:
    """Class to manage configuration settings."""

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        """
        Initialize ConfigurationManager.

        Args:
        - config_filepath (str): Filepath of the configuration file.
        - params_filepath (str): Filepath of the parameters file.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieve DataIngestionConfig from the configuration.

        Returns:
        - DataIngestionConfig: Configuration for data ingestion.
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
