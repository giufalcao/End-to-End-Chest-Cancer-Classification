from src.config.configuration import ConfigurationManager
from src.preprocess.data_ingestion import DataIngestion
from src.logging import logger
from src.constants import constants


class DataIngestionTrainingPipeline:
    """
    Class for orchestrating the data ingestion stage of the training pipeline.
    """

    def __init__(self):
        pass

    def main(self):
        """
        Main method to execute the data ingestion stage.
        """
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            logger.info(f"Stage {constants.STAGE_NAME} completed successfully")
        except Exception as e:
            logger.exception(f"Error occurred in stage {constants.STAGE_NAME}: {e}")
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {constants.STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {constants.STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
