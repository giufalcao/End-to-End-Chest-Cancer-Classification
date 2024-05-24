from src.constants import constants
from preprocess.preprocess import DataIngestionTrainingPipeline
from src.logging import logger


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {constants.STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {constants.STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
