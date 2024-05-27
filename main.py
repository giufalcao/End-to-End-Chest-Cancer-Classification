from src.constants import constants
from preprocess.preprocess import DataIngestionTrainingPipeline
from models.base_model_training import BaseModelTrainingPipeline
from training.training_pipeline import ModelTrainingPipeline
from src.logging import logger


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {constants.DATA_INGESTION_STEP} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {constants.DATA_INGESTION_STEP} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    try: 
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {constants.PREPARE_BASE_MODEL_STEP} started <<<<<<")
        prepare_base_model = BaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>> stage {constants.PREPARE_BASE_MODEL_STEP} completed <<<<<<\n\nx==========x")
    except Exception as e:
            logger.exception(e)
            raise e
    

    try: 
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {constants.TRAINING_STEP} started <<<<<<")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        logger.info(f">>>>>> stage {constants.TRAINING_STEP} completed <<<<<<\n\nx==========x")
    except Exception as e:
            logger.exception(e)
            raise e
