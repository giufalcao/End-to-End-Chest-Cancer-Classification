from src.config.configuration import ConfigurationManager
from training.training import Training
from src.logging import logger


class ModelTrainingPipeline:
    """
    Pipeline for training the model.
    """

    def __init__(self):
        pass

    def main(self):
        """
        Main method to execute the training pipeline.
        """
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

