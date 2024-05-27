from src.config.configuration import ConfigurationManager
from src.logging import logger
from src.constants import constants
from src.models.base_model import BaseModel

class BaseModelConfigurationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = BaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()