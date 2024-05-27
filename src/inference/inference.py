import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.entity.config_entity import EvaluationConfig
from src.utils.common import save_json


class Evaluation:
    """
    Class to evaluate a model.

    Attributes:
    - config (EvaluationConfig): Configuration for evaluation.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize Evaluation.

        Args:
        - config (EvaluationConfig): Configuration for evaluation.
        """
        self.config = config

    def _valid_generator(self):
        """
        Create a validation data generator.
        """
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Load a Keras model from a given path.

        Args:
        - path (Path): Path to the model file.

        Returns:
        - tf.keras.Model: Loaded Keras model.
        """
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """
        Evaluate the loaded model using the validation generator.
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
        self.log_into_mlflow()

    def save_score(self):
        """
        Save evaluation scores to a JSON file.
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """
        Log evaluation metrics and model to MLflow.
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
