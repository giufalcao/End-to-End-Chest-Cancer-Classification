import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from src.entity.config_entity import BaseModelConfig


class BaseModel:
    """
    Class to prepare and update a base model.

    Attributes:
    - config (BaseModelConfig): Configuration for preparing the base model.
    """

    def __init__(self, config: BaseModelConfig):
        """
        Initialize BaseModel.

        Args:
        - config (BaseModelConfig): Configuration for preparing the base model.
        """
        self.config = config

    def get_base_model(self):
        """
        Retrieve the base model.

        Downloads the VGG16 model and saves it to the specified path.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepare the full model.

        Args:
        - model (tf.keras.Model): Base model.
        - classes (int): Number of classes for the classification task.
        - freeze_all (bool): Whether to freeze all layers of the base model.
        - freeze_till (int or None): Number of layers to freeze from the top.
        - learning_rate (float): Learning rate for model training.

        Returns:
        - tf.keras.Model: Compiled full model.
        """
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Update the base model.

        Prepares a new full model based on the existing base model and updates it.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the model to a specified path.

        Args:
        - path (Path): Path to save the model.
        - model (tf.keras.Model): Model to be saved.
        """
        model.save(path)
