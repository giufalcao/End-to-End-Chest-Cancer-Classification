from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for data ingestion.

    Attributes:
    - root_dir (Path): Root directory for data ingestion.
    - source_URL (str): URL of the data source.
    - local_data_file (Path): Path to the local data file.
    - unzip_dir (Path): Directory for unzipping the data.
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class BaseModelConfig:
    """
    Configuration for preparing a base model.

    Attributes:
    - root_dir (Path): Root directory for model preparation.
    - base_model_path (Path): Path to the base model.
    - updated_base_model_path (Path): Path to save the updated base model.
    - params_image_size (List[int]): Size of the input images.
    - params_learning_rate (float): Learning rate for training.
    - params_include_top (bool): Whether to include the top layer in the model.
    - params_weights (str): Type of weights to initialize the model.
    - params_classes (int): Number of classes in the dataset.
    """
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: List[int]
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for model training.

    Attributes:
    - root_dir (Path): Root directory for training.
    - trained_model_path (Path): Path to save the trained model.
    - updated_base_model_path (Path): Path to the updated base model.
    - training_data (Path): Path to the training data.
    - params_epochs (int): Number of epochs for training.
    - params_batch_size (int): Batch size for training.
    - params_is_augmentation (bool): Whether data augmentation is enabled.
    - params_image_size (List[int]): Size of the input images.
    """
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: List[int]

@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for model evaluation.

    Attributes:
    - path_of_model (Path): Path to the model for evaluation.
    - training_data (Path): Path to the evaluation data.
    - all_params (dict): All parameters used for evaluation.
    - mlflow_uri (str): URI for MLflow tracking.
    - params_image_size (list): Size of the input images.
    - params_batch_size (int): Batch size for evaluation.
    """
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
