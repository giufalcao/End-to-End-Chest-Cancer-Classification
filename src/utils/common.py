import os
import yaml
import json
import joblib
import base64
from box import ConfigBox
from pathlib import Path
from typing import Any

from src.logging import logger 


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads YAML file and returns its contents as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: For any other unexpected error.

    Returns:
        ConfigBox: Contents of the YAML file as a ConfigBox.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not content:
                raise ValueError("YAML file is empty")
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)
    except Exception as e:
        raise e


def create_directories(path_to_directories: list, verbose=True):
    """Create directories specified in the list.

    Args:
        path_to_directories (list): List of paths of directories to be created.
        verbose (bool, optional): Whether to log directory creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


def save_json(path: Path, data: dict):
    """Save data as JSON to the specified path.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to be saved as JSON.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


def load_json(path: Path) -> ConfigBox:
    """Load data from a JSON file and return as a ConfigBox.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data loaded from the JSON file.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


def save_bin(data: Any, path: Path):
    """Save binary data to the specified path using joblib.

    Args:
        data (Any): Data to be saved as binary.
        path (Path): Path to save the binary file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


def load_bin(path: Path) -> Any:
    """Load binary data from the specified path using joblib.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Loaded object from the binary file.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


def get_size(path: Path) -> str:
    """Get the size of a file in kilobytes.

    Args:
        path (Path): Path to the file.

    Returns:
        str: Size of the file in kilobytes.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decode_image(imgstring, file_name):
    """Decode a base64 encoded image string and save it to a file.

    Args:
        imgstring (str): Base64 encoded image string.
        file_name (str): Name of the file to save the image.
    """
    imgdata = base64.b64decode(imgstring)
    with open(file_name, 'wb') as f:
        f.write(imgdata)


def encode_image_into_base64(cropped_image_path):
    """Encode an image file into base64.

    Args:
        cropped_image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    with open(cropped_image_path, "rb") as f:
        return base64.b64encode(f.read())
