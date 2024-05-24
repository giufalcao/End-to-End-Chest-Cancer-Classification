from dataclasses import dataclass
from pathlib import Path

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
