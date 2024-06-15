from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    kaggle_dataset_dir: Path
    kaggle_dataset_name: str

@dataclass(frozen=True)
class DataPreprocessingConfig:
    load_dataset_dir:Path
