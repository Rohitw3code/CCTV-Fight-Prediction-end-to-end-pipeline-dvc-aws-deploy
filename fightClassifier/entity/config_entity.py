from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    kaggle_dataset_dir: Path
    kaggle_source_URL: str
