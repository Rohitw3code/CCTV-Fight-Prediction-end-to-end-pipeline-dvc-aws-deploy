from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    kaggle_dataset_dir: Path
    kaggle_dataset_name: str

@dataclass(frozen=True)
class DataPreprocessingConfig:
    load_dataset_dir:Path

@dataclass(frozen=True)
class ModelTrainConfig:
    save_model_dir:Path
    save_model_name: Path

@dataclass(frozen=True)
class IntermediateDataConfig:
    preprocessed_video_label_path: Path
    train_loader_path: Path
    test_loader_path: Path
    val_loader_path: Path


@dataclass(frozen=True)
class MLFlowConfig:
    mlflow_tracking_uri:str
    mlflow_tracking_username:str
    mlflow_tracking_password:str
    repo_owner:str
    repo_name:str
    mlflow:bool
    project_name:str
    model_name:str
    evaluation_param_path: Path
    model_param_path: Path