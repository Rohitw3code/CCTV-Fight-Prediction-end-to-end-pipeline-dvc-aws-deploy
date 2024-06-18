from fightClassifier.utils.read_yaml import read_yaml
from fightClassifier.entity.config_entity import *

from fightClassifier.entity.param_entity import *

from fightClassifier import logger

class ConfigurationManager:
    def __init__(self,CONFIG_PATH='config/config.yaml',PARAMS_PATH='params.yaml'):
        self.config = read_yaml(CONFIG_PATH)
        self.params = read_yaml(PARAMS_PATH)
    
    def config_data_ingestion(self)->DataIngestionConfig:
        config = self.config['data_ingestion']
        try:
            config = DataIngestionConfig(
                kaggle_dataset_dir=config['kaggle_dataset_dir'],
                kaggle_dataset_name=config['kaggle_dataset_name']
            )
        except Exception as e:
            logger.error(e)            

        return config
    
    def config_data_preprocessing(self)->DataPreprocessingConfig:
        config = self.config['data_preprocessing']
        try:
            config = DataPreprocessingConfig(
                load_dataset_dir=config['load_dataset_dir']
            )
        except Exception as e:
            logger.error(e)
        return config
    

    def config_model_train(self)->ModelTrainConfig:
        try:
            config = self.config['model_train']
            config = ModelTrainConfig(save_model_dir=config['save_model_dir'],
                                      save_model_name=config['save_model_name'])
        except Exception as e:
            logger.error(e)
        return config
    
    def config_intermediate_data(self)->IntermediateDataConfig:
        try:
            config = self.config['intermediate_data']
            config = IntermediateDataConfig(preprocessed_video_label_path=config['preprocessed_video_label_path'],
                                            train_loader_path=config['train_loader_path'],
                                            test_loader_path=config['test_loader_path'],
                                            val_loader_path=config['val_loader_path'])
        except Exception as e:
            logger.error(e)
        return config

    


    def config_mlflow(self)->MLFlowConfig:
        try:
            config = self.config['mlflow_credential']
            config = MLFlowConfig(mlflow_tracking_uri=config['MLFLOW_TRACKING_URI'],
                                  mlflow_tracking_username=config['MLFLOW_TRACKING_USERNAME'],
                                  mlflow_tracking_password=config['MLFLOW_TRACKING_PASSWORD'],
                                  repo_name=config['REPO_NAME'],
                                  repo_owner=config['REPO_OWNER'],
                                  mlflow=config['MLFLOW'],
                                  project_name=config['project_name'],
                                  model_name=config['model_name'],
                                  evaluation_param_path=config['evaluation_param_path'],
                                  model_param_path=config['model_param_path'])
            
        except Exception as e:
            logger.error(e)
        return config
    