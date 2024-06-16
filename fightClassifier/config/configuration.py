from fightClassifier.utils.read_yaml import read_yaml
from fightClassifier.entity.config_entity import (DataIngestionConfig,
                                                  DataPreprocessingConfig,
                                                  ModelTrainConfig,
                                                  MLFlowConfig)

from fightClassifier.entity.param_entity import (DataParam,
                                                 ViViTArchitectureParam,
                                                 TrainingParam,
                                                 OptimizerParam,
                                                 TubeletEmbeddingParam)


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

    def config_mlflow(self)->MLFlowConfig:
        try:
            config = self.config['mlflow_credential']
            config = MLFlowConfig(mlflow_tracking_uri=config['MLFLOW_TRACKING_URI'],
                                  mlflow_tracking_username=config['MLFLOW_TRACKING_USERNAME'],
                                  mlflow_tracking_password=config['MLFLOW_TRACKING_PASSWORD'],
                                  repo_name=config['REPO_NAME'],
                                  repo_owner=config['REPO_OWNER'],
                                  mlflow=config['MLFLOW'])
            
        except Exception as e:
            logger.error(e)
        return config
    
    def param_data(self)->DataParam:
        try:
            param = self.params['data']
            param = DataParam(
                dataset_name=param['DATASET_NAME'],
                batch_size=param['BATCH_SIZE'],
                input_shape=param['INPUT_SHAPE'],
                num_classes=param['NUM_CLASSES']
            )
        except Exception as e:
            logger.error(e)
    
    def param_optimizer(self)->OptimizerParam:
        try:
            param = self.params['optimizer']
            param = OptimizerParam(
                learning_rate=param['LEARNING_RATE'],
                weight_decay=param['WEIGHT_DECAY']
            )
        except Exception as e:
            logger.error(e)

    def param_training(self)->TrainingParam:
        try:
            param = self.params['training']
            param = TrainingParam(
                epochs=param['EPOCHS']
            )
        except Exception as e:
            logger.error(e)
    
    def param_tubelet_embedding(self)->TubeletEmbeddingParam:
        try:
            param = self.params['tubelet_embedding']
            param = TubeletEmbeddingParam(
                patch_size=param['PATCH_SIZE']
            )
        except Exception as e:
            logger.error(e)

    def param_vivit_architecture(self)->ViViTArchitectureParam:
        try:
            param = self.params['vivit_architecture']
            param = ViViTArchitectureParam(
                layer_norm_eps=param['LAYER_NORM_EPS'],
                projection_dim=param['PROJECTION_DIM'],
                num_heads=param['NUM_HEADS'],
                num_layers=param['NUM_LAYERS']
            )
        except Exception as e:
            logger.error(e)
