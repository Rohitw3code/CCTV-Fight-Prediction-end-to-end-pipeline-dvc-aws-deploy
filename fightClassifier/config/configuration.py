from fightClassifier.utils.read_yaml import read_yaml
from fightClassifier.entity.config_entity import DataIngestionConfig
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