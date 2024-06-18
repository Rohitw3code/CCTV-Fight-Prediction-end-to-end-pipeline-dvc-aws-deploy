from fightClassifier import logger
from pathlib import Path
import os
import kaggle
from fightClassifier.config.configuration import ConfigurationManager
from fightClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config
    
    def auth(self):

        try:
            kaggle.api.authenticate()
            logger.info('Authenticated :}')
        except Exception as e:
            logger.error('Authentication Problem : '+e)

    def download_dataset_from_kaggle(self):
        os.makedirs(self.config.kaggle_dataset_dir,exist_ok=True)
        if any(Path(self.config.kaggle_dataset_dir).iterdir()):
            return logger.info('Dataset already downloaded')
        try:
            logger.info('downloading dataset from kaggle')
            print(self.config)
            kaggle.api.dataset_download_files(self.config.kaggle_dataset_name, path=self.config.kaggle_dataset_dir, unzip=True)
            logger.info('download completed!!')
        except Exception as e:
            logger.error(e)



if __name__ == '__main__':
    data_ingestion = DataIngestion()
    data_ingestion.download_dataset_from_kaggle()