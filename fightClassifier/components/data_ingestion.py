from fightClassifier import logger
from pathlib import Path
import os
import kaggle
from fightClassifier.config.configuration import ConfigurationManager


class DataIngestion:
    def __init__(self):
        pass
    
    def auth(self):

        configurationManager = ConfigurationManager()
        config = configurationManager.config_data_ingestion()
        print('config == > ',config)

        try:
            kaggle.api.authenticate()
            logger.info('Authenticated :}')
        except Exception as e:
            logger.error('Authentication Problem : '+e)

    def download_dataset_from_kaggle(self):
        try:
            logger.info('downloading dataset from kaggle')
            os.makedirs('artifacts/kaggle_dataset',exist_ok=True)
            # kaggle.api.dataset_download_files('naveenk903/movies-fight-detection-dataset', path='./artifacts/kaggle_dataset', unzip=True)
            logger.info('download completed')
        except Exception as e:
            logger.error('Error while downloading : '+e)



if __name__ == '__main__':
    data_ingestion = DataIngestion()
    data_ingestion.download_dataset_from_kaggle()