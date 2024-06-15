from fightClassifier import logger
from fightClassifier.components.data_ingestion import DataIngestion
from fightClassifier.config.configuration import ConfigurationManager

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        configurationManager = ConfigurationManager()
        config = configurationManager.config_data_ingestion()
        data_ingestion = DataIngestion(config=config)
        data_ingestion.auth()
        data_ingestion.download_dataset_from_kaggle()

if __name__ == '__main__':
    pipe = DataIngestionPipeline()
    pipe.main()