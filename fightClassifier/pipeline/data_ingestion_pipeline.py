from fightClassifier.components.data_ingestion import DataIngestion


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        data_ingestion = DataIngestion()
        data_ingestion.auth()
        data_ingestion.download_dataset_from_kaggle()
        print('pipeline--->end')

if __name__ == '__main__':
    pipe = DataIngestionPipeline()
    pipe.main()