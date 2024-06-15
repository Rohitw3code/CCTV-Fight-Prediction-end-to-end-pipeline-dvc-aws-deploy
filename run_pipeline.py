from fightClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from fightClassifier import logger

if __name__ == '__main__':
    logger.info('Started---->  Data Ingestion Pipeline')
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info('Ended------>  Data Ingestion Pipeline')

