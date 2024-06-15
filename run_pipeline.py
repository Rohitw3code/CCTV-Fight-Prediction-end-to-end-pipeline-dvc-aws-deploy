from fightClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from fightClassifier.pipeline.data_preprocessing import DataPreprocessingPipeline
from fightClassifier import logger

if __name__ == '__main__':
    logger.info('Started---->  Data Ingestion Pipeline')
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info('Ended------>  Data Ingestion Pipeline')

    logger.info('Started---->  Data PreProcessing Pipeline')
    data_process = DataPreprocessingPipeline()
    final_dataset,dims_df = data_process.main()
    logger.info('Ended------>  Data PreProcessing Pipeline')

    

