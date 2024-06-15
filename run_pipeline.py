from fightClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from fightClassifier.pipeline.data_preprocessing import DataPreprocessingPipeline
from fightClassifier.pipeline.data_loader_pipeline import DataLoaderPipeline

from fightClassifier import logger

if __name__ == '__main__':
    logger.info('Started---->  Data Ingestion Pipeline')
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info('Ended------>  Data Ingestion Pipeline')

    logger.info('Started---->  Data PreProcessing Pipeline')
    data_process = DataPreprocessingPipeline()
    final_dataset,final_labels,dims_df = data_process.main()
    logger.info('Ended------>  Data PreProcessing Pipeline')

    logger.info('Started---->  DataLoader Pipeline')
    data_loader = DataLoaderPipeline(final_dataset,final_labels)
    trainLoader,testLoader,validLoader = data_loader.main()
    logger.info('Ended------>  DataLoader Pipeline')




