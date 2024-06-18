from fightClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from fightClassifier.pipeline.data_preprocessing import DataPreprocessingPipeline
from fightClassifier.pipeline.data_loader_pipeline import DataLoaderPipeline
from fightClassifier.pipeline.model_training_pipeline import ModelTrainingPipeline
from fightClassifier.pipeline.evaluate_pipeline import EvaluateModelPipeline
from fightClassifier.pipeline.daghub_mlflow_pipeline import MLFlowPipeline

from fightClassifier import logger

if __name__ == '__main__':

    logger.info('Started---->  Data Ingestion Pipeline')
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info('Ended------>  Data Ingestion Pipeline')

    logger.info('Started---->  Data PreProcessing Pipeline')
    data_process = DataPreprocessingPipeline()
    data_process.main()
    logger.info('Ended------>  Data PreProcessing Pipeline')

    logger.info('Started---->  DataLoader Pipeline')
    data_loader = DataLoaderPipeline()
    trainLoader,testLoader,validLoader = data_loader.main()
    logger.info('Ended------>  DataLoader Pipeline')

    logger.info('Started---->  ModelTraining Pipeline')
    model_train = ModelTrainingPipeline(trainLoader=trainLoader,
                                        testLoader=testLoader,
                                        validLoader=validLoader)
    
    model = model_train.main()
    logger.info('Ended------>  ModelTraining Pipeline')


    logger.info('Started---->  Evaluation Pipeline')
    eval_ = EvaluateModelPipeline(testLoader=testLoader)    
    eval_.main()
    logger.info('Ended------>  Evaluation Pipeline')

    logger.info('setup MLFlow tracker started--->')
    mlflow_pipeline = MLFlowPipeline()
    mlflow_pipeline.main()
    logger.info('Ended---->  MLFlow setup done :)')