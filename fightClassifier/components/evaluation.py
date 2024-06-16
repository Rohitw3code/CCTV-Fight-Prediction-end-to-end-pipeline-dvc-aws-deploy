from fightClassifier import logger
from fightClassifier.entity.config_entity import ModelTrainConfig,MLFlowConfig
from fightClassifier.components.model_training import ModelTraining
import mlflow.keras
from urllib.parse import urlparse
import dagshub
import mlflow
import os


class Evaluate:
    def __init__(self,testLoader,config:MLFlowConfig):
        self.testLoader = testLoader
        self.accuracy = 0
        self.top_5_accuracy = 0
        self.config = config
        self.model = None        

    def evaluate(self):
        self.model = ModelTraining().load_model()
        _, self.accuracy, self.top_5_accuracy = self.model.evaluate(self.testLoader)
        logger.info(f"Test accuracy: {round(self.accuracy * 100, 2)}%")
        logger.info(f"Test top 5 accuracy: {round(self.top_5_accuracy * 100, 2)}%")
    
    def mlflow_track(self):
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            # mlflow.log_param('accuracy', accuracy)
            mlflow.log_metrics({
                'accuracy':self.accuracy,
                'top_5_accuracy':self.top_5_accuracy
            })
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="video-vision")
            else:
                mlflow.keras.log_model(self.model, "model")

