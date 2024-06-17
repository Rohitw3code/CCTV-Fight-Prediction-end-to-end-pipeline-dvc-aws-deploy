from fightClassifier import logger
from fightClassifier.components.model_training import ModelTraining
from fightClassifier.config.configuration import ConfigurationManager
import mlflow
import os
from fightClassifier.utils.read_yaml import save_json



class Evaluate:
    def __init__(self,testLoader):
        self.testLoader = testLoader
        self.accuracy = 0
        self.top_5_accuracy = 0
        self.model = None        

    def evaluate(self):
        self.model = ModelTraining().load_model()
        _, self.accuracy, self.top_5_accuracy = self.model.evaluate(self.testLoader)
        logger.info(f"Test accuracy: {round(self.accuracy * 100, 2)}%")
        logger.info(f"Test top 5 accuracy: {round(self.top_5_accuracy * 100, 2)}%")
    
    def mlflow_track(self):
        mlflow_config = ConfigurationManager().config_mlflow()
        folder_path = os.path.dirname(mlflow_config.evaluation_param_path)
        os.makedirs(folder_path,exist_ok=True)

        save_json(mlflow_config.evaluation_param_path,{
                'accuracy':self.accuracy,
                'top_5_accuracy':self.top_5_accuracy
            })
    
