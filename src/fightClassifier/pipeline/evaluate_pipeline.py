from fightClassifier.components.evaluation import Evaluate
from fightClassifier.config.configuration import ConfigurationManager

class EvaluateModelPipeline:
    def __init__(self,testLoader):
        self.testLoader = testLoader

    def main(self):
        configManager = ConfigurationManager()
        config = configManager.config_mlflow()        
        evaluate = Evaluate(self.testLoader,config=config)
        evaluate.evaluate()
        evaluate.mlflow_track()