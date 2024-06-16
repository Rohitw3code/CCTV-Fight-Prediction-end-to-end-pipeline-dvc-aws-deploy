from fightClassifier.components.evaluation import Evaluate
from fightClassifier.config.configuration import ConfigurationManager

class EvaluateModelPipeline:
    def __init__(self,testLoader):
        self.testLoader = testLoader

    def main(self):
        evaluate = Evaluate(self.testLoader)
        evaluate.evaluate()
        evaluate.mlflow_track()