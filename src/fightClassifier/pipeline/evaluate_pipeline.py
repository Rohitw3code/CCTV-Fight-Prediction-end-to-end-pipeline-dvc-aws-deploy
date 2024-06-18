from fightClassifier.components.evaluation import Evaluate
from fightClassifier.config.configuration import ConfigurationManager
from fightClassifier.utils import load_loader

class EvaluateModelPipeline:
    def __init__(self):
        self.testLoader = load_loader('artifacts/loader/test-loader')

    def main(self):
        evaluate = Evaluate(self.testLoader)
        evaluate.evaluate()
        evaluate.mlflow_track()