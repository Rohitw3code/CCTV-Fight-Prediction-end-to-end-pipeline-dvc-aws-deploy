from fightClassifier.components.evaluation import Evaluate
from fightClassifier.config.configuration import ConfigurationManager
from fightClassifier.utils import load_loader

class EvaluateModelPipeline:
    def __init__(self):
        self.config = ConfigurationManager().config_intermediate_data()
        self.testLoader = load_loader(self.config.test_loader_path)

    def main(self):
        evaluate = Evaluate(self.testLoader)
        evaluate.evaluate()
        evaluate.mlflow_track()


if __name__ == '__main__':
    pipeline = EvaluateModelPipeline()
    pipeline.main()