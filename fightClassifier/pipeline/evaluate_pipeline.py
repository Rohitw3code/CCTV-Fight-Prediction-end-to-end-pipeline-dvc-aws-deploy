from fightClassifier.components.evaluation import Evaluate

class EvaluateModelPipeline:
    def __init__(self,testLoader):
        self.testLoader = testLoader

    def main(self):
        evaluate = Evaluate(self.testLoader)
        evaluate.evaluate()