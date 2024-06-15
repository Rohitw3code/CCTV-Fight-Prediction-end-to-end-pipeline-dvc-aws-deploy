from fightClassifier import logger
from fightClassifier.entity.config_entity import ModelTrainConfig
from fightClassifier.components.model_training import ModelTraining

class Evaluate:
    def __init__(self,testLoader):
        self.testLoader = testLoader

    def evaluate(self):
        model = ModelTraining().load_model()
        _, accuracy, top_5_accuracy = model.evaluate(self.testLoader)
        logger.info(f"Test accuracy: {round(accuracy * 100, 2)}%")
        logger.info(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
