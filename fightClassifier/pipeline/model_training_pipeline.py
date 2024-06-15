from fightClassifier.components.model_training import ModelTraining
from fightClassifier.config.configuration import ConfigurationManager


class ModelTrainingPipeline:
    def __init__(self,trainLoader,testLoader,validLoader):
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.validLoader = validLoader
        self.config = ConfigurationManager()

    def main(self):
        config = self.config.config_model_train()
        modelTrain = ModelTraining(trainLoader=self.trainLoader,
                                   testLoader=self.testLoader,
                                   validLoader=self.validLoader,
                                   config=config
                                   )
        
        model = modelTrain.train()
        modelTrain.save_model()
        return model

