from fightClassifier.components.model_training import ModelTraining
from fightClassifier.params.param_manager import ParamManager
from fightClassifier.config.configuration import ConfigurationManager
from fightClassifier.utils import load_from_pickle


class ModelTrainingPipeline:
    def __init__(self,trainLoader,testLoader,validLoader):
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.validLoader = validLoader

    def main(self):
        paramManager = ParamManager()
        params = paramManager.param_mega()

        configManager = ConfigurationManager()
        config = configManager.config_intermediate_data()
        

        modelTrain = ModelTraining(trainLoader=self.trainLoader,
                                   testLoader=self.testLoader,
                                   validLoader=self.validLoader,
                                   params=params
                                   )
        
        model = modelTrain.train()
        modelTrain.save_model()
        modelTrain.mlflow_tracker()
        return model

