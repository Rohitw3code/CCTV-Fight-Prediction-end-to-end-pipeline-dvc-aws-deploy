from fightClassifier.components.model_training import ModelTraining
from fightClassifier.params.param_manager import ParamManager
from fightClassifier.utils import load_loader
from fightClassifier.config.configuration import ConfigurationManager


class ModelTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager().config_intermediate_data()

    def main(self):
        
        trainLoader = load_loader(self.config.train_loader_path)
        testLoader = load_loader(self.config.test_loader_path)
        validLoader = load_loader(self.config.val_loader_path)

        paramManager = ParamManager()
        params = paramManager.param_mega()

        modelTrain = ModelTraining(trainLoader=trainLoader,
                                   testLoader=testLoader,
                                   validLoader=validLoader,
                                   params=params)

        model = modelTrain.train()
        modelTrain.save_model()
        modelTrain.mlflow_tracker()
        return model

if __name__ == '__main__':
    pipeline = ModelTrainingPipeline()
    pipeline.main()