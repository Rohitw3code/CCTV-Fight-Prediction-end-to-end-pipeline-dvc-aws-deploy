from fightClassifier.components.model_training import ModelTraining
from fightClassifier.params.param_manager import ParamManager
from fightClassifier.utils import load_loader
from fightClassifier.pipeline.data_loader_pipeline import DataLoaderPipeline


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        
        trainLoader = load_loader('artifacts/loader/train-loader')
        testLoader = load_loader('artifacts/loader/test-loader')
        validLoader = load_loader('artifacts/loader/val-loader')

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
