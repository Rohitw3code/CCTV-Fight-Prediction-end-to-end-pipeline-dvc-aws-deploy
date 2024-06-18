from fightClassifier.components.data_loader import DataLoader
from fightClassifier.params.param_manager import ParamManager
from fightClassifier.config.configuration import ConfigurationManager
from fightClassifier.utils import load_dataset , save_to_tensor

import numpy as np

class DataLoaderPipeline:
    def __init__(self):
        pass

    def main(self):
        
        config = ConfigurationManager()
        config = config.config_data_preprocessing()

        dataset,label = load_dataset('artifacts/preprocessed_data/video.npz')

        paramManager = ParamManager()
        param = paramManager.param_data()
        data_loader = DataLoader(dataset=dataset,
                                 label=label,
                                 param=param)
        
        data_loader.train_test_valid_split()

        trainLoader,testLoader,valLoader = data_loader.get_loaders()


        return trainLoader,testLoader,valLoader