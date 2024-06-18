from fightClassifier.components.data_loader import DataLoader
from fightClassifier.params.param_manager import ParamManager
from fightClassifier.config.configuration import ConfigurationManager
from fightClassifier.utils import load_dataset

import numpy as np

class DataLoaderPipeline:
    def __init__(self,dataset:np.asarray,label:np.concatenate):
        self.dataset = dataset
        self.label = label

    def main(self):
        
        config = ConfigurationManager()
        config = config.config_data_preprocessing()

        dataset,label = load_dataset('artifacts/preprocessed_data/video.npz')

        paramManager = ParamManager()
        param = paramManager.param_data()
        data_loader = DataLoader(dataset=self.dataset,
                                 label=self.label,
                                 param=param)
        

        data_loader.train_test_valid_split()
        return data_loader.get_loaders()