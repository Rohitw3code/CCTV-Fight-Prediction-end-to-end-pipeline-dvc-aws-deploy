from fightClassifier.components.data_loader import DataLoader
from fightClassifier.params.param_manager import ParamManager
from fightClassifier.config.configuration import ConfigurationManager
from fightClassifier.utils import load_dataset , save_loader

import numpy as np

class DataLoaderPipeline:
    def __init__(self):
        pass

    def main(self):
        
        configManager = ConfigurationManager()
        configInterm = configManager.config_intermediate_data()
        dataset,label = load_dataset(configInterm.preprocessed_video_label_path)

        paramManager = ParamManager()
        param = paramManager.param_data()
        data_loader = DataLoader(dataset=dataset,
                                 label=label,
                                 param=param)
        
        data_loader.train_test_valid_split()
        trainLoader,testLoader,valLoader = data_loader.get_loaders()


        save_loader(loader=trainLoader,
                    path=configInterm.train_loader_path)
        save_loader(loader=testLoader,
                    path=configInterm.test_loader_path)
        save_loader(loader=valLoader,
                    path=configInterm.val_loader_path)
        return trainLoader,testLoader,valLoader
    
if __name__ == '__main__':
    pipeline = DataLoaderPipeline()
    pipeline.main()