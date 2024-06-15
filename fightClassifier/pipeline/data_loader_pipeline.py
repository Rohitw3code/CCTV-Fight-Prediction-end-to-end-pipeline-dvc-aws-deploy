from fightClassifier.components.data_loader import DataLoader
import numpy as np

class DataLoaderPipeline:
    def __init__(self,dataset:np.asarray,label:np.concatenate):
        self.dataset = dataset
        self.label = label

    def main(self):
        data_loader = DataLoader(dataset=self.dataset,
                                 label=self.label)
        data_loader.train_test_valid_split()
        return data_loader.get_loaders()


