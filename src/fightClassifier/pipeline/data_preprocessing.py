from fightClassifier.config.configuration import ConfigurationManager
from fightClassifier.components.data_preprocessing import DataPreprocessing
from fightClassifier import logger
from fightClassifier.utils import save_dataset
import pandas as pd
import numpy as np
from typing import Tuple

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self)->Tuple[np.asarray,
                          np.concatenate,
                          np.asarray]:
        
        configManager = ConfigurationManager()
        config = configManager.config_data_preprocessing()
        data_processing = DataPreprocessing(config.load_dataset_dir)
        video_dataset,video_labels,video_dims = data_processing.load_video()
        logger.info(f'Feature : {video_dataset.shape}')
        logger.info(f'Label   : {np.asarray(video_labels).shape}')
        dims_df = pd.DataFrame(video_dims,columns=['frames','width','height','channel'])
        # logger.info('video dims info -> ',dims_df.describe())
        # logger.info('dims dataframe -> ')
        # logger.info(dims_df.head())

        config = configManager.config_intermediate_data()
        save_dataset(video_data=video_dataset,labels=video_labels
                     ,path=config.preprocessed_video_label_path)


        return video_dataset,video_labels,dims_df


if __name__ == '__main__':
    dpp = DataPreprocessingPipeline()
    dpp.main()
