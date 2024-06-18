import pickle
import tensorflow as tf
import numpy as np
import os
import json
from pathlib import Path
import yaml
from fightClassifier import logger


from pathlib import Path


def read_yaml(path_to_yaml: Path) -> dict:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file
    """
    create_folder_for_path(path=path_to_yaml)
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        raise e
    
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    create_folder_for_path(path=path)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")

def load_json(path: Path) -> dict:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    create_folder_for_path(path=path)
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return content

def create_folder_for_path(path):
    folder_path = os.path.dirname(path)
    if os.path.isdir(folder_path) == False:
        os.makedirs(folder_path,exist_ok=True)
        logger.info(f'Dirs created folder : -> {path}')


def save_dataset(video_data,labels,path):
    create_folder_for_path(path)
    np.savez(path,video_data=video_data,labels=labels)


def load_dataset(path):
    create_folder_for_path(path)
    data = np.load(path)
    final_dataset = data['video_data']
    labels = data['labels']   
    return final_dataset,labels 

def save_loader(loader,path):
    create_folder_for_path(path)
    # Save trainloader to a file
    tf.data.experimental.save(loader,path=path)

def load_loader(path):
    create_folder_for_path(path)
    # To load the trainloader from the saved file
    loader = tf.data.experimental.load(path=path)
    return loader

