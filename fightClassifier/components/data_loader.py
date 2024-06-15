from fightClassifier import logger
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf  # for data preprocessing only
import keras
from keras import layers, ops
from typing import Tuple
import numpy as np



# Setting seed for reproducibility
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)

# Setting seed for reproducibility
SEED = 77
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
tf.random.set_seed(SEED)

# DATA
DATASET_NAME = "fight/nofights"
BATCH_SIZE = 4
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (42, 128, 128, 3)
NUM_CLASSES = 2

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 20

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2

class DataLoader:
    def __init__(self,dataset:np.asarray,label:np.concatenate):
        self.dataset = dataset
        self.label = label

        self.train_X = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.valid_x = None
        self.valid_y = None

    def train_test_valid_split(self):
        self.train_X,self.test_x,self.train_y,self.test_y = train_test_split(self.dataset,self.label,test_size=0.2,
                                                                             shuffle=True,random_state=42)

        self.test_x,self.valid_x,self.test_y,self.valid_y = train_test_split(self.test_x,self.test_y,test_size=0.5,
                                                                             shuffle=True,random_state=42)
        logger.info('Train Test Valid Split done :)')
        

    @tf.function
    def preprocess(self,frames: tf.Tensor, label: tf.Tensor):
        """Preprocess the frames tensors and parse the labels."""
        # Preprocess images
        frames = tf.image.convert_image_dtype(
            frames[
                ..., tf.newaxis
            ],  # The new axis is to help for further processing with Conv3D layers
            tf.float32,
        )
        # Parse label
        label = tf.cast(label, tf.float32)
        return frames, label


    def prepare_dataloader(
        self,
        videos: np.ndarray,
        labels: np.ndarray,
        loader_type: str = "train",
        batch_size: int = BATCH_SIZE,
    ):
        """Utility function to prepare the dataloader."""
        dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

        if loader_type == "train":
            dataset = dataset.shuffle(BATCH_SIZE * 2)

        dataloader = (
            dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return dataloader
    
    def get_loaders(self)->Tuple[prepare_dataloader,
                                 prepare_dataloader,
                                 prepare_dataloader]:
        
        print(self.train_X.shape)
        trainloader = self.prepare_dataloader(self.train_X, self.train_y, "train")
        testloader = self.prepare_dataloader(self.test_x, self.test_y, "test")
        validloader = self.prepare_dataloader(self.valid_x, self.valid_y, "test")
        return trainloader,testloader,validloader






