import pickle
import tensorflow as tf
import numpy as np


def save_dataset(video_data,labels,path):
    np.savez(path,video_data=video_data,labels=labels)


def load_dataset(path):
    data = np.load(path)
    final_dataset = data['video_data']
    labels = data['labels']   
    return final_dataset,labels 

def save_loader(loader,path):
    # Save trainloader to a file
    tf.data.experimental.save(loader,path=path)

def load_loader(path):
    # To load the trainloader from the saved file
    loader = tf.data.experimental.load(path=path)
    return loader