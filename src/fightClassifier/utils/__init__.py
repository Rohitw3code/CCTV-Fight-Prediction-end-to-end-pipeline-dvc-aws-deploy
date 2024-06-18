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

def save_to_pickle(obj, filename):
    tensor_serialized = tf.io.serialize_tensor(obj)
    tf.io.write_file(filename, tensor_serialized)

def load_from_pickle(filename):
    tensor_serialized = tf.io.read_file(filename)
    tensor = tf.io.parse_tensor(tensor_serialized, out_type=tf.uint8)  # Adjust the data type according to your tensor type
    return tensor