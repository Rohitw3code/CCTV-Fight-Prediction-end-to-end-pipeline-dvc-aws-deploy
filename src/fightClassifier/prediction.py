from fightClassifier.components.data_preprocessing import DataPreprocessing
from fightClassifier.components.data_loader import DataLoader
import numpy as np
import tensorflow as tf


@tf.function
def single_preprocess(frames: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],
        tf.float32,
    )
    return frames


class ModelPredictor:
    def __init__(self,video_file=True):
        self.label_dict = {0: 'noFight', 1: 'Fight'}
        self.model = tf.keras.models.load_model('model/video.keras')



    def video_preprocessing(self,video_file):
        dp = DataPreprocessing('.')
        frames = dp._load_all_frames(video_file)
        trim_video = dp._trim_video_frames(frames['frames'],42)
        return trim_video

    def predict(self,video_file):
        tensor = single_preprocess(self.video_preprocessing(video_file))
        video_ = tf.squeeze(tensor, axis=-1).numpy()
        output = self.model.predict(tf.expand_dims(video_,axis=0))[0]
        pred = np.argmax(output, axis=0)
        prediction = {
            'output':self.label_dict[pred]
        }
        print(f"prediction [ {video_file} ] : ",prediction)
        return prediction