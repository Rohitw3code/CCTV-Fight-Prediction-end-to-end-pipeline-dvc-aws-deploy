from fightClassifier.components.data_preprocessing import DataPreprocessing
from fightClassifier.components.data_loader import DataLoader
from fightClassifier import logger
import numpy as np
import tensorflow as tf
import imageio
import io
import cv2


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
        self.video_file = '.'

    def _predict_video_array(self,frame_limit):
        video_ = tf.squeeze(single_preprocess(frame_limit), axis=-1).numpy()
        output = self.model.predict(tf.expand_dims(video_,axis=0))[0]
        pred = np.argmax(output, axis=0)
        return pred


    def load_long_video(self,video_file):
        self.video_file = video_file
        dp = DataPreprocessing('.')
        frames = dp._load_all_frames(video_file)['frames']

        total_frames = frames.shape[0]
        video_batch = int(total_frames // 42)  # n will be 50

        # Trim the array to be a multiple of 42
        trimmed_video = frames[:video_batch * 42]

        # Reshape the array to the new shape (n, 42, 128, 128, 3)
        video_array = trimmed_video.reshape(video_batch, 42, 128, 128, 3)
        violence = []
        print('video array : ',video_array.shape)
        for batch_num,short in enumerate(video_array):
            tensor = single_preprocess(short)
            video_ = tf.squeeze(tensor, axis=-1).numpy()
            output = self.model.predict(tf.expand_dims(video_,axis=0))[0]
            if np.argmax(output) == 1:
                violence.append((batch_num,short))
                violence.append((batch_num,(short*255)))
            print('batch no : ',batch_num)
        
        return violence

    def video_preprocessing(self,video_file):
        dp = DataPreprocessing('.')
        frames = dp._load_all_frames(video_file)
        trim_video = dp._trim_video_frames(frames['frames'],42)
        return trim_video
    
    def video_info(self):
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(self.video_file)
            duration       = clip.duration
            fps            = clip.fps
            width, height  = clip.size
            return duration, fps, (width, height)
        except Exception as e:
            logger.info('can read the video data')

    def predict(self,video_file):
        self.video_file = video_file
        tensor = single_preprocess(self.video_preprocessing(video_file))
        video_ = tf.squeeze(tensor, axis=-1).numpy()
        output = self.model.predict(tf.expand_dims(video_,axis=0))[0]
        pred = np.argmax(output, axis=0)
        prediction = {
            'output':self.label_dict[pred]
        }
        print(f"prediction [ {video_file} ] : ",prediction)
        return prediction