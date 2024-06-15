import cv2
import numpy as np
import os
from pathlib import Path

class DataPreprocessing:
    def __init__(self,path:Path):
        self.path = path

    def _load_all_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
    
        if not cap.isOpened():
            return {'frames':None,'frames_dim':None,'success':False}
        
        frames_dims = []
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h,w,c = frame.shape
            frames_dims.append(list([0,h,w,c]))
            frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_CUBIC)
            frames.append(frame)
        
        cap.release()
        return {'frames':np.asarray(frames),'frames_dim':frames_dims,'success':True}

    def _trim_video_frames(self,video,max_frame):
        '''
        Args:
            video: video (collection of frames)
            max_frame: max number of frames
        '''
        f,_,_,_ = video.shape
        startf = f//2 - max_frame//2
        return video[startf:startf+max_frame, :, :, :]

    def load_video(self):
        fights=[]
        video_dims = []
        for filename in os.listdir(self.path):
            video_path = os.path.join(self.path, filename)
            load_data = self._load_all_frames(video_path)
            if load_data['success']==False:
                continue
            video = load_data['frames']
            dims = np.asarray(load_data['frames_dim'])
            dims[:,0] = video.shape[0]
            video_dims += dims.tolist()
            fights.append(self._trim_video_frames(video,42)) #42 means each video trimed to 42 frames only
        return fights,video_dims


    def preprocessing(self):
        pass