import numpy as np
import pandas as pd
import random
import os
import sys
import numpy as np
# import matplotlib.pyplot as plt 
# import plotly
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder 
# from tensorflow.keras.utils import to_categorical, plot_model
# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from skimage.transform import resize
import h5py
from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from keras.preprocessing import image
# from keras.regularizers import l2
import cv2


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        
        # Load the model configuration in read-only mode
        with h5py.File("model_checkpoints/mobilenet-sgd-loss_0.6980-accuracy_0.8534.h5", "r") as f:
            model_config = f.attrs.get("model_config")
            # If the model_config is bytes, you can decode it; otherwise, it's already a string.
            if isinstance(model_config, bytes):
                model_config = model_config.decode("utf-8")
            if '"groups": 1,' in model_config:
                model_config = model_config.replace('"groups": 1,', '')
                # If needed, encode it back if your environment expects bytes
                # Note: You cannot modify the file in read-only mode, so this line is removed.
        
        self.emotion_model = load_model("model_checkpoints/mobilenet-sgd-loss_0.6980-accuracy_0.8534.h5", compile=False)

    def __del__(self):
        self.video.release()

    def LoadData(self):
        unedited_train_path = "data/FER-2013/train/"
        unedited_test_path = "data/Actual_FER-2013/test/"
        emotions = os.listdir(unedited_train_path)
        len_unedited_train = {}
        len_unedited_test = {}

        for emotion in emotions:
            file_train = os.listdir(os.path.join(unedited_train_path, emotion))
            len_unedited_train[emotion] = len(file_train)
            
            file_test = os.listdir(os.path.join(unedited_test_path, emotion))
            len_unedited_test[emotion] = len(file_test)

        print("Unedited Train: ", len_unedited_train)
        print("Unedited Test: ", len_unedited_test)

        total_unedited_train = sum(len_unedited_train.values())
        total_unedited_test = sum(len_unedited_test.values())

        print()
        print("Total Unedited Train: ", total_unedited_train)
        print("Total Unedited Test: ", total_unedited_test)

        return total_unedited_train, total_unedited_test

    def GetFrame(self):
        ret, frame = self.video.read()
        # print(ret) #if camera is read properly then this is true

        # total_unedited_train, total_unedited_test = self.LoadData()

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        # num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        num_faces = face_detector.detectMultiScale(gray_frame)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            roi_gray_frame = roi_gray_frame/255.0
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (96,96)), -1), 0)
            cropped_img_rgb = np.concatenate([cropped_img, cropped_img, cropped_img], axis=-1)
            # predict the emotions
            emotion_prediction = self.emotion_model.predict(cropped_img_rgb, verbose=0)
    #         print(emotion_prediction)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, self.emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        

        #change image matrix to jpg
        ret, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes()