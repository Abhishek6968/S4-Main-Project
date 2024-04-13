from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import os
from violencemodel import *
from flask import Flask , request , jsonify , Response
from PIL import Image
from io import BytesIO
import time
from skimage.transform import resize
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import  Dropout, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from collections import deque
import numpy as np
import argparse
import pickle
import streamlit as st
import tempfile
from skimage.transform import resize


# Load your trained model
model = souhaiel_model(tf)

# Define some constants
ok = 'Normal'
okk = 'Violence'

def process_frames(frames):
    # Resize each frame and stack them to form a sequence
    processed_frames = []
    for frame in frames:
        frame = resize(frame, (160, 160, 3))
        processed_frames.append(frame)
    processed_frames = np.array(processed_frames)
    
    # Stack frames to form a sequence of 30 frames
    num_frames = len(processed_frames)
    if num_frames < 30:
        # Pad with duplicates if the video is too short
        processed_frames = np.concatenate([processed_frames] * (30 // num_frames + 1))
        processed_frames = processed_frames[:30]
    elif num_frames > 30:
        # If video has more than 30 frames, take the first 30 frames
        processed_frames = processed_frames[:30]
    
    if np.max(processed_frames) > 1:
        processed_frames = processed_frames / 255.0
    
    return np.expand_dims(processed_frames, axis=0)
# Streamlit app
def main():
    st.title("Violence Detection in Videos")

    # Flag to control display of upload widget and message
    upload_displayed = False

    while True:
        if not upload_displayed:
            st.write("Upload a video file to detect violence.")
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4"], key="file_uploader_1")
            upload_displayed = True

        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())

            vs = cv2.VideoCapture(temp_file.name)
            fps = vs.get(cv2.CAP_PROP_FPS)
            Q = deque(maxlen=128)

            stframe = st.empty()  # Placeholder to display the video
            stop_button_displayed = False  # Flag to track whether stop button has been displayed

            while True:
                grabbed, frm = vs.read()

                if not grabbed:
                    break

                frames = []
                for _ in range(30):
                    rval, frame = vs.read()
                    if not rval:
                        break
                    frames.append(frame)

                datav = process_frames(frames)

                preds = model.predict(datav)
                prediction = preds.argmax(axis=1)
                Q.append(preds)

                results = np.array(Q).mean(axis=0)
                maxprob = np.max(results)
                i = np.argmax(results)
                rest = 1 - maxprob
                diff = maxprob - rest
                th = 100 if diff > 0.80 else diff

                if preds[0][1] < th:
                    text = "Alert : {} - {:.2f}%".format(ok, 100 - (maxprob * 100))
                else:
                    text = "Alert : {} - {:.2f}%".format(okk, maxprob * 100)

                frm = cv2.putText(frm, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
                stframe.image(frm, channels="BGR")
                
                # Display stop button only once
                if not stop_button_displayed:
                    if st.button("Stop", key="stop_button"):
                        break
                    stop_button_displayed = True

                # Check if user pressed 'q' to stop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            vs.release()
            break  # Break out of the outer loop once the video is processed

if __name__ == "__main__":
    main()