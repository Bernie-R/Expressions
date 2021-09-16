import cv2
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import copy
from keras.models import load_model
from keras.optimizers import Adam
import time


import numpy as np

model = load_model(r"C:\Users\berna\Documents\GitHub\Emotiondetection/")
model.load_weights(r"C:\Users\berna\Documents\GitHub\Emotiondetection/model_weights.h5")

def video_capture():

    print('Inputs: %s' % model.inputs)
    print('Outputs: %s' % model.outputs)

    # Create and initialize a face cascacde. This loads the face cascade into memory
    # so it is ready for use.
    # The cascade is an XML file that contins the data to detect faces.
    # Haar Cascade is a classifier that is used to identify specific objects
    # in this case it is for frontal faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # We use a function called VideoCapture to capture the webcam's video feed
    #video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video = cv2.VideoCapture(0)

    # Create a loop that will perform operations on every single frame
    while True:
        # Extracting Frames
        check, frame = video.read()
        img = copy.deepcopy(frame)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Scalefactor diminshes the image by a given factor, (10% in this case)
        faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 5)

        # x,y represents the upper left cornet of the fa
        # w,h represents the widht and height
        # we create a rectangle with the rectangle function to frame the face

        for x,y,w,h in faces:
            frame_face = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3);
            #faces_gray = face_cascade.detectMultiScale(gray, 1.3, 5)

            gray = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
            fc = gray[y:y+h, x:x+w]

            roi = cv2.resize(fc, (56,56))
            pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
            text_idx=np.argmax(pred)
            text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            text = text_list[text_idx]
            print(text)

            cv2.putText(frame, text, (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
            img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)


        try:
            x = faces[0][0]
            y = faces[0][1]
            w = faces[0][2]
            h = faces[0][3]
        except:
            x = 1
            y = 1
            w = 1
            h = 1

        # display the image
        cv2.imshow('Face Detector', frame_face)

        # To continue to display the image, we need to use waitKey
        # On detecting the "q" key being pressed while OpenCv window is active,
        # the window will close.
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    #Once we are outside of the loop, we release the video capture mechanism
    # and close all opencv windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_capture()
