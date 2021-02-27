import cv2
import sys
import tensorflow as tf
from tensorflow import keras
import copy
from keras.models import load_model
from keras.optimizers import Adam


import numpy as np

def video_capture():
    opt = Adam(lr=0.0001)
    model = load_model('model_weights.h5')
    model.summary()


    print('Inputs: %s' % model.inputs)
    print('Outputs: %s' % model.outputs)

    # Create and initialize a face cascacde. This loads the face cascade into memory
    # so it is ready for use.
    # The cascade is an XML file that contins the data to detect faces.
    # Haar Cascade is a classifier that is used to identify specific objects
    # in this case it is for frontal faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

    # We use a function called VideoCapture to capture the webcam's video feed
    #video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video = cv2.VideoCapture(0)

    # Create a loop that will perform operations on every single frame
    while True:
        # Extracting Frames
        check, frame = video.read()


        img = copy.deepcopy(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Scalefactor diminshes the image by a given factor, (10% in this case)
        faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 5)
        faces_gray = face_cascade.detectMultiScale(gray, 1.3, 5)

        # x,y represents the upper left cornet of the fa
        # w,h represents the widht and height
        # we create a rectangle with the rectangle function to frame the face
        for x,y,w,h in faces:
            frame_face = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3);


        for (x,y,w,h) in faces_gray:
            fc = gray[y:y+h, x:x+w]

            roi = cv2.resize(fc, (56,56))
            pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
            print(pred)
            text_idx=np.argmax(pred)

            text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            if text_idx == 0:
                text= text_list[0]
            if text_idx == 1:
                text= text_list[1]
            elif text_idx == 2:
                text= text_list[2]
            elif text_idx == 3:
                text= text_list[3]
            elif text_idx == 4:
                text= text_list[4]
            elif text_idx == 5:
                text= text_list[5]
            elif text_idx == 6:
                text= text_list[6]
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

video_capture()
