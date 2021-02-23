import cv2
import sys

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import decode_predictions

import numpy as np

def video_capture():

    model = VGGFace(model='resnet50')
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

        # Scalefactor diminshes the image by a given factor, (10% in this case)
        faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.1,
        minNeighbors = 5)

        # x,y represents the upper left cornet of the face
        # w,h represents the widht and height
        # we create a rectangle with the rectangle function to frame the face
        for x,y,w,h in faces:
            frame_face = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3);
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
        cropped = frame[y:y+h, x:x+w]


        width = 224
        height = 224
        dim = (width, height)

        cropped_resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)

        cropped_resized = np.expand_dims(cropped_resized, axis=0)

        yhat = model.predict(cropped_resized)

        # convert prediction into names
        results = decode_predictions(yhat)
        # display most likely results
        for result in results[0]:
        	print('%s: %.3f%%' % (result[0], result[1]*100))

        #except:
        #    pass
        #cropped = faces[[0]:[1],[2]:[3]]

        # display the imageq
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
