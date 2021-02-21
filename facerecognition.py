import cv2
import sys


def video_capture():
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
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3);

        # display the image
        cv2.imshow('Face Detector', frame);

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
