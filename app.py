import cv2
import dlib
import os
import time
import numpy as np
import math
from imutils import face_utils

def euclidian_distance(p1, p2):
    diff_x = abs(p2[0]-p1[0])
    diff_y = abs(p2[1]-p1[1])
    return math.sqrt(diff_x*diff_x + diff_y*diff_y)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#(mouth_lower, mouth_upper) = face_utils.FACIAL_LANDMARKS_68_IDXS['mouth']
(mouth_lower, mouth_upper) = (48,60)

ACTIVATION_RATIO = 3.0
RECORDING_TIME = 2.0
ACTIVATION_TIME = 0.5


activated = False
recording = False
color = (0,0,255) #red


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    mouth = frame
    rects = detector(frame, 0)
    
    # Our operations on the frame come here
    for (i, rect) in enumerate(rects):

        shape = sp(frame, rect)
        shape = face_utils.shape_to_np(shape)

        # https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg #
        left = (shape[48, 0], shape[48,1])
        right = (shape[54, 0], shape[54,1])
        
        top = (shape[51, 0], shape[51,1])
        bottom = (shape[57, 0], shape[57,1])

        diff_h = euclidian_distance(top, bottom)
        diff_w = euclidian_distance(left, right)

        ratio = round(diff_w/diff_h, 5)

        if ratio >= ACTIVATION_RATIO and not recording:  # mouth shut & not recording
            print("mouth shut\r", end='') 
            activated = False   
            color = (0,0,255) #yellow
        elif recording:
            print("recording \r", end='') 
            now_rec = time.perf_counter()               
            color = (0,255,0) #yellow
            if now_rec - now_active > 1: # recording for 5s
                recording = False 
        elif activated:
            print("checking  \r", end='')   
            color = (0,255,255) #yellow
            now_active = time.perf_counter()
            if now_active - when_opened > 0.5:
                recording = True
                activated = False
        else:   # Mouth open
            print("mouth open\r", end='')
            when_opened = time.perf_counter()
            activated = True
        
        """if recording:
            print("recording")
            now_rec = time.perf_counter()
            print("now_active: %s", now_rec)
            print("now_rec: %s", now_active)
            if now_rec - now_active > 5: # recording for 5s
                recording = False
                activated = False
        elif activated:
            if ratio >= ACTIVATION_RATIO:
                activated = False
            else:
            print("activated")
            now_active = time.perf_counter()
            print("seconds: %s", seconds)
            print("now_active: %s", now_active)
            if now_active-seconds > 1:
                color = (0,255,0) #green
                recording = True
        elif ratio < ACTIVATION_RATIO:
            print("mouth open")
            activated = True
            color = (0,255,255) #yellow
            seconds = time.perf_counter()
        else:
            print("mouth shut")
            activated = False
            color = (0,0,255) #red
        """
        cv2.line(frame, left, right, color)
        cv2.line(frame, top, bottom, color)

        

        #fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) # not supported by my webcam

        cv2.putText(frame, "ratio: %s" % ratio, (0, height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

        for (x, y) in shape[mouth_lower:mouth_upper]:
            cv2.circle(frame, (x, y), 1, color, -1)
        (x, y, w, h) = cv2.boundingRect(np.array([shape[mouth_lower:mouth_upper]]))
        mouth = frame[y-25 : y+h+25, x-25 : x+w+25]
    

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('mouth',mouth)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
