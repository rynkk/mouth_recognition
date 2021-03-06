import cv2
import dlib
import control
import os
import time
import numpy as np
import math
from imutils import face_utils

from lipnet_NN import HCI_LipNet as daniels_netz

def euclidian_distance(p1, p2):
    diff_x = abs(p2[0]-p1[0])
    diff_y = abs(p2[1]-p1[1])
    return math.sqrt(diff_x*diff_x + diff_y*diff_y)

model = daniels_netz()
ui_control = control.ControlUnit()

transl=['blue', 'green', 'red', 'white',
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS) # not supported by my webcam
print(fps)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#(mouth_lower, mouth_upper) = face_utils.FACIAL_LANDMARKS_68_IDXS['mouth']
(mouth_lower, mouth_upper) = (48,60)

ACTIVATION_RATIO = 3.0
RECORDING_TIME = 2.0
ACTIVATION_TIME = 0.5

frames = []
current_frame = 0
first_frame = True
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
        
        cv2.line(frame, left, right, color)
        cv2.line(frame, top, bottom, color)

        cv2.putText(frame, "ratio: %s" % ratio, (0, height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

        for (x, y) in shape[mouth_lower:mouth_upper]:
            cv2.circle(frame, (x, y), 1, color, -1)
        
        if first_frame:
            (x, y, w, h) = cv2.boundingRect(np.array([shape[mouth_lower:mouth_upper]]))
            first_frame = False
        else:
            (x, y, abcd, efgh) = cv2.boundingRect(np.array([shape[mouth_lower:mouth_upper]]))


        mouth = frame[y-10 : y+h+10, x-5 : x+w+5]
        mouth = cv2.resize(mouth, (100, 50), interpolation=cv2.INTER_LINEAR)

        

        if ratio >= ACTIVATION_RATIO and not recording:  # mouth shut & not recording
            #print("mouth shut\r", end='') 
            activated = False   
            color = (0,0,255) #red
        elif recording:
            #print("recording \r", end='') 
            now_rec = time.perf_counter()
            color = (0,255,0) #green

            #print("\n")
            print("current_frame %d\r" % current_frame)

            if current_frame == 75:
                #print(np.array(frames))
                # print(np.array(frames).shape)
                #pred_color, pred_number, lettera_h, letteri_r = model.predict(frames)
                print(model.predict(np.array(frames)))

                ui_control.controlfunction(model.predict(np.array(frames)))
                recording = False
                frames = []
            else:
                frames.append(mouth)
            current_frame += 1


        elif activated:
            #print("checking  \r", end='')   
            color = (0,255,255) #yellow
            now_active = time.perf_counter()
            if now_active - when_opened > 0.5:
                recording = True
                current_frame = 0
                activated = False
        else:   # Mouth open
            #print("mouth open\r", end='')
            when_opened = time.perf_counter()
            activated = True
    

    # Display the resulting frame
    #cv2.imshow('frame',frame)
    cv2.imshow('mouth',mouth)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ui_control.mainloop()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
