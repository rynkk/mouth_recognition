import cv2
import dlib
import os
import sys
import time
import numpy as np
import math
from imutils import face_utils

import os
"""
Move this script and the facial recoginiton datasets into the folder containing the .mpg files. It will automatically search all subdirectories for mpgs

"""


# select files path
file_dir = os.path.dirname(os.path.realpath(__file__))

paths = {}
total_clips = 0

print("\nSearching files..")
for root, dirs, files in os.walk(file_dir, topdown=False):
    for name in files:
        if name.endswith('.mpg') and not name.startswith("cropped_"):
            paths.setdefault(root, []).append(name)
            total_clips +=1 
print("Success: Found %s .mpg files\n" % total_clips)

print("Loading facial recognition files")
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
print("Done\n")

print("Cropping files")
(mouth_lower, mouth_upper) = (48,60)

files_processed = 0
for f_dir in paths:
    if not os.path.exists(f_dir+"/cropped"):
        try:
            os.mkdir(f_dir+"/cropped")
        except OSError:
            print ("Creation of the directory %s failed" % (f_dir+"/cropped"))

    for f_name in paths[f_dir]:

        print("Progress: [%i out of %i]" % (files_processed, total_clips))
        sys.stdout.write("\033[F") # Cursor up one line
        cap = cv2.VideoCapture(os.path.join(f_dir, f_name))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        f_name_avi = f_name[:-3]+"avi"
        writer = cv2.VideoWriter(os.path.join(f_dir+"/cropped", "cropped_"+f_name_avi),fourcc, fps, (100,50))

        counter = 0
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                files_processed +=1
                break
            rects = detector(frame, 0)
            if counter == 0:
                counter += 1
                for (i, rect) in enumerate(rects):
                    shape = sp(frame, rect)
                    shape = face_utils.shape_to_np(shape)
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[mouth_lower:mouth_upper]]))
            else:
                for (i, rect) in enumerate(rects):
                    shape = sp(frame, rect)
                    shape = face_utils.shape_to_np(shape)
                    (x, y, abcd, efgh) = cv2.boundingRect(np.array([shape[mouth_lower:mouth_upper]]))
            
            frame = frame[y-10 : y+h+10, x-5 : x+w+5]
            frame = cv2.resize(frame, (100, 50), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        writer.release()