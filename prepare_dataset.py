import cv2
import numpy as np
import os
from collections import Counter

def prepare_videos(filepath = './dataset/cropped_videos/'):
    videos = []
    for index, file in enumerate(sorted(os.listdir(filepath))):
        if index < 119:
            print(filepath + file)
            cap = cv2.VideoCapture(filepath + file)
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

            fc = 0
            ret = True

            while fc < frameCount and ret:
                ret, buf[fc] = cap.read()
                fc += 1

            videos.append(buf.shape)
            cap.release()
    print(np.array(videos).shape)
    print(np.array(videos)[0].shape)

def prepare_labels(filepath = './dataset/alignments/'):
    labels = []
    i = 0
    for index, file in enumerate(sorted(os.listdir(filepath))):
        if i == 0:
            #i += 1
            with open(filepath + file, 'r') as f:
                lines = f.readlines()
                label = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]
            label = strip(label, ['sp','sil'])
            #print(label)
            #print(np.array(label))
            for item in np.array(label).ravel():

                if not str(item).isnumeric():
                    labels.append(item)
    print(np.array(labels).shape)
    cnt = Counter(np.array(labels).ravel())
    for elem in sorted(cnt):
        print(elem)

def strip(align, items):
    return [sub for sub in align if sub[2] not in items]


if __name__ == '__main__':
    #prepare_videos()
    prepare_labels()