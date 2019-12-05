import cv2
import numpy as np
import os
from collections import Counter
import pandas as pd


def prepare_videos(filepath='./dataset/cropped_videos/'):
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

            videos.append(buf)
            cap.release()

    print(np.array(videos).shape)
    print(np.array(videos)[0])
    print(np.array(videos)[0].shape)
    print(np.array(videos)[0][0].shape)


def prepare_labels(poss_labels, alignpath='./dataset/alignments/', videopath='./dataset/cropped_videos/'):
    r = {'videopath': []}
    for po in poss_labels:
        r[po] = []
    video_paths = sorted(os.listdir(videopath))
    for index, file in enumerate(sorted(os.listdir(alignpath))):
        with open(alignpath + file, 'r') as f:
            lines = f.readlines()
            label = [y[2] for y in [x.strip().split(" ") for x in lines]]
            label = set(label)
            label.remove('sil')
            r['videopath'].append(video_paths[index])
            for item in poss_labels:
                if item in label:
                    r[item].append(1)
                else:
                    r[item].append(0)
    return r


if __name__ == '__main__':
    # prepare_videos()
    labels = ['again', 'at', 'bin', 'blue', 'by', 'eight', 'five', 'four', 'green', 'in',
              'lay', 'place', 'please', 'nine', 'now', 'one', 'red', 'set', 'seven', 'six', 'soon',
              'three', 'two', 'white', 'with', 'zero', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
              'j', 'k', 'l', 'm', 'n', 'o', 'q', 'r', 's', 't', 'p', 'u', 'v', 'x', 'y', 'z']
    df = pd.DataFrame.from_dict(prepare_labels(labels))
    print(df)
    prepare_videos()


