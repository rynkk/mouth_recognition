import numpy as np
from tensorflow.keras.utils import Sequence
import pandas as pd
from PIL import Image
import cv2
import os
import random

lipnet_features = ['again', 'at', 'bin', 'blue', 'by', 'green', 'in', 'lay', 'place', 'please',
                   'now', 'red', 'set', 'soon', 'white', 'with',
                   'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'q',
                   'r', 's', 't', 'p', 'u', 'v', 'x', 'y', 'z']

cols = ['videopath', 'blue', 'green', 'red', 'white', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
        'eight', 'nine', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']


class DataGenerator(Sequence):

    def __init__(self, batch_size=10, dim=(75, 50, 100), n_channels=3, n_classes=32, val_split=0.99, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = self.get_dataframe(cols)
        test_train_IDs = list(range(len(self.df.videopath)))
        # remove this shuffle if the validation data is supposed to stay the same over multiple runs
        random.shuffle(test_train_IDs)
        max_train_index = len(test_train_IDs) - int(len(test_train_IDs) * (1 - val_split))
        self.train_IDs = test_train_IDs[0:max_train_index]
        self.valid_IDs = test_train_IDs[max_train_index:]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.test_index = 0

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.train_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.train_IDs[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.train_IDs)

    def __data_generation(self, list_ids_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # Store sample
            y[i] = np.array(self.df.iloc[ID, 1:].values)
            X[i] = np.array(self.load_video(self.df.iloc[ID, 0]))
        return X, y

    def get_dataframe(self, cols, alignpath='./dataset/alignments/', videopath='./dataset/cropped_videos/'):
        if 'train.csv' in os.listdir('./dataset/'):
            df = pd.read_csv('./dataset/train.csv', sep='\t')
            return df[cols]



        # ONLY DURING FIRST PREPARATION OF THE DATA SET
        # OBSOLETE IF train.csv WAS ALREADY PERPARED
        r = {'videopath': []}
        for po in cols:
            r[po] = []
        video_paths = sorted(os.listdir(videopath))
        broken_index = 0
        for index, file in enumerate(sorted(os.listdir(alignpath))):
            print(str(np.round(index / len(os.listdir(alignpath)) * 100, decimals=1)), end="\r")
            with open(alignpath + file, 'r') as f:
                cap = cv2.VideoCapture(videopath + video_paths[index])
                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frameCount == 75:
                    lines = f.readlines()
                    label = [y[2] for y in [x.strip().split(" ") for x in lines]]
                    label = set(label)
                    label.remove('sil')
                    r['videopath'].append(videopath + video_paths[index])
                    for item in cols:
                        if item in label:
                            r[item].append(1)
                        else:
                            r[item].append(0)
                else:
                    broken_index += 1
        print("There are " + str(broken_index) + " broken files")
        r = pd.DataFrame.from_dict(r)
        r.to_csv("dataset/train.csv", sep='\t')
        return r

    def load_video(self, filepath):
        cap = cv2.VideoCapture(filepath)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

        for count in range(frame_count):
            ret, buf[count] = cap.read()
        cap.release()
        #print("loaded_videos = " + str(self.test_index))
        self.test_index += 1
        return buf

    def get_valid_data(self):
        X = np.empty((len(self.valid_IDs), *self.dim, self.n_channels))
        y = np.empty((len(self.valid_IDs), self.n_classes), dtype=int)
        for index, ID in enumerate(self.valid_IDs):
            y[index] = np.array(self.df.iloc[ID, 1:].values)
            X[index,] = np.array(self.load_video(self.df.iloc[ID, 0]))
        return X, y
