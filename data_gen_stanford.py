import numpy as np
from tensorflow.keras.utils import Sequence
import pandas as pd
from PIL import Image
import cv2
import os

lipnet_features = ['again', 'at', 'bin', 'blue', 'by', 'eight', 'five', 'four', 'green', 'in',
              'lay', 'place', 'please', 'nine', 'now', 'one', 'red', 'set', 'seven', 'six', 'soon',
              'three', 'two', 'white', 'with', 'zero', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
              'j', 'k', 'l', 'm', 'n', 'o', 'q', 'r', 's', 't', 'p', 'u', 'v', 'x', 'y', 'z']

class DataGenerator(Sequence):

    def __init__(self, batch_size=10, dim=(75,50,100), n_channels=3, n_classes=51, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = self.get_dataframe(lipnet_features)
        print(self.df.head(10))
        self.list_IDs = list(range(len(self.df.videopath)))
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            y[i]  = np.array(self.df.iloc[ID, 2:].values)
            X[i,] = np.array(self.load_video(self.df.iloc[ID, 1]))

        return X, y



    def get_dataframe(self, poss_labels, alignpath='./dataset/alignments/', videopath='./dataset/cropped_videos/'):
        if 'train.csv' in os.listdir('./dataset/'):
            return pd.read_csv('./dataset/train.csv', sep='\t')

        r = {'videopath': []}
        for po in poss_labels:
            r[po] = []
        video_paths = sorted(os.listdir(videopath))
        broken_index = 0
        for index, file in enumerate(sorted(os.listdir(alignpath))):
            print(str(np.round(index/len(os.listdir(alignpath)) *100, decimals=1)), end="\r")
            with open(alignpath + file, 'r') as f:
                cap = cv2.VideoCapture(videopath + video_paths[index])
                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frameCount == 75:
                    lines = f.readlines()
                    label = [y[2] for y in [x.strip().split(" ") for x in lines]]
                    label = set(label)
                    label.remove('sil')
                    r['videopath'].append(videopath + video_paths[index])
                    for item in poss_labels:
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
        print(filepath)
        cap = cv2.VideoCapture(filepath)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

        for count in range(frameCount):
            ret, buf[count] = cap.read()
        cap.release()
        return buf / 255