import numpy as np
import keras
import tensorflow as tf
from keras.layers import Input, Conv3D, BatchNormalization, Activation, SpatialDropout3D, MaxPooling3D, \
    TimeDistributed, Flatten, Bidirectional, GRU, Dense
from keras import Model
from data_gen_stanford import DataGenerator

tf.get_logger().setLevel('ERROR')


# https://github.com/rizkiarm/LipNet/blob/master/lipnet/model2.py

class HCI_LipNet:

    def __init__(self, use_trained=True, test_network=False,
                 use_big_network=True):  # todo: Maybe split into multiple networks?
        self.labels = ['blue', 'green', 'red', 'white',
                       'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
        if use_big_network:
            self.name = "HCI_big"
        else:
            self.name = "HCI_small"
        self.setup_GPUs()
        self.model = self.configure_network(n_classes=32, summary=False, big=use_big_network)
        if use_trained:
            self.model.load_weights(self.name + "_weights.h5")

        else:
            self.data_generator = DataGenerator(batch_size=10, val_split=0.99)
            self.model.fit_generator(generator=self.data_generator, epochs=1, shuffle=True,
                                     validation_data=self.data_generator.get_valid_data())
            self.model.save_weights(self.name + "_weights.h5")

        if test_network:
            x, y = self.data_generator.get_valid_data()
            for index, x_ in enumerate(x):
                predictions = self.model.predict(np.reshape(x_, (1, 75, 50, 100, 3)))
                for i, p in enumerate(predictions):
                    difference = np.abs(y[index][i] - p[i])
                    if difference < 0.4:
                        print("pred: {:4f},  true: {:2f}, correct".format(float(p[i]), y[index][i]))
                    else:
                        print("pred: {:4f},  true: {:2f}, ERROR".format(float(p[i]), y[index][i]))

    def predict(self, video):
        video = np.reshape(video, (1, 75, 50, 100, 3))
        prediction = self.model.predict(video)
        return prediction

    def configure_network(self, n_classes=32, summary=False, big=True):
        multiplier = 1
        if big:
            multiplier = 2
        input_layer = Input(name='the_input', shape=(75, 50, 100, 3), dtype='float32')
        x = Conv3D(32 * multiplier, (3, 5, 5), strides=(1, 2, 2), padding="same", kernel_initializer='he_normal',
                   name='conv1')(input_layer)
        x = BatchNormalization(name='batc1')(x)
        x = Activation('relu', name='actv1')(x)
        x = SpatialDropout3D(0.5)(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(x)

        x = Conv3D(64 * multiplier, (3, 5, 5), strides=(1, 1, 1), padding="same", kernel_initializer='he_normal',
                   name='conv2')(x)
        x = BatchNormalization(name='batc2')(x)
        x = Activation('relu', name='actv2')(x)
        x = SpatialDropout3D(0.5)(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(x)

        x = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), padding="same", kernel_initializer='he_normal', name='conv3')(x)
        x = BatchNormalization(name='batc3')(x)
        x = Activation('relu', name='actv3')(x)
        x = SpatialDropout3D(0.5)(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(x)

        x = TimeDistributed(Flatten())(x)

        x = Bidirectional(GRU(128 * multiplier, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'),
                          merge_mode='concat')(x)
        x = Bidirectional(GRU(128, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'),
                          merge_mode='concat')(x)
        x = Flatten()(x)
        outputs = Dense(n_classes, kernel_initializer='he_normal', name='dense1', activation="sigmoid")(x)

        model = Model(inputs=input_layer, outputs=outputs)
        if summary:
            keras.utils.plot_model(model, 'HCI_LipNet.png', show_shapes=True)
            print(self.model.summary())

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, lr=1e-4),
                      metrics=['acc', 'mse', keras.metrics.AUC()])
        return model

    @staticmethod
    def setup_GPUs():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)


if __name__ == "__main__":
    HCI_LipNet(use_trained=False, test_network=True, use_big_network=False)
