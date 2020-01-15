from keras.layers import Input, ZeroPadding3D, Conv3D, BatchNormalization, Activation, SpatialDropout3D, MaxPooling3D, \
    TimeDistributed, Flatten, Bidirectional, GRU, Dense, AveragePooling3D
from keras import Model
import tensorflow as tf
import keras
import numpy as np
from data_gen_stanford import DataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

'''Labels:
 ['blue', 'green','red', 'white',
   'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q','r'
], 32
'''


# https://github.com/rizkiarm/LipNet/blob/master/lipnet/model2.py

def get_Lipnet(n_classes=32, summary=False):
    input_layer = Input(name='the_input', shape=(75, 50, 100, 3), dtype='float32')
    x = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), padding="same", kernel_initializer='he_normal', name='conv1')(
        input_layer)
    x = BatchNormalization(name='batc1')(x)
    x = Activation('relu', name='actv1')(x)
    x = SpatialDropout3D(0.5)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(x)

    x = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), padding="same", kernel_initializer='he_normal', name='conv2')(x)
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

    x = Bidirectional(GRU(128, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'),
                            merge_mode='concat')(x)
    x = Bidirectional(GRU(128, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'),
                            merge_mode='concat')(x)
    x = Flatten()(x)
    outputs = Dense(n_classes, kernel_initializer='he_normal', name='dense1', activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=outputs)
    if summary:
        keras.utils.plot_model(model, 'x.png', show_shapes=True)
        print(model.summary())

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, lr=1e-4),
                  metrics=['acc', 'mse', keras.metrics.AUC()])
    return model


def get_Lipnet_no_GRU(n_classes=10, summary=False):
    input_layer = Input(name='the_input', shape=(75, 50, 100, 3), dtype='float32')
    x = MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2), name='max1')(input_layer)
    x = BatchNormalization(name='bach_norm_1')(x)
    x = Activation('relu', name='actv1')(x)
    x = Conv3D(64, (3, 3, 3), padding="same", kernel_initializer='he_normal', name='conv1', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='max2')(x)
    x = BatchNormalization(name='bach_norm_2')(x)
    x = Activation('relu', name='actv2')(x)

    x = Conv3D(128, (1, 1, 1), strides=(1, 1, 1), padding="same", kernel_initializer='he_normal', name='conv2')(x)
    x = Conv3D(128, (3, 3, 3), strides=(1, 2, 2), padding="same", kernel_initializer='he_normal', name='conv3')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(x)

    x = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), padding="same", kernel_initializer='he_normal', name='conv4')(x)
    x = BatchNormalization(name='batc3')(x)
    x = Activation('relu', name='actv3')(x)
    x = SpatialDropout3D(0.5)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max4')(x)
    x = Flatten()(x)
    outputs = Dense(n_classes, kernel_initializer='he_normal', name='dense1', activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=outputs)
    if summary:
        keras.utils.plot_model(model, 'network.png', show_shapes=True)
        print(model.summary())

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, lr=1e-4),
                  metrics=['acc', 'mse', keras.metrics.AUC()])
    return model


if __name__ == '__main__':
    from keras.models import load_model
    model = get_Lipnet(n_classes=32, summary=False)
    data_gen = DataGenerator(batch_size=10, val_split=0.99)
    #model.fit_generator(generator=data_gen, epochs=1, shuffle=True, validation_data=data_gen.get_valid_data())
    #model.save_weights('model_weights.h5')
    model.load_weights('model_weights.h5')
    x, y = data_gen.get_valid_data()
    for index, x_ in enumerate(x):
        pred = model.predict(np.reshape(x_, (1, 75, 50, 100, 3)))
        for i, p in enumerate(pred):
            print("pred: {:4f},  true: {:2f}, diff: {:3f}".format(float(p[i]), y[index][i], float(y[index][i] - p[i])))


