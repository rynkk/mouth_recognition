from keras.layers import Input, ZeroPadding3D, Conv3D, BatchNormalization, Activation, SpatialDropout3D, MaxPooling3D, \
    TimeDistributed, Flatten, Bidirectional, GRU, Dense
from keras import Model
import keras
import numpy as np


# https://github.com/rizkiarm/LipNet/blob/master/lipnet/model2.py

def get_Lipnet(n_classes=10, summary=False):
    input_layer = Input(name='the_input', shape=(75, 50, 100, 3), dtype='float32')
    network = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(input_layer)
    network = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(network)
    network = BatchNormalization(name='batc1')(network)
    network = Activation('relu', name='actv1')(network)
    network = SpatialDropout3D(0.5)(network)
    network = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(network)

    network = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(network)
    network = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(network)
    network = BatchNormalization(name='batc2')(network)
    network = Activation('relu', name='actv2')(network)
    network = SpatialDropout3D(0.5)(network)
    network = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(network)

    network = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(network)
    network = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(network)
    network = BatchNormalization(name='batc3')(network)
    network = Activation('relu', name='actv3')(network)
    network = SpatialDropout3D(0.5)(network)
    network = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(network)

    network = TimeDistributed(Flatten())(network)

    network = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'),
                            merge_mode='concat')(network)
    network = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'),
                            merge_mode='concat')(network)

    outputs = Dense(n_classes, kernel_initializer='he_normal', name='dense1', activation="softmax")(network)

    model = Model(inputs=input_layer, outputs=outputs)
    if summary:
        print(model.summary())

    model.compile(optimizer=keras.optimizers.adam(lr=1e-4),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = get_Lipnet(n_classes=20, summary=True)
    model.fit()
