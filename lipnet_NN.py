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


# https://github.com/rizkiarm/LipNet/blob/master/lipnet/model2.py

def get_Lipnet(n_classes=10, summary=False):
    input_layer = Input(name='the_input', shape=(75, 50, 100, 3), dtype='float32')
    network = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), padding="same", kernel_initializer='he_normal', name='conv1')(
        input_layer)
    network = BatchNormalization(name='batc1')(network)
    network = Activation('relu', name='actv1')(network)
    network = SpatialDropout3D(0.5)(network)
    network = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(network)

    network = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), padding="same", kernel_initializer='he_normal', name='conv2')(
        network)
    network = BatchNormalization(name='batc2')(network)
    network = Activation('relu', name='actv2')(network)
    network = SpatialDropout3D(0.5)(network)
    network = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(network)

    network = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), padding="same", kernel_initializer='he_normal', name='conv3')(
        network)
    network = BatchNormalization(name='batc3')(network)
    network = Activation('relu', name='actv3')(network)
    network = SpatialDropout3D(0.5)(network)
    network = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(network)

    network = TimeDistributed(Flatten())(network)

    network = Bidirectional(GRU(128, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'),
                            merge_mode='concat')(network)
    network = Bidirectional(GRU(128, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'),
                            merge_mode='concat')(network)
    network = Flatten()(network)
    outputs = Dense(n_classes, kernel_initializer='he_normal', name='dense1', activation="softmax")(network)

    model = Model(inputs=input_layer, outputs=outputs)
    if summary:
        keras.utils.plot_model(model, 'network.png', show_shapes=True)
        print(model.summary())

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, lr=1e-4),
                  metrics=['acc', 'mse', keras.metrics.AUC()])
    return model


if __name__ == '__main__':
    model = get_Lipnet(n_classes=51, summary=False)

    data_gen = DataGenerator(batch_size=10, val_split=0.99)
    model.fit_generator(generator=data_gen, epochs=1, shuffle=True, validation_data=data_gen.get_valid_data())
    model.save("model")
    print(model.evaluate(data_gen.get_valid_data(), batch_size=10, steps=1))


































































































































































































































































































































