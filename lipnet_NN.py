from keras.layers import Input, ZeroPadding3D, Conv3D, BatchNormalization, Activation, SpatialDropout3D, MaxPooling3D, \
    TimeDistributed, Flatten, Bidirectional, GRU, Dense, AveragePooling3D
from keras import Model
import keras
import numpy as np
from data_gen_stanford import DataGenerator


# https://github.com/rizkiarm/LipNet/blob/master/lipnet/model2.py

def get_Lipnet(n_classes=10, summary=False):
    input_layer = Input(name='the_input', shape=(75, 50, 100, 3), dtype='float32')
    network = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), padding="same", kernel_initializer='he_normal', name='conv1')(input_layer)
    network = BatchNormalization(name='batc1')(network)
    network = Activation('relu', name='actv1')(network)
    network = SpatialDropout3D(0.5)(network)
    network = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(network)

    network = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), padding="same", kernel_initializer='he_normal', name='conv2')(network)
    network = BatchNormalization(name='batc2')(network)
    network = Activation('relu', name='actv2')(network)
    network = SpatialDropout3D(0.5)(network)
    network = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(network)

    network = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), padding="same", kernel_initializer='he_normal', name='conv3')(network)
    network = BatchNormalization(name='batc3')(network)
    network = Activation('relu', name='actv3')(network)
    network = SpatialDropout3D(0.5)(network)
    network = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(network)

    #network = TimeDistributed(Flatten())(network)

    #network = Bidirectional(GRU(128, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'),
    #                        merge_mode='concat')(network)
    #network = Bidirectional(GRU(128, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'),
    #                        merge_mode='concat')(network)
    network = Conv3D(64, (1, 1, 1))(network)
    network = AveragePooling3D(pool_size=(2, 2, 2))(network)
    network = Flatten()(network)
    outputs = Dense(n_classes, kernel_initializer='he_normal', name='dense1', activation="sigmoid")(network)

    model = Model(inputs=input_layer, outputs=outputs)
    if summary:
        keras.utils.plot_model(model, 'network.png', show_shapes=True)
        print(model.summary())

    model.compile(optimizer=keras.optimizers.adam(lr=1e-4),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = get_Lipnet(n_classes=51, summary=True)
    datagen = DataGenerator(batch_size=10, val_split=0.99)
    if False:
        model.fit_generator(generator=datagen, epochs=1, shuffle=True, 
            use_multiprocessing=True, workers=6)
    else:
        model.fit_generator(generator=datagen, epochs=1, shuffle=True, validation_data=datagen.get_valid_data())
        