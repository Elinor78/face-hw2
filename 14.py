from keras.models import Sequential
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from data_util import load_data
from keras.optimizers import Adam
from keras import backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import csv

def _reshape(data):
    '''
    channel position depends on backend image format
    returns data in correct shape divided by 255
    returns input_shape for use in first cnn layer
    '''
    if backend.image_data_format() == 'channels_first':
        data = data.reshape(data.shape[0], 1, 48, 48)
        input_shape = (1, 48, 48)
    else:
        data = data.reshape(data.shape[0], 48, 48, 1)
        input_shape = (48, 48, 1)

    data = data.astype('float32')
    data /= 255

    return data, input_shape

def _cnn():

    train_data, train_target, test_data = load_data()# load data from utility
    train_data,validation_data,train_target,validation_target = train_test_split(train_data,train_target, test_size=0.2, random_state=42) #randomly split for validation

    test_data, input_shape = _reshape(test_data)# see docstring
    train_data, input_shape = _reshape(train_data)# see docstring
    validation_data, input_shape = _reshape(validation_data)# see docstring

    model = Sequential()# sequential model

    model.add(convolutional.Conv2D(# first convolitional layer
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        input_shape=input_shape,
        activation='relu'
    ))


    model.add(convolutional.Conv2D(64, (3,3), padding='same'))# 2nd convo layer
    model.add(Activation('relu'))#using relu
    model.add(pooling.MaxPooling2D(pool_size=(2,2), padding='same'))# 1st pooling

    model.add(convolutional.Conv2D(64, (3,3), padding='same'))# 3rd convo layer
    model.add(Activation('relu'))# using relu
    model.add(pooling.MaxPooling2D(pool_size=(2,2), padding='same'))# 2nd pooling

    model.add(convolutional.Conv2D(64, (3,3), padding='same'))# 4th convo layer
    model.add(Activation('relu'))# using relu
    model.add(pooling.MaxPooling2D(pool_size=(2,2), padding='same'))# 3nd pooling

    model.add(Dropout(0.25))# prevent overfit w/dropout 1
    model.add(Flatten())# flatten for dnn
    model.add(Dense(1024))# 1st dnn layer
    model.add(Activation('relu'))# using relu

    model.add(Dropout(0.5))# prevent overfit w/dropout 2
    model.add(Dense(3))# output dnn layer
    model.add(Activation('softmax'))# using softmax

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)# adam optimizer

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])# compile

    model.fit(train_data, train_target, epochs=12, batch_size=128, verbose=1)# fit using training data

    loss, accuracy = model.evaluate(validation_data,validation_target,verbose=0)# evaluate using validation data

    print "accuracy: {}".format(accuracy)
    output = model.predict_classes(test_data)# predict on test_data

    return output

if __name__ == '__main__':
    class_output = _cnn()

    with open('possible_submissions/fourteen.csv', 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(class_output)

