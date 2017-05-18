import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, convolutional, pooling

from keras import backend
from data_util import load_data
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
    train_data,validation_data,train_target,validation_target = train_test_split(train_data,train_target, test_size=0.2, random_state=42)#r andomly split data into traingin and validation sets

    test_data, input_shape = _reshape(test_data)# see docstring
    train_data, input_shape = _reshape(train_data)# see docstring
    validation_data, input_shape = _reshape(validation_data)# see docstring

    model = Sequential()# sequential model

    model.add(convolutional.Conv2D(# first convolitional layer
        filters=32, 
        kernel_size=(3, 3),
        activation='relu',
        input_shape=input_shape
    ))

    model.add(convolutional.Conv2D(64, (3, 3), activation='relu'))# 2nd convo layer, using relu for activation
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))#1st pooling

    model.add(Dropout(0.25))# prevent overfit w/dropout 1
    model.add(Flatten())# flatten for dnn
    model.add(Dense(128, activation='relu'))# 1st dnn layer

    model.add(Dropout(0.5))# prevent overfit w/dropout 2
    model.add(Dense(3, activation='softmax'))# using softmax for activation

    model.compile(# compile using Adadelta for optimizer
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy']
        )

    model.fit(train_data, train_target, batch_size=128, epochs=12, verbose=1)# fit using training data
    loss, accuracy = model.evaluate(validation_data, validation_target, verbose=0)# evaluate using validation data
    print "accuracy: {}".format(accuracy)

    class_output = model.predict_classes(test_data)# predict on test_data
    return class_output

if __name__ == '__main__':
    class_output = _cnn()
    with open('possible_submissions/twelve.csv', 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(class_output)
