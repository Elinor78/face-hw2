from keras.models import Sequential
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from data_util import load_data, write_csv
from keras.optimizers import Adam
from keras import backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import csv

def _reshape(data):
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

    train_data, train_target, test_data = load_data()
    train_data,validation_data,train_target,validation_target = train_test_split(train_data,train_target, test_size=0.2, random_state=42)

    test_data, input_shape = _reshape(test_data)
    train_data, input_shape = _reshape(train_data)
    validation_data, input_shape = _reshape(validation_data)

    model = Sequential()

    model.add(convolutional.Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        input_shape=input_shape,
        activation='relu'
    ))

    model.add(pooling.MaxPooling2D(
        pool_size=(2,2),
        padding='same'
    ))

    model.add(convolutional.Conv2D(64, (5,5), padding='same'))
    model.add(Activation('relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add(convolutional.Conv2D(64, (5,5), padding='same'))
    model.add(Activation('relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add(convolutional.Conv2D(64, (5,5), padding='same'))
    model.add(Activation('relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add(convolutional.Conv2D(64, (5,5), padding='same'))
    model.add(Activation('relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_target, epochs=20, batch_size=64, verbose=2)

    loss, accuracy = model.evaluate(validation_data,validation_target,verbose=2)

    print "accuracy: {}".format(accuracy)
    output = model.predict_classes(test_data)

    return output

if __name__ == '__main__':
    class_output = _cnn()

    with open('possible_submissions/ten.csv', 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(class_output)

