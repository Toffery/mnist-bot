from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import BatchNormalization, MaxPooling2D, Dropout
from keras.preprocessing import image
import numpy as np


def create_conv_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    # model.compile()
    model.load_weights('conv_mnist.h5')
    return model


def get_conv_pred(img, model):
    a = image.load_img(img, color_mode='grayscale', target_size=(28, 28, 1))
    a = image.img_to_array(a)
    # a = a / 255
    a = np.expand_dims(a, axis=0)
    prediction = model.predict(a)
    pred = np.argmax(prediction)
    print(prediction)
    return pred
