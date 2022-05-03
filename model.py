from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import image
import numpy as np


def create_model():
    model = Sequential()
    model.add(Dense(800, input_dim=784, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile()
    model.load_weights('mnist.h5')
    return model


def get_pred(img, model):
    a = image.load_img(img, color_mode='grayscale', target_size=(28,28))
    a = image.img_to_array(a)
    a = a.reshape(784)
    a = a / 255
    a = np.expand_dims(a, axis=0)
    prediction = model.predict(a)
    pred = np.argmax(prediction)
    return pred

