import numpy as py
import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_model():
    model = keras.Sequential([
        layers.Convolution2D(32, (3,3), padding = "same", input_shape=(28,28,1), activation="relu"),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Convolution2D(64, (3, 3), padding = "same", activation="relu"),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(32, activation="softmax"),
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model