import tensorflow as tf
from tensorflow import keras

def train_model(train_data, test_data, model):
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)
    history = model.fit(
        train_data,
        validation_data = test_data,
        epochs = 10
    )
    return history