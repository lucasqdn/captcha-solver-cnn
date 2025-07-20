import tensorflow as tf
from tensorflow import keras

def dataset_prep():
    train_data = tf.keras.utils.image_dataset_from_directory(
        "data/processed",
        validation_split = 0.2,
        subset = "training",
        seed = 69,
        labels = "inferred",
        label_mode = "categorical",
        color_mode = "grayscale",
        image_size = (28,28),
        batch_size = 16
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        "data/processed",
        validation_split = 0.2,
        subset = "validation",
        seed = 69,
        labels = "inferred",
        label_mode = "categorical",
        color_mode = "grayscale",
        image_size = (28,28),
        batch_size = 16
    )

    normalization = keras.layers.Rescaling(1./255)
    train_data = train_data.map(lambda x, y: (normalization(x), y))
    test_data = test_data.map(lambda x, y: (normalization(x), y))

    return train_data, test_data