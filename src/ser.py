__version__ = "0.1.0"

import tensorflow as tf
# import tensorflow_cloud as tfc
import load
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


def bam():
    model = keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dense(2, activation='sigmoid'),
    ])
    return model


def short_bam():
    model = keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='tanh', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='tanh', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='tanh', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='tanh', input_shape=(1707, 26, 1), padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(250),
        tf.keras.layers.Dense(250),
        tf.keras.layers.Dense(2, activation='sigmoid'),
    ])
    return model


def main():
    loader = load.serLoader()
    X, y = loader.load_training_data()
    # X, y = loader.load_short_training_data()
    input_shape = X.shape[1:]
    print(f"input_shape: {input_shape}")
    model = short_bam()
    # for i in range(5):
    #     print(f"prev shape {X[i].shape}, predicted shape {model.predict(X[i].reshape((1, 1707, 26, 1))).shape}")
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.5, decay=0.1),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    print('start training')
    train_history = model.fit(X, y, epochs=20, batch_size=5, verbose=True)
    print('finished training')
    # Plot training loss values
    plt.plot(train_history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig('loss_per_epoch.png')
    print(f"accuracy: {train_history.history['accuracy'][-1]}")
    model.save('./model')

def test_LSTM():
    model = keras.Sequential([
        tf.keras.layers.LSTM(2, return_sequences=True),
        tf.keras.layers.LSTM(5, return_sequences=False),
        tf.keras.layers.Dense(8)
    ])
    for i in range(1):
        a = np.ones((1, 3, 3))
        print(a.shape)
        print(model.predict(a))


def calc_mean():
    loader = load.serLoader()
    # X, y = loader.load_short_training_data()
    X, y = loader.load_training_data()
    mean = 0
    for i in range(len(X)):
        mean = mean * i / (i + 1) + len(X[i]) / (i + 1)
    print(mean)


if __name__ == "__main__":
    main()
