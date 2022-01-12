__version__ = "0.1.0"

import tensorflow as tf
# import tensorflow_cloud as tfc
import load
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


def bam(four_class):
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
    ])
    if four_class:
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    return model, 'padded'


def short_bam(four_class):
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
    ])
    if four_class:
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    return model, 'padded'


def lstm(four_class):
    model = keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(None, 26), ragged=True),
        tf.keras.layers.LSTM(250, activation='tanh', return_sequences=True),
        #tf.keras.layers.LSTM(250, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(100, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(100, activation='tanh', return_sequences=False),
        tf.keras.layers.Dense(100, activation='tanh'),
        tf.keras.layers.Dense(50, activation='tanh'),
    ])
    if four_class:
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    return model, 'ragged'


def main():
    print('get model')
    four_class = True
    model, mode = lstm(four_class)
    print(model.summary())
    loader = load.serLoader()
    print('load data')
    # X, y = loader.load_short_training_data(mode=mode)
    X, y = loader.load_training_data(mode=mode, four_class=four_class)
    if four_class:
        loss = keras.losses.CategoricalCrossentropy()
    else:
        loss = keras.losses.MeanSquaredError()
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.2, decay=0.01),
                  loss=loss,
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


def argmax_vector(vec):
    result = np.zeros(vec.shape)
    result[vec.argmax()] = 1
    return result


def make_predictions(model_path, X):
    model = keras.models.load_model(model_path)
    y_ = model.predict(X)
    y_ = [argmax_vector(el) for el in y_]
    return np.array(y_)



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
    #main()
    loader = load.serLoader()
    model_path = '../some_models/LSTM_model1/'
    X, y = loader.load_short_training_data(mode='ragged')
    y_ = make_predictions(model_path, X)
    print(y)
    print(y_)
