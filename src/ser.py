__version__ = "0.1.0"

import tensorflow as tf
# import tensorflow_cloud as tfc
import load
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import json


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


def model6_bam(four_class):
    model = keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 26), activation='tanh', input_shape=(1707, 26, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(250, activation='tanh'),
        tf.keras.layers.Dense(250, activation='tanh'),
        tf.keras.layers.Dense(100, activation='tanh'),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(25, activation='tanh'),
    ])
    if four_class:
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    return model, 'padded'


def model8_bam(four_class):
    model = keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3, 26), activation='tanh', input_shape=(1707, 26, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='tanh'),
        tf.keras.layers.Dense(25, activation='tanh'),
    ])
    if four_class:
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    return model, 'padded'


def milli_bam(four_class):
    model = keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 26), strides=(1, 1), activation='tanh', input_shape=(1707, 26, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(4, 1), strides=(4, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='tanh'),
        tf.keras.layers.Dense(100, activation='tanh'),
        tf.keras.layers.Dense(100, activation='tanh'),
        tf.keras.layers.Dense(100, activation='tanh'),
    ])
    if four_class:
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    return model, 'padded'


def lstm(four_class):
    model = keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(None, 26), ragged=True),
        # tf.keras.layers.LSTM(250, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(250, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(100, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(100, activation='tanh', return_sequences=False),
        tf.keras.layers.Dense(250, activation='tanh'),
        tf.keras.layers.Dense(250, activation='tanh'),
        tf.keras.layers.Dense(50, activation='tanh'),
    ])
    if four_class:
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    return model, 'ragged'


def small_lstm(four_class):
    model = keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(None, 26), ragged=True),
        # tf.keras.layers.LSTM(250, activation='tanh', return_sequences=True),
        # tf.keras.layers.LSTM(26, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(26, activation='relu', return_sequences=False),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
    ])
    if four_class:
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    return model, 'ragged'



def main():
    print('get model')
    four_class = True
    model, mode = milli_bam(four_class)
    print(model.summary())
    loader = load.serLoader()
    print('load data')
    # X, y = loader.load_short_training_data(mode=mode)
    X, y = loader.load_training_data(mode=mode, four_class=four_class)
    if four_class:
        # loss = keras.losses.CategoricalCrossentropy()
        print('using categorical crossentropy')
        loss = 'categorical_crossentropy'
    else:
        loss = keras.losses.MeanSquaredError()
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.2, decay=0.01),
                  loss=loss,
                  metrics=['accuracy'])
    print('start training')
    train_history = model.fit(X, y, epochs=1500, batch_size=8, verbose=True)
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


def print_metrics(y, y_):
    y = np.array(y)
    y_ = np.array(y_)
    print(classification_report(y, y_))
    print(confusion_matrix(y, y_))


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


def save_predictions(y, json_path):
    j = dict()
    if len(y[0]) == 4:
        for i in range(len(y)):
            j[str(i)] = {'valence': 0, 'activation': 0}
            if y[i][1] == 1:
                j[str(i)]['valence'] = 1
            elif y[i][2] == 1:
                j[str(i)]['activation'] = 1
            elif y[i][3] == 1:
                j[str(i)]['valence'] = 1
                j[str(i)]['activation'] = 1
    with open(json_path, 'w') as f:
        json.dump(j, f)


def predict(model_path='./model/'):
    loader = load.serLoader()
    # X, y = loader.load_training_data(four_class=True, mode='padded')
    X = loader.load_dev_data(mode='padded')
    y_ = make_predictions(model_path, X)
    # print_metrics(y, y_)
    save_predictions(y_, './predictions.json')

def training_metrics(model_path='./model/'):
    loader = load.serLoader()
    X, y = loader.load_training_data(four_class=True, mode='padded')
    y_ = make_predictions(model_path, X)
    print_metrics(y, y_)


if __name__ == '__main__':
    # main()
    model_path = '../some_models/CNN_model7'
    training_metrics()
    predict()

