#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import keras
import keras.backend as K
#
# !pip install py_vollib
#
from sklearn.model_selection import ParameterGrid
from py_vollib import black_scholes_merton as bsm
from progressbar import ProgressBar
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import uniform
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split


# testing neural network (full data)
fullDF = pd.read_csv("dataFull.csv")


def baseline_model():
    # create model
    i = Input(shape=(6,))
    x = Dense(10, activation='relu')(i)
    y = Dense(10, activation='relu')(x)
    o = Dense(1)(y)
    model = Model(i, o)
    model.compile(loss="mse", optimizer="adam")
    return model


model_full = baseline_model()
X = fullDF[['S', 'K', 'q', 'r', 'sigma', 't']]
y = fullDF[['price']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=7)
history_full = model_full.fit(X_train, y_train, batch_size=64, epochs=20,
                              verbose=2, validation_split=0.2)  # set batch size to 1, otherwise there are errors when trying to add the custom_error above

plt.plot(history_full.history['val_loss'])
plt.title('Model validation loss')
plt.ylabel('Validation Loss')
plt.xlabel('Epoch')
plt.legend(['Error', 'Test'], loc='upper left')
plt.show()
X_test_full = X_test
y_test_full = y_test
model_full.evaluate(x=X_test, y=y_test)

model_full.save("model_full.h5")

# testing neural network (sparse data)
sparseDF = pd.read_csv("dataSparse.csv")


def baseline_model():
    # create model
    i = Input(shape=(6,))
    x = Dense(10, activation='relu')(i)
    y = Dense(10, activation='relu')(x)
    o = Dense(1)(y)
    model = Model(i, o)
    model.compile(loss="mse", optimizer="adam")
    return model


model_sparse = baseline_model()
X = sparseDF[['S', 'K', 'q', 'r', 'sigma', 't']]
y = sparseDF[['price']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=7)
history_sparse = model_sparse.fit(X_train, y_train, batch_size=64, epochs=20,
                                  verbose=2, validation_split=0.2)  # set batch size to 1, otherwise there are errors when trying to add the custom_error above
plt.plot(history_sparse.history['val_loss'])
plt.title('Model validation loss')
plt.ylabel('Validation Loss')

plt.xlabel('Epoch')
plt.legend(['Error', 'Test'], loc='upper left')
plt.show()
model_sparse.evaluate(x=X_test_full, y=y_test_full)

model_sparse.save("model_sparse.h5")


# testing neural network (extremes data)
extremesDF = pd.read_csv("dataExtremes.csv")


def baseline_model():
    # create model
    i = Input(shape=(6,))
    x = Dense(10, activation='relu')(i)
    y = Dense(10, activation='relu')(x)
    o = Dense(1)(y)
    model = Model(i, o)
    model.compile(loss="mse", optimizer="adam")
    return model


model_extremes = baseline_model()
X = extremesDF[['S', 'K', 'q', 'r', 'sigma', 't']]
y = extremesDF[['price']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=7)
history_extremes = model_extremes.fit(X_train, y_train, batch_size=64, epochs=20,
                                      verbose=2, validation_split=0.2)  # set batch size to 1, otherwise there are errors when trying to add the custom_error above
plt.plot(history_extremes.history['val_loss'])
plt.title('Model validation loss')
plt.ylabel('Validation Loss')
plt.xlabel('Epoch')
plt.legend(['Error', 'Test'], loc='upper left')
plt.show()
model_extremes.evaluate(x=X_test_full, y=y_test_full)

model_extremes.save("model_extremes.h5")


tableOutput = pd.DataFrame({'Full': history_full.history['val_loss'],
                            'Sparse': history_sparse.history['val_loss'],
                            'Extremes': history_extremes.history['val_loss']}, columns=['Full', 'Sparse', 'Extremes'])
tableOutput.to_csv("tableResultsValidaton.csv")
print(len(fullDF.index))
print(len(sparseDF.index))
print(len(extremesDF.index))
