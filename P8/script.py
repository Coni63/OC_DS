import time
import pickle
import numpy as np
np.random.seed(42)

from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, SpatialDropout1D
from keras.layers import LSTM, SimpleRNN, GRU
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.datasets import imdb

from qrnn.qrnn import QRNN

def regenerate_dataset(timesteps=256):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

    X_train = sequence.pad_sequences(X_train, maxlen=timesteps,  padding=padding_mode, truncating=truncating_mode)
    X_test = sequence.pad_sequences(X_test, maxlen=timesteps,  padding=padding_mode, truncating=truncating_mode)
    X_train, y_train, X_test, y_test = X_train[::2], y_train[::2], X_test[::2], y_test[::2]
    return X_train, y_train, X_test, y_test

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = 0
        
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times = time.time() - self.epoch_time_start
        
time_callback = TimeHistory()

def time_lstm(batch_size, train_set, test_set):
    model_lstm = Sequential()
    model_lstm.add(Embedding(max_features, 128))
    model_lstm.add(LSTM(128))
    model_lstm.add(Dense(1))
    model_lstm.add(Activation('sigmoid'))

    model_lstm.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model_lstm.fit(*train_set, batch_size=batch_size, epochs=1,
                         validation_data=train_set,
                         verbose=0, callbacks=[time_callback])
    
    return time_callback.times

def time_qrnn(batch_size, train_set, test_set):
    model_qrnn = Sequential()
    model_qrnn.add(Embedding(max_features, 128))
    model_qrnn.add(QRNN(128, window_size=5))
    model_qrnn.add(Dense(1))
    model_qrnn.add(Activation('sigmoid'))

    model_qrnn.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model_qrnn.fit(*train_set, batch_size=batch_size, epochs=1,
                         validation_data=train_set,
                         verbose=0, callbacks=[time_callback])
    
    return time_callback.times
    
max_features = 20000
maxlen = 256
batch_size = 32
epochs = 1
padding_mode = "pre"
truncating_mode = "pre"

matrice_results = []
for sequence_length in [512]:  #32, 64, 128, 256, 
    print("Sequence :", sequence_length)
    X_train, y_train, X_test, y_test = regenerate_dataset(sequence_length)
    batch_matrice = []
    for batch_size in [256]: #8, 16, 32, 64, 128, 
        print("Batch :", batch_size)
        t1 = time_qrnn(batch_size=batch_size, train_set=(X_train, y_train), test_set=(X_test, y_test))
        t2 = time_lstm(batch_size=batch_size, train_set=(X_train, y_train), test_set=(X_test, y_test))
        batch_matrice.append(t2/t1)
        print(t2/t1)
    matrice_results.append(batch_matrice)

with open('perf', 'wb') as file:
    pickle.dump(matrice_results, file)