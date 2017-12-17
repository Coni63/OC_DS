import numpy as np
np.random.seed(42)
import time
import pickle

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

max_features = 20000
maxlen = 256
batch_size = 32
epochs = 25
padding_mode = "pre"
truncating_mode = "post"

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen,  padding=padding_mode, truncating=truncating_mode)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen,  padding=padding_mode, truncating=truncating_mode)


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

early = EarlyStopping(
    monitor='loss',
    min_delta=0,
    patience=3,
    verbose=1,
    mode='auto')

#### SRNN
print("training SRNN")
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(SimpleRNN(128))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history1 = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=(X_test, y_test),
                     verbose=0, callbacks=[early, time_callback])

history1.history["time"] = time_callback.times
with open('srnn', 'wb') as file:
    pickle.dump(history1.history, file)

#### LSTM
print("training LSTM")
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history2 = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=(X_test, y_test),
                     verbose=0, callbacks=[early, time_callback])

history2.history["time"] = time_callback.times
with open('lstm', 'wb') as file:
    pickle.dump(history2.history, file)

#### GRU
print("training GRU")
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(GRU(128))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history3 = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=(X_test, y_test),
                     verbose=0, callbacks=[early, time_callback])

history3.history["time"] = time_callback.times
with open('gru', 'wb') as file:
    pickle.dump(history3.history, file)

#### QRNN
print("training QRNN")
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(QRNN(128, window_size=3, dropout=0.2,
               kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4),
               kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history4 = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=(X_test, y_test),
                     verbose=0, callbacks=[early, time_callback])

history4.history["time"] = time_callback.times
with open('qrnn', 'wb') as file:
    pickle.dump(history4.history, file)
