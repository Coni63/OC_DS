import numpy as np
import seaborn as sns
import time

import create_dataset as reber

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import LSTM, SimpleRNN, GRU
from keras.preprocessing import sequence

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def generate(x0):
    end = np.array([0.,  0.,  0.,  0.,  0.,  0.,  1.])
    y=x0[1:]
    y.append(end)
    return y

def Pick_From_Output(x):
    y = np.zeros_like(x)
    x = np.where(x < 0.1, 0, x)
    x = x[0]/x[0].sum(axis=1)
    i = np.random.choice(list(range(7)), size=1, p=x[0])
    y[0,0,i] = 1
    return y

def evaluate(model, nb_word = 1, max_iter = 50):
    good_pred = 0
    for _ in range(nb_word):
        model.reset_states()
        first_input = np.array([[[1,0,0,0,0,0,0]]])
        word = "B"
        loop = 0
        nextLetter = "B"
        next_seq = first_input
        while nextLetter != "E" and loop < max_iter:
            y_pred = model.predict(next_seq)
            next_seq = Pick_From_Output(y_pred)
            nextLetter = reber.sequenceToWord(next_seq[0])
            loop += 1
            word += nextLetter
        if reber.in_grammar(word):
            good_pred += 1
    acc = 100*good_pred/nb_word
    print("Good prediction : {:.2f}%".format(acc))
    return acc


min_length = 10
X_train, y_train = [], []
X_test, y_test = [], []
X_val, y_val = [], []
y_possible = []

for i in range(2048):
    x, y = reber.get_one_example(min_length)
    X_train.append(x)
    y_train.append(generate(x))

for i in range(256):
    x, y = reber.get_one_example(min_length)
    X_test.append(x)
    y_test.append(generate(x))

for i in range(1):
    x, y = reber.get_one_example(min_length)
    X_val.append(x)
    y_val.append(generate(x))
    y_possible.append(y)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_val = np.array(X_val)
y_val = np.array(y_val)
y_possible = np.array(y_possible)

maxlen = 20
X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post', truncating='post')
y_train = sequence.pad_sequences(y_train, maxlen=maxlen, padding='post', truncating='post')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post', truncating='post')
y_test = sequence.pad_sequences(y_test, maxlen=maxlen, padding='post', truncating='post')
X_val = sequence.pad_sequences(X_val, maxlen=maxlen, padding='post', truncating='post')
y_val = sequence.pad_sequences(y_val, maxlen=maxlen, padding='post', truncating='post')
y_possible = sequence.pad_sequences(y_possible, maxlen=maxlen, padding='post', truncating='post')

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# print(X_val.shape)
# print(y_val.shape)
# print(y_possible.shape)

nb_unit = 7
inp_shape = (maxlen, 7)
loss_ = "mean_squared_error"
metrics_ = "mean_squared_error"
optimizer_ = "Nadam"
nb_epoch = 50
batch_size = 64

# model = Sequential()
#
# model.add(LSTM(units=nb_unit, input_shape=inp_shape, return_sequences=True))  # single LSTM
# model.compile(loss=loss_,
#               optimizer=optimizer_,
#               metrics=[metrics_])



checkpoint = ModelCheckpoint("test.h5",
    monitor=loss_,
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1)
early = EarlyStopping(
    monitor='loss',
    min_delta=0,
    patience=10,
    verbose=1,
    mode='auto')

model = load_model("test.h5")
#
# start = time.time()
# history = model.fit(X_train, y_train,
#                     validation_data=(X_test, y_test),
#                     epochs=nb_epoch,
#                     batch_size=batch_size,
#                     verbose=1,
#                     callbacks = [checkpoint, early])
# stop = time.time()
# t1 = stop-start
# print(model.summary())
# print("Training time : {}s".format(t1))



newModel = Sequential()
newModel.add(LSTM(units=nb_unit, stateful=True, batch_input_shape=(1,1,7), return_sequences=True))
newModel.set_weights(model.get_weights())

# result_LSTM = []
# for _ in range(5):
#     result_LSTM.append(evaluate(newModel, 100, 50))

evaluate(newModel, 100, 50)