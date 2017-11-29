import numpy as np
import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error

loaded = np.load('data/matrices.npz')

X_train = loaded["X_train"]
y_train = loaded["y_train"]
X_test = loaded["X_test"]
y_test = loaded["y_test"]

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(250, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error', 'mean_squared_error'])

early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')

model.fit(X_train, y_train, epochs=10, batch_size=2000, callbacks=[early_stop], verbose=2)

y_pred = model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)

with open("log_result.txt", "a") as f:
    current_date = datetime.datetime.now().strftime("%d/%m/%y - %H:%M:%S")
    f.write("{} => ANN ({} Epoch) : MSE {:.4f}, MAE {:.4f} \n".format(current_date, 10 , MSE, MAE))
