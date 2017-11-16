import numpy as np
import datetime

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid

loaded = np.load('data/matrices.npz')

X_train = loaded["X_train"]
y_train = loaded["y_train"]
X_test = loaded["X_test"]
y_test = loaded["y_test"]

params = [
    {
    "loss" : ["huber"],
    "penalty" : [None, 'l2', 'l1'],
    "max_iter": [3, 5, 10]
    },
    {
    "loss" : ["huber"],
    "penalty" : ['elasticnet'],
    "l1_ratio" : [0.15, 0.50, 0.85],
    "max_iter": [3, 5, 10]
    }]

for param in ParameterGrid(params):
    print(param)
    model = SGDRegressor(**param)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)

    with open("log_result.txt", "a") as f:
        current_date = datetime.datetime.now().strftime("%d/%m/%y - %H:%M:%S")
        f.write("{} => SGD_Regressor ({}) : MSE {:.4f}, MAE {:.4f}\n".format(current_date, param, MSE, MAE) )

    # a = 10
    # s = 100
    # pred = model.predict(X_test[s:s+a])
    # for x, y in zip(y_pred, y_test[s:s+a]):
    #     print(x, " => ", y)