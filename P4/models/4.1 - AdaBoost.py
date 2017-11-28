import numpy as np
import datetime

from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.externals import joblib

loaded = np.load('data/matrices.npz')

X_train = loaded["X_train"]
y_train = loaded["y_train"]
X_test = loaded["X_test"]
y_test = loaded["y_test"]

best_params_single_SGDR = {'loss': 'huber', 'max_iter': 10, 'penalty': None}

# Grid Search

# params = [
# {
#     "base_estimator" : [SGDRegressor(**best_params_single_SGDR)],
#     "n_estimators" : [2, 5, 10, 20],
#     "loss" : ['linear', 'square', 'exponential']
# }]
#
# for param in ParameterGrid(params):
#     print(param)
#     booster = AdaBoostRegressor(**param)
#
#     booster.fit(X_train, y_train)
#
#     y_pred = booster.predict(X_test)
#     MSE = mean_squared_error(y_test, y_pred)
#     MAE = mean_absolute_error(y_test, y_pred)
#
#     with open("log_result.txt", "a") as f:
#         current_date = datetime.datetime.now().strftime("%d/%m/%y - %H:%M:%S")
#         f.write("{} => AdaBoost ({}) : MSE {:.4f}, MAE {:.4f} \n".format(current_date, param , MSE, MAE))

# Sauvegarde du model final

# booster = AdaBoostRegressor(SGDRegressor(**best_params_single_SGDR), n_estimators=2, loss="exponential")
# booster.fit(X_train, y_train)
# joblib.dump(booster, '../prod/model.pkl')

booster = joblib.load("../prod/model.pkl")

y_pred = booster.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
print(MSE, MAE)

print(y_pred)