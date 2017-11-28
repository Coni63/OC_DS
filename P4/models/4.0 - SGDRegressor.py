import numpy as np
import datetime

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid

from sklearn.externals import joblib

loaded = np.load('data/matrices.npz')

X_train = loaded["X_train"]
y_train = loaded["y_train"]
X_test = loaded["X_test"]
y_test = loaded["y_test"]

# params = [
#     {
#     "loss" : ["huber"],
#     "penalty" : [None, 'l2', 'l1'],
#     "max_iter": [3, 5, 10]
#     },
#     {
#     "loss" : ["huber"],
#     "penalty" : ['elasticnet'],
#     "l1_ratio" : [0.15, 0.50, 0.85],
#     "max_iter": [3, 5, 10]
#     }]
#
# for param in ParameterGrid(params):
#     print(param)
#     model = SGDRegressor(**param)
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#
#     MSE = mean_squared_error(y_test, y_pred)
#     MAE = mean_absolute_error(y_test, y_pred)
#
#     with open("log_result.txt", "a") as f:
#         current_date = datetime.datetime.now().strftime("%d/%m/%y - %H:%M:%S")
#         f.write("{} => SGD_Regressor ({}) : MSE {:.4f}, MAE {:.4f}\n".format(current_date, param, MSE, MAE) )


# Sauvegarde du model final

# best_params_single_SGDR = {'loss': 'huber', 'max_iter': 10, 'penalty': None}
# model = SGDRegressor(**best_params_single_SGDR)
# model.fit(X_train, y_train)
# joblib.dump(model, '../prod/model.pkl')

model = joblib.load("../prod/model.pkl")

y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
print(MSE, MAE)

print(y_pred)