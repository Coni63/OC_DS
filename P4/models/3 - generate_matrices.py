import pandas as pd
import numpy as np
import scipy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

week_format = True
OHE_aiport = True

df = pd.read_csv("../dataset/merged/prod_dataset.csv")
df.dropna(axis=1, how="all", inplace=True)

if week_format:
    df = pd.concat([df, pd.get_dummies(df["UNIQUE_CARRIER"])], axis=1)
    df = pd.concat([df, pd.get_dummies(df["DAY_OF_WEEK"], prefix="DAY_")], axis=1)
    df = pd.concat([df, pd.get_dummies(df["NUM_WEEK"], prefix="WEEK_")], axis=1)

    df.drop(["UNIQUE_CARRIER"], axis=1, inplace = True)
    df.drop("DAY_OF_WEEK", axis=1, inplace = True)
    df.drop("NUM_WEEK", axis=1, inplace = True)
else:
    df = pd.concat([df, pd.get_dummies(df["UNIQUE_CARRIER"])], axis=1)
    df = pd.concat([df, pd.get_dummies(df["MONTH"], prefix="MONTH_")], axis=1)
    df = pd.concat([df, pd.get_dummies(df["DAY_OF_MONTH"], prefix="DAYM_")], axis=1)
    df = pd.concat([df, pd.get_dummies(df["DAY_OF_WEEK"], prefix="DAY_")], axis=1)

    df.drop(["UNIQUE_CARRIER"], axis=1, inplace=True)
    df.drop("MONTH", axis=1, inplace=True)
    df.drop("DAY_OF_MONTH", axis=1, inplace=True)
    df.drop("DAY_OF_WEEK", axis=1, inplace=True)

if OHE_aiport:
    df = pd.concat([df, pd.get_dummies(df["ORIGIN_AIRPORT_ID"], prefix="ORIGIN_RANK_")], axis=1)
    df = pd.concat([df, pd.get_dummies(df["DEST_AIRPORT_ID"], prefix="DEST_RANK_")], axis=1)
    df.drop(["ORIGIN_AIRPORT_ID"], axis=1, inplace=True)
    df.drop("DEST_AIRPORT_ID", axis=1, inplace=True)

y = df["DEP_DELAY"]
X = df.drop("DEP_DELAY", axis=1)

scaler = MinMaxScaler()

if OHE_aiport:
    X[["SHIFT"]] = scaler.fit_transform(X[["SHIFT"]])
else:
    X[['ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID', "SHIFT"]] = scaler.fit_transform(X[['ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID', "SHIFT"]])

# X_scale = scaler.fit_transform(X.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# np.save("data/X_train", X_train.as_matrix())
# np.save("data/y_train", y_train.as_matrix())
# np.save("data/X_test", X_test.as_matrix())
# np.save("data/y_test", y_test.as_matrix())

np.savez_compressed('data/matrices', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# scipy.sparse.save_npz('/data/X_train.npz', X_train)
# scipy.sparse.save_npz('/data/X_train.npz', X_train)
# scipy.sparse.save_npz('/data/X_train.npz', X_train)
# scipy.sparse.save_npz('/data/X_train.npz', X_train)

