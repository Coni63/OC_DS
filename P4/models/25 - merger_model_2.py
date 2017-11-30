import numpy as np
import pandas as pd
import datetime
import pickle

def load_obj(name):
    with open('../prod/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name):
    with open('../prod/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def to_hour(x):
    if not pd.isnull(x):
        x = int(x)
        if x < 60:
            return 0
        else:
            return int(str(x)[:-2])

def Score(x, d):
    if x <= d[0.25]:
        return 0.25
    elif x <= d[0.50]:
        return 0.5
    elif x <= d[0.75]:
        return 0.75
    else:
        return 1


print("Load Month: 1")
df = pd.read_csv("../dataset/lighted/2016_01_light.csv")
for i in range(2, 13):
    print("Load Month:", i)
    df2 = pd.read_csv("../dataset/lighted/2016_{:02d}_light.csv".format(i))
    df = df.append(df2)

print("Selecting features")
df["FL_HOUR"] = df["CRS_DEP_TIME"].apply(to_hour)
df['WEEK'] = pd.to_datetime(df['FL_DATE']).dt.week
df["NUM_FLIGHT"] = 0

df = df[["WEEK", "DAY_OF_WEEK", "UNIQUE_CARRIER", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "DEP_DELAY", "FL_HOUR", "NUM_FLIGHT"]]

print("Converting Airport")
df2 = df.groupby("ORIGIN_AIRPORT_ID")["ORIGIN_AIRPORT_ID", "DEP_DELAY"].mean()
quantiles = df2["DEP_DELAY"].quantile(q=[0.25, 0.5, 0.75])
quantiles = quantiles.to_dict()

df2["RANK"] = df2["DEP_DELAY"].apply(Score, args=(quantiles,))
converter = df2.set_index('ORIGIN_AIRPORT_ID')["RANK"].to_dict()
df['ORIGIN_AIRPORT_RANK'] = df['ORIGIN_AIRPORT_ID'].apply(lambda x: converter[x])

df.drop("ORIGIN_AIRPORT_ID", axis=1, inplace=True)

print("Aggregating")
df = df.groupby(("WEEK", "DAY_OF_WEEK", "FL_HOUR", "UNIQUE_CARRIER", "ORIGIN_AIRPORT_RANK")).agg({
    "DEP_DELAY" : "mean",
    "NUM_FLIGHT" : "count"
}).reset_index()

print("Saving ...")
df.to_csv("../dataset/merged/model2_dataset.csv", index=False)

print("Done")