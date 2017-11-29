import numpy as np
import pandas as pd
import datetime
import pickle

def to_hour(x):
    if not pd.isnull(x):
        x = int(x)
        if x < 60:
            return 0
        else:
            return int(str(x)[:-2])

print("Load Month: 1")
df = pd.read_csv("../dataset/lighted/2016_01_light.csv")
for i in range(2, 13):
    print("Load Month:", i)
    df2 = pd.read_csv("../dataset/lighted/2016_{:02d}_light.csv".format(i))
    df = df.append(df2)

print("Selecting features")
df["FL_HOUR"] = df["CRS_DEP_TIME"].apply(to_hour)
df = df[["FL_DATE", "DAY_OF_WEEK", "UNIQUE_CARRIER", "ORIGIN_AIRPORT_ID", "DEP_DELAY", "FL_HOUR"]]

print("Saving ...")
df.to_csv("../dataset/merged/model3_dataset.csv", index=False)

print("Done")