import pandas as pd

df = pd.read_csv("../dataset/2017/2017_01_light.csv")
for i in range(2, 7):
    print("Load Month:", i)
    df2 = pd.read_csv("../dataset/2017/2017_{:02d}_light.csv".format(i))
    df = df.append(df2)

df.dropna(axis=1, how="all", inplace=True)
df.drop(["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "UNIQUE_CARRIER", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "ARR_TIME", "ARR_DELAY", "CANCELLED", "DIVERTED", "DISTANCE", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"], axis=1, inplace=True)

df.to_csv("../dataset/arima/arima_test_dataset.csv", index=False)