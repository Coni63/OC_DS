import numpy as np
import pandas as pd
import datetime
import pickle


def save_obj(obj, name):
    with open('../prod/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def Score(x, d):
    if x <= d[0.25]:
        return 0.25
    elif x <= d[0.50]:
        return 0.5
    elif x <= d[0.75]:
        return 0.75
    else:
        return 1

def hhmm_to_min(x):
    if not pd.isnull(x):
        x = int(x)
        if x <= 59:
            return x
        else:
            x = str(x)
            h, m = x[:-2], x[-2:]
            return int(h)*60 + int(m)

def to_week_num(x):
    return ((x - datetime.datetime(2016,1,1)).days // 7) + 1


def generate_dataset(week_format = False):
    df = pd.read_csv("../dataset/lighted/2016_01_light.csv")
    for i in range(2, 13):
        print("Load Month:", i)
        # if i == 4:
        #     df2 = pd.read_csv("dataset/original/2016_{:02d}.csv".format(i), error_bad_lines=False)
        #     df2 = df2[df2["MONTH"] == 4]
        # else:
        #     df2 = pd.read_csv("dataset/original/2016_{:02d}.csv".format(i))
        df2 = pd.read_csv("../dataset/lighted/2016_{:02d}_light.csv".format(i))
        df = df.append(df2)

    # Convert Column
    print("Converting Datas")
    df["CRS_DEP_TIME"] = df["CRS_DEP_TIME"].apply(lambda x:hhmm_to_min(x))
    df["CRS_ARR_TIME"] = df["CRS_ARR_TIME"].apply(lambda x:hhmm_to_min(x))

    to_convert = ["DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "DEP_DELAY", "CRS_ARR_TIME", "ARR_DELAY"]
    for col in to_convert:
        df[col] = df[col].fillna(0).astype("int")

    # Handle Unpredictable events
    print("Cleanup non-predictible events")
    non_predictible_reasons = ["CANCELLED", "DIVERTED"]
    for reason in non_predictible_reasons:
        df = df[df[reason] == 0]

    semi_predictible_reasons = ["WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]  #
    for reason in semi_predictible_reasons:
        df = df[(df[reason]<60) | (df[reason].isnull())]

    df.drop(["DIVERTED", "CANCELLED", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY", "CRS_ARR_TIME"],
            axis=1, inplace=True)

    print("Converting Datas")
    df["SHIFT"] = (df["CRS_DEP_TIME"] - 15).abs()

    print("Converting Dates")
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format='%Y-%m-%d')
    df["NUM_WEEK"] = df["FL_DATE"].apply(lambda x: to_week_num(x))
    df.drop(["FL_DATE"], axis=1, inplace=True)

    print("Saving Exploration dataset")
    df.to_csv("../dataset/merged/explo_dataset.csv", index=False)

    print("Preparing Production dataset")
    if week_format:
        df.drop(["CRS_DEP_TIME", "DISTANCE", "ARR_DELAY", "MONTH", "DAY_OF_MONTH"], axis=1, inplace=True)
    else:
        df.drop(["CRS_DEP_TIME", "DISTANCE", "ARR_DELAY", "NUM_WEEK"], axis=1, inplace=True)

    print("Convert Airports")
    df2 = df.groupby("ORIGIN_AIRPORT_ID")["ORIGIN_AIRPORT_ID", "DEP_DELAY"].mean()
    quantiles = df2["DEP_DELAY"].quantile(q=[0.25, 0.5, 0.75])
    quantiles = quantiles.to_dict()

    df2["RANK"] = df2["DEP_DELAY"].apply(Score, args=(quantiles,))
    converter = df2.set_index('ORIGIN_AIRPORT_ID')["RANK"].to_dict()
    df['ORIGIN_AIRPORT_RANK'] = df['ORIGIN_AIRPORT_ID'].apply(lambda x: converter[x])
    df['DEST_AIRPORT_RANK'] = df['DEST_AIRPORT_ID'].apply(lambda x: converter[x])

    save_obj(converter, "airport_converter")

    print("Saving Production dataset")
    df.to_csv("../dataset/merged/prod_dataset.csv", index=False)

    print("Done")

if __name__ == "__main__":
    generate_dataset(week_format=True)