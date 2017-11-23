import pandas as pd

for i in range(1, 13):
    print("Month", i)
    df = pd.read_csv("../dataset/original/2016_{:02d}.csv".format(i), error_bad_lines=False)

    print("Filtering")
    df = df[df["MONTH"] == i]

    print("Removing rows")
    for col in df:
        if col.startswith(("ORIGIN", "DEST")) and not col.endswith("AIRPORT_ID"):
            df.drop([col], axis=1, inplace=True)

    to_delete = ["QUARTER", "YEAR", "CARRIER", "AIRLINE_ID", "DEP_TIME", "DEP_DELAY_NEW", "DEP_DEL15",
                 "DEP_DELAY_GROUP", "DEP_TIME_BLK", "ARR_TIME", "ARR_DELAY_NEW", "ARR_DEL15", "ARR_DELAY_GROUP",
                 "ARR_TIME_BLK", "AIR_TIME", "FLIGHTS", "DISTANCE_GROUP", "FIRST_DEP_TIME", "TOTAL_ADD_GTIME",
                 "LONGEST_ADD_GTIME", "TAIL_NUM", "FL_NUM", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
                 "ACTUAL_ELAPSED_TIME", "CRS_ELAPSED_TIME", "CANCELLATION_CODE", "CARRIER_DELAY"]
    df.drop(to_delete, axis=1, inplace=True)

    print("save")
    df.to_csv("../dataset/lighted/2016_{:02d}_light.csv".format(i), index=False)
print("DONE")