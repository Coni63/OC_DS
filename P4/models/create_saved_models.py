import pickle
import pandas as pd

def save_obj(obj, name ):
    with open('../prod/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

df = pd.read_csv("../dataset/merged/prod_dataset.csv")
df.dropna(axis=1, how="all", inplace=True)
df.drop("DEP_DELAY", axis=1, inplace=True)

# Liste des aéroports
airports = pd.read_csv("../L_AIRPORT_ID.csv")
depart = df["DEST_AIRPORT_ID"].unique()
arrive = df["ORIGIN_AIRPORT_ID"].unique()
union_aeroport = list(set(depart) | set(arrive))
aiport_dict = {key : airports[airports["Code"] == key]["Description"].values[0] for key in union_aeroport}
save_obj(aiport_dict, "airport")

# Liste des compagnies
companies = pd.read_csv("../L_UNIQUE_CARRIERS.csv")
carrier_list = df["UNIQUE_CARRIER"].unique()
comp_dict = {key : companies[companies["Code"] == key]["Description"].values[0] for key in carrier_list}
save_obj(comp_dict, "company")



# .values[0].split(":")[-1].strip()