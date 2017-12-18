#!/usr/bin/python

import sys
import pandas as pd
import pickle
import datetime 
import warnings

from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def convert_to_moment(x):
    if 6 <= x <12:
        return "Morning"
    elif 12 <= x < 17:
        return "Afternoon"
    elif 17 <= x <=20:
        return "Late"

def remove_postage(df):
	try:
		df = df[(df["StockCode"] != "DOT") & (df["StockCode"] != "POST")]
	except:
		pass 
	return df
	
def remove_cancelled(df):
	try:
		df["Cancelled"] = df["InvoiceNo"].str.startswith("C")
		df["Cancelled"] = df["Cancelled"].fillna(False)
		df = df[df["Cancelled"] == False]
		df = df.drop("Cancelled", axis=1)
	except:
		pass 
	return df
	
def rework_time(df):
	df["InvoiceDate"] = df["InvoiceDate"].astype("datetime64[ns]")
	now = datetime.date(2011, 12, 9)
	df["Recency"] = now-df["InvoiceDate"]
	df["Recency"] = pd.to_timedelta(df["Recency"]).astype("timedelta64[D]")
	df["Weekday"] = df["InvoiceDate"].dt.weekday
	df["Time"] = df["InvoiceDate"].dt.hour
	df["Time"] = df["Time"].apply(convert_to_moment)
	return df

def labelize(df):
	country_encoder = load_obj("encoder_countries")
	df["Country"] = df["Country"].apply(lambda x:country_encoder.get(x, 25)) # 25 = Unspecified
	
	obj_to_cluster = load_obj("dict_obj_cluster")
	df["Description"] = df["Description"].apply(lambda x : obj_to_cluster.get(x, 48)) # si l'article n'existe pas, on met tout dans le cluster 48
	return df
	
def add_columns(df):
	df["nb_visits"] = 1
	df["Afternoon"] = 0
	df["Late"] = 0
	df["Morning"] = 0

	for i in range(49):
		df["Cluster_{}".format(i)] = 0

	for i in range(6):
		df["Weekday_{}".format(i)] = 0
		
	df["Price"] = df["Quantity"]*df["UnitPrice"]
	df["price_avg_visits"] = 0
	df["avg_cart"] = 0
	df["freq_visit"] = 0
	return df
	
def perform_OHE(df):
	for index, row in df.iterrows():
		day = row["Weekday"]
		if day == 6:  # car on n'a pas de vente le samedi
			day = 5
		df.loc[index, "Weekday_{}".format(day)] = 1
		
	for index, row in df.iterrows():
		cluster = row["Description"]
		df.loc[index, "Cluster_{}".format(cluster)] = 1
		
	for index, row in df.iterrows():
		time = row["Time"]
		df.loc[index, time] = 1
		
	df.drop("Weekday", axis=1, inplace=True)
	return df
	
def aggregate_intermediaire(df):
	custom_aggregation = {}
	custom_aggregation["Price"] = "sum"
	custom_aggregation["CustomerID"] = lambda x:x.iloc[0]
	custom_aggregation["Country"] = lambda x:x.iloc[0]
	custom_aggregation["Quantity"] = "sum"
	custom_aggregation["Recency"] = lambda x:x.iloc[0]
	custom_aggregation["nb_visits"] = lambda x:1
	custom_aggregation["freq_visit"] = lambda x:1
	for col in df:
		if col.startswith(("Cluster", "Weekday")):
			custom_aggregation[col] = "sum"
		elif col in ["Afternoon", "Late", "Morning"]:
			custom_aggregation[col] = "mean"

	partial_df = df.groupby("InvoiceNo").agg(custom_aggregation)
	return partial_df

def aggregate_final(df):
	custom_aggregation = {}
	custom_aggregation["nb_visits"] = "count"
	custom_aggregation["Quantity"] = "mean"
	custom_aggregation["Recency"] = ["min", "max"]
	custom_aggregation["freq_visit"] = lambda x:1
	custom_aggregation["Price"] = "mean"
	custom_aggregation["Country"] = lambda x:x.iloc[0]
	for col in df:
		if col.startswith(("Cluster", "Weekday")) or col in ["Afternoon", "Late", "Morning"]:
			custom_aggregation[col] = "sum"
			
	final_df = df.groupby("CustomerID").agg(custom_aggregation)
	final_df["freq_visit"] = (final_df["Recency", "max"] - final_df["Recency", "min"])/final_df["nb_visits", "count"]
	return final_df

def rename_dataset(df):
	cols = [
		'nb_visits', 
		'Quantity', 
		'Recency_min', 
		'Recency_max', 
		'freq_visit', 
		'Avg_Price', 
		'Country', 
		'Afternoon', 
		'Late', 
		'Morning'
	]

	for i in range(49):
		cols.append("Cluster_{}".format(i))
	for i in range(6):
		cols.append("Weekday_{}".format(i))
		
	df.columns = cols
	return df
	
def make_prediction(file):
	df = pd.read_csv(file, index_col=0)
	
	# preparation du dataset
	df = remove_postage(df)
	df = remove_cancelled(df)
	df = rework_time(df)
	df = labelize(df)
	df = add_columns(df)
	
	df = df.reset_index(drop = True)
	df = perform_OHE(df)
	
	# aggregation pré-classification
	intermediaire_df = aggregate_intermediaire(df)
	final_df = aggregate_final(intermediaire_df)
	final_df = rename_dataset(final_df)
	
	# Scaling et Prediction
	scaler = joblib.load("scaler.pkl")
	model = joblib.load("final_model.pkl")
	X = scaler.transform(final_df)
	return model.predict(X)

	
if __name__ == "__main__":
	files = sys.argv[1:]
	for file in files:
		prediction = make_prediction(file)
		print("Prediction sur", file)
		print("Ce client est predit pour appartenir au Groupe {}".format(prediction[0]), "\n")
