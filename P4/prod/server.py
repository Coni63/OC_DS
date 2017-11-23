from flask import Flask
from flask import render_template
from flask import request
import datetime
import numpy as np
import pandas as pd
from operator import itemgetter
import pickle
from sklearn.externals import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

@app.route('/predict/', methods=['POST'])
def hello():
	if request.method == 'POST':
		print(request.values)
		index_arr = load_obj("index")

		a = [0 for _ in range(len(index_arr))]
		for i, j in request.values.items():
			if i == "departure":
				a[3] = j
			elif i == "arrival":
				a[4] = j
			elif i == "date":
				yyyy, mm, dd = j.split("-")
				a[0] = mm
				a[1] = dd
				a[2] = datetime.datetime.strptime(j, "%Y-%m-%d").isoweekday()
			elif i == "time":
				hh, mm = j.split(":")
				a[5] = int(hh)*60+int(mm)
			elif i =="company":
				a[index_arr.index(j)] = 1
			else:
				print(i, j)
		a = [a]
		X = np.array(a)
		print(X)
		scaler = joblib.load("scaler.pkl")
		X_scaled = scaler.transform(X)
		print(X_scaled)
		model = joblib.load("model.pkl")
		lateness = model.predict(X_scaled)[0]
		if lateness > 0:
			result = "<SMALL>Late :</SMALL>"
		else:
			result = "<SMALL>Early :</SMALL>"
		min, sec = int(lateness), int((lateness%1)*60)
		return_str = result + "{:02d}min and {:02d}s".format(min, sec)
		print(return_str)
		return return_str


@app.route('/')
def index():
	airports = load_obj("airport")
	companies = load_obj("company")
	airports = sorted([(x, y) for x, y in airports.items()], key=lambda x:x[1])
	companies = sorted([(x, y) for x, y in companies.items()], key=lambda x:x[1])
	return render_template('main.html', airports=airports, companies=companies)

if __name__ == "__main__":
    app.run(debug=True)