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

def to_week_num(x):
    return ((((x - datetime.datetime(2016,1,1)).days // 7)% 52) + 1)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

@app.route('/predict/', methods=['POST'])
def hello():
	if request.method == 'POST':
		print(request.values)
		index_arr = load_obj("index")
		airport_converter = load_obj("airport_converter")
		freq_airport_converter = load_obj("freq_airport")

		print(airport_converter)
		print(index_arr)

		a = [0 for _ in range(len(index_arr))]
		for i, j in request.values.items():
			if i == "departure":
				airport_id = float(j)
				groupe = airport_converter[airport_id]
				column = "RANK__" + str(groupe)
				a[index_arr.index(column)] = 1
			elif i == "arrival":
				pass
				# groupe = airport_converter[float(j)]
				# column = "ORIGIN_RANK__" + str(groupe)
				# a[index_arr.index(column)] = 1
			elif i == "date":
				date_ = datetime.datetime.strptime(j, "%Y-%m-%d")
				print(date_)
				weekday = "DAY__" + str(date_.isoweekday())
				a[index_arr.index(weekday)] = 1
				weeknum = "WEEK__" + str(to_week_num(date_))
				a[index_arr.index(weeknum)] = 1
				print("weekday : {} - numweek {}".format(weekday, weeknum))
			elif i == "time":
				hh, mm = [int(x) for x in j.split(":")]
				# time_minute = hh*60+mm
				# quarter = time_minute//15
				# shift = abs(quarter - 15) / (96-15)
				# a[index_arr.index("SHIFT")] = shift  # /81 evite le scale
				# print(j, " : shift", shift)
				a[index_arr.index("FL_HOUR")] = hh
			elif i =="company":
				a[index_arr.index(j)] = 1
				print("company", j)
			else:
				print(i, j)

		a[index_arr.index("NUM_FLIGHT")] = freq_airport_converter[airport_id][hh]

		X = np.array([a]).reshape(1, -1)
		print(X)
		scaler = joblib.load("scaler.pkl")
		X_scaled = scaler.transform(X)
		print(X_scaled)
		model = joblib.load("model.pkl")
		lateness = model.predict(X_scaled)[0]
		if lateness > 0:
			result = "<SMALL>Retard :</SMALL>"
		else:
			result = "<SMALL>En avance :</SMALL>"
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