from flask import Flask
from flask import render_template
from flask import request

import numpy as np
import pandas as pd
from operator import itemgetter
import datetime
import pickle

from sklearn.externals import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

##############################
############  P3  ############
##############################

def df_to_arr(result):
    a = []
    a.append(str(result["movie_title"]).strip().replace(u'\xa0', u' '))
    a.append(int(result["title_year"]))
    a.append(int(result["duration"]))
    a.append(result["genres"])
    a.append(result["country"])
    a.append(result["imdb_score"])
    a.append(result["movie_facebook_likes"])
    a.append(result["director_name"])
    a.append("{}, {}, {}".format(result["actor_1_name"], result["actor_2_name"], result["actor_3_name"]))
    return a

def get_closest(film_id=1, nb_movies = 5):
    df = pd.read_csv("mysite/p3/prod_dataset.csv")          # mysite/ a jouter en ligne
    selected_movie = film_id  # 1, 2, 6, 32, 222, 1250, 2500
    X_embedded = df[["X", "Y", "Z"]].as_matrix()
    center = X_embedded[selected_movie]
    relative_position = X_embedded - center
    distance = np.sqrt(np.sum(np.square(relative_position), axis=1))
    n_closest = np.argsort(distance)[1:nb_movies+1]

    X = np.load('mysite/p3/df_encoded.npz')["X"]             # mysite/ a jouter en ligne
    center = X[film_id]
    relative_position = X - center
    dist = np.sqrt(np.sum(np.square(relative_position), axis=1))
    distance = np.sqrt(np.sum(np.square(relative_position), axis=1))
    n_closest_2 = np.argsort(distance)[1:nb_movies+1]
    
    vu = df_to_arr(df.iloc[film_id])
    a_voir = [df_to_arr(df.iloc[x]) for x in n_closest]
    a_voir_2 = [df_to_arr(df.iloc[x]) for x in n_closest_2]
    return vu, a_voir, a_voir_2

@app.route('/p3/recommend/<int:post_id>')
def recommend(post_id):
    vu, a_voir, a_voir_2 = get_closest(post_id, 5)
    return render_template('p3_recommend.html', vu=vu, a_voir=a_voir, a_voir_2 = a_voir_2)
	
@app.route('/p3/')
def load_p3():
    df = pd.read_csv("mysite/p3/prod_dataset.csv")     # mysite/ a jouter en ligne
    liste_movie = df["movie_title"].values
    liste_movie = [(i, str(x).strip().replace(u'\xa0', u' ')) for i, x in enumerate(liste_movie)]
    liste_movie = sorted(liste_movie, key=itemgetter(1))
    return render_template('p3_main.html', droplist=liste_movie, x_max=len(liste_movie)-1 )

##############################
############  P4  ############
##############################


def to_week_num(x):
    return ((((x - datetime.datetime(2016,1,1)).days // 7)% 52) + 1)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

@app.route('/p4/predict/', methods=['POST'])
def make_prediction():
	if request.method == 'POST':
		print(request.values)
		index_arr = load_obj("mysite/p4/index")          					 # mysite/ a jouter en ligne
		airport_converter = load_obj("mysite/p4/airport_converter")          # mysite/ a jouter en ligne
		freq_airport_converter = load_obj("mysite/p4/freq_airport")          # mysite/ a jouter en ligne

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
		scaler = joblib.load("mysite/p4/scaler.pkl")          		# mysite/ a jouter en ligne
		X_scaled = scaler.transform(X)
		print(X_scaled)
		model = joblib.load("mysite/p4/model.pkl")          		# mysite/ a jouter en ligne	
		lateness = model.predict(X_scaled)[0]
		if lateness > 0:
			result = "<SMALL>Retard :</SMALL>"
		else:
			result = "<SMALL>En avance :</SMALL>"
		min, sec = int(lateness), int((lateness%1)*60)
		return_str = result + "{:02d}min and {:02d}s".format(min, sec)
		print(return_str)
		return return_str

@app.route('/p4/')
def load_p4():
	airports = load_obj("mysite/p4/airport")          				# mysite/ a jouter en ligne
	companies = load_obj("mysite/p4/company")          				# mysite/ a jouter en ligne
	airports = sorted([(x, y) for x, y in airports.items()], key=lambda x:x[1])
	companies = sorted([(x, y) for x, y in companies.items()], key=lambda x:x[1])
	return render_template('p4_main.html', airports=airports, companies=companies)	
	
@app.route('/')
def index():
    return render_template('main.html')	

if __name__ == "__main__":
    app.run(debug=True)