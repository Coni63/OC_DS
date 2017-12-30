from flask import Flask
from flask import render_template
from flask import request

import numpy as np
import pandas as pd
from operator import itemgetter
import datetime
import pickle
import nltk

from collections import Counter

from scipy.stats import entropy

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
        index_arr = load_obj("mysite/p4/index")                              # mysite/ a jouter en ligne
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
        scaler = joblib.load("mysite/p4/scaler.pkl")                # mysite/ a jouter en ligne
        X_scaled = scaler.transform(X)
        print(X_scaled)
        model = joblib.load("mysite/p4/model.pkl")                  # mysite/ a jouter en ligne 
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
    airports = load_obj("mysite/p4/airport")                        # mysite/ a jouter en ligne
    companies = load_obj("mysite/p4/company")                       # mysite/ a jouter en ligne
    airports = sorted([(x, y) for x, y in airports.items()], key=lambda x:x[1])
    companies = sorted([(x, y) for x, y in companies.items()], key=lambda x:x[1])
    return render_template('p4_main.html', airports=airports, companies=companies)  

##############################
############  P6  ############
##############################

stemmer = nltk.stem.PorterStemmer()
tokenizer = nltk.RegexpTokenizer(r'(\w+|\d+)')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = tokenizer.tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def get_tiidf_matrice(text):
    vectorizer = joblib.load("mysite/p6/TfidfVectorizer")
    return vectorizer.transform([text])

def get_classes(X, qte=5):
    model = joblib.load("mysite/p6/SGDR")
    y_pred = model.predict_proba(X)
    classes = np.argsort(y_pred, axis=1)[:, -qte:].tolist()[0][::-1]  # pour les avoir dans l'ordre de proba
    # print(classes)
    return classes

def classes_to_tags(c):
    mlb = joblib.load("mysite/p6/MultiLabelBinarizer")
    classname = mlb.classes_
    return [classname[classnum] for classnum in c]

def get_tf_matrice(text):
    vectorizer = joblib.load("mysite/p6/CountVectorizer")
    return vectorizer.transform([text])

def perform_lda(X):
    lda = joblib.load('mysite/p6/lda.pkl')
    topic = lda.transform(X)
    print(topic)
    return topic

def JS_Divergence(P, Q):
    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def get_tags_from_lda(X, nb_post_close=10):
    ref = np.load("mysite/p6/LDAMatrix.npy")
    js_result = np.apply_along_axis(JS_Divergence, axis=1, arr=ref, Q=X[0])
    closest_post = np.argsort(js_result)[1: nb_post_close + 1]
    y_train_clean = load_obj("mysite/p6/y_train")
    counter_train = Counter()
    for post in y_train_clean:
        for label in post:
            counter_train[label] += 1

    counter = Counter()
    for post_index in closest_post:
        for key in y_train_clean[post_index]:
            counter[key] += 1

    counter_norm = Counter()
    for tag, freq in counter.items():
        counter_norm[tag] = counter[tag] / counter_train[tag]

    result_non_norm = [tag for tag, freq in counter.most_common(5)]
    result_norm = [tag for tag, freq in counter_norm.most_common(5)]
    return result_non_norm, result_norm

def arr_to_string(arr):
    return "".join(["<code>"+tag+"</code>" for tag in arr])

@app.route('/p6/predict/', methods=['POST'])
def predict_tag():
    if request.method == 'POST':
        for key, content in request.values.items():
            if key == "corpus":
                # partie supervisé
                try:
                    tfidf = get_tiidf_matrice(content)
                    classes = get_classes(tfidf)
                    tags = classes_to_tags(classes)
                except Exception as inst:
                    tags = inst.args

                # partie non sup
                tf = get_tf_matrice(content)
                topics = perform_lda(tf)
                r_non_norm, r_norm = get_tags_from_lda(topics)

                # rendering
                r_non_norm = arr_to_string(r_non_norm)
                r_norm = arr_to_string(r_norm)
                tags = arr_to_string(tags)
        return r_non_norm + "!" + r_norm + "!" + tags
    return "!!!"

@app.route('/p6/')
def load_p6():
    return render_template('p6_main.html')

@app.route('/')
def index():
    return render_template('main.html') 

if __name__ == "__main__":
    app.run(debug=True)