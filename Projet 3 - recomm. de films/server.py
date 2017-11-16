from flask import Flask
from flask import render_template
import numpy as np
import pandas as pd
from operator import itemgetter

app = Flask(__name__)

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
    df = pd.read_csv("prod_dataset.csv")
    selected_movie = film_id  # 1, 2, 6, 32, 222, 1250, 2500
    X_embedded = df[["X", "Y", "Z"]].as_matrix()
    center = X_embedded[selected_movie]
    relative_position = X_embedded - center
    distance = np.sqrt(np.sum(np.square(relative_position), axis=1))
    n_closest = np.argsort(distance)[1:nb_movies+1]
    #result = df.iloc[n_closest]
    
    vu = df_to_arr(df.iloc[film_id])
    a_voir = [df_to_arr(df.iloc[x]) for x in n_closest]
    return vu, a_voir

@app.route('/recommend/<int:post_id>')
def recommend(post_id):
    vu, a_voir = get_closest(post_id, 5)
#    print(vu)
#    print(a_voir)
    return render_template('recommend.html', vu=vu, a_voir=a_voir)

@app.route('/')
def index():
    df = pd.read_csv("prod_dataset.csv")
    liste_movie = df["movie_title"].values
    liste_movie = [(i, str(x).strip().replace(u'\xa0', u' ')) for i, x in enumerate(liste_movie)]
    liste_movie = sorted(liste_movie, key=itemgetter(1))
    return render_template('main.html', droplist=liste_movie, x_max=len(liste_movie)-1 )

if __name__ == "__main__":
    app.run(debug=True)