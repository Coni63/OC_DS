import pandas as pd
import numpy as np

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
    df = pd.read_csv("p3/prod_dataset.csv")          # mysite/ a jouter en ligne
    selected_movie = film_id  # 1, 2, 6, 32, 222, 1250, 2500
    X_embedded = df[["X", "Y", "Z"]].as_matrix()
    center = X_embedded[selected_movie]
    relative_position = X_embedded - center
    distance = np.sqrt(np.sum(np.square(relative_position), axis=1))
    n_closest = np.argsort(distance)[1:nb_movies+1]

    X = np.load("p3/df_encoded.npz")["X"]             # mysite/ a jouter en ligne
    center = X[film_id]
    relative_position = X - center
    dist = np.sqrt(np.sum(np.square(relative_position), axis=1))
    distance = np.sqrt(np.sum(np.square(relative_position), axis=1))
    n_closest_2 = np.argsort(distance)[1:nb_movies+1]
    
    vu = df_to_arr(df.iloc[film_id])
    a_voir = [df_to_arr(df.iloc[x]) for x in n_closest]
    a_voir_2 = [df_to_arr(df.iloc[x]) for x in n_closest_2]
    return vu, a_voir, a_voir_2