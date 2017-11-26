import numpy as np
import pandas as pd

df = pd.read_csv("prod_dataset.csv")

def recommander(film_id = 1, nb_movie=5):
    selected_movie = film_id
    X_embedded = df[["X", "Y", "Z"]].as_matrix()
    center = X_embedded[selected_movie]
    relative_position = X_embedded - center
    distance = np.sqrt(np.sum(np.square(relative_position), axis=1))
    n_closest = np.argsort(distance)[0:nb_movie+1]

    X = np.load('df_encoded.npz')["X"]
    center = X[film_id]
    relative_position = X - center
    dist = np.sqrt(np.sum(np.square(relative_position), axis=1))
    distance = np.sqrt(np.sum(np.square(relative_position), axis=1))
    n_closest_2 = np.argsort(distance)[1:nb_movie+1]

    print("TSNE :")
    print(df.iloc[n_closest][["movie_title", "title_year", "duration", "genres", "country", "imdb_score", "movie_facebook_likes", "director_name"]])
    print("\nModele Simple :")
    print(df.iloc[n_closest_2][["movie_title", "title_year", "duration", "genres", "country", "imdb_score", "movie_facebook_likes","director_name"]])
    print("\n######################################")

if __name__ == "__main__":
    movie_check = [1, 2, 6, 32, 222, 1250, 2500]
    for movie in movie_check:
        recommander(movie)
        print("\n\n")