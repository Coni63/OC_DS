import numpy as np
import pandas as pd

df = pd.read_csv("prod_dataset.csv")

# selected_movie = 1  # 1, 2, 6, 32, 222, 1250, 2500
# X_embedded = df[["X", "Y"]].as_matrix()
# print(X_embedded.shape)
#
# center = X_embedded[selected_movie]
# relative_position = X_embedded - center
# distance = np.sqrt(np.sum(np.square(relative_position), axis=1))
# n_closest = np.argsort(distance)[0:5+1]
# result = df.iloc[n_closest]
# print(result)

# result = df.iloc[selected_movie]
# print(result)
#
# a = []
# a.append(result["movie_title"].replace(u'\xa0', u' '))
# a.append(int(result["title_year"]))
# a.append(int(result["duration"]))
# a.append(result["genres"])
# a.append(result["country"])
# a.append(result["imdb_score"])
# a.append(result["movie_facebook_likes"])
# a.append(result["director_name"])
# a.append("{}, {}, {}".format(result["actor_1_name"], result["actor_2_name"], result["actor_3_name"]))
# print(a)

a = df["movie_title"].values
print(a[:10])
a = [(i, str(x).strip().replace(u'\xa0', u' ')) for i, x in enumerate(a)]
print(a[:10])
from operator import itemgetter
a = sorted(a, key=itemgetter(1))
print(a[:10])