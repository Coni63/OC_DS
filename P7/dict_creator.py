import pandas as pd
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

df = pd.read_csv("label_augmented.csv", index_col=0)
print(df.set_index("id").to_dict())
save_obj(df.set_index("id").to_dict(), "breed_dict_train")