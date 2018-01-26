import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from models import model_extended as md

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

dimensions = np.array(load_obj("datas/dimensions"))
data = np.load('datas/dataset_multi_matrices.npz')
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test, dim_train, dim_test = train_test_split(data["X1"], data["X2"], data["X3"], data["y"], dimensions, test_size=0.1, random_state=42)

h, w, c = X1_train.shape[1:]
BATCH_SIZE = 3
NB_IMAGES_TRAIN = X1_train.shape[0]
NB_IMAGES_TEST = X1_test.shape[0]
EPOCHS = 50
SEED = 42

model = md.create_model(h, w, c)
# print(model.summary())
predictions = md.prediction_history(model, X1_test, X2_test, X3_test, y_test, "extended_large")  #  1D, 2D, light, resized, rgb, extended
train_generator, test_generator = md.create_generator(X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test, BATCH_SIZE=BATCH_SIZE, seed=SEED)

model.fit_generator(generator = train_generator,
                    steps_per_epoch = NB_IMAGES_TRAIN//BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=test_generator,
                    validation_steps = NB_IMAGES_TEST//BATCH_SIZE,
                    callbacks = [predictions]
                    )

model.save("models/model_extended.h5")