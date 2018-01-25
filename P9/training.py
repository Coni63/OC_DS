import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from models import model as md

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

dimensions = np.array(load_obj("datas/dimensions"))
data = np.load('datas/dataset_resized.npz')
X_train, X_test, y_train, y_test, dim_train, dim_test = train_test_split(data["X"], data["y"], dimensions, test_size=0.1, random_state=42)

h, w, c = X_train.shape[1:]
BATCH_SIZE= 5
NB_IMAGES_TRAIN = X_train.shape[0]
NB_IMAGES_TEST = X_test.shape[0]
EPOCHS = 10
SEED = 42

model = md.create_model(h, w, c)
predictions = md.prediction_history(model, X_test, y_test, "resized")  #  1D, 2D, light, resized, rgb
train_generator, test_generator = md.create_generator(X_train, X_test, y_train, y_test, BATCH_SIZE=BATCH_SIZE, seed=SEED)

model.fit_generator(generator = train_generator, 
                    steps_per_epoch = NB_IMAGES_TRAIN//BATCH_SIZE, 
                    epochs=EPOCHS, 
                    validation_data=test_generator,
                    validation_steps = NB_IMAGES_TEST//BATCH_SIZE, 
                    callbacks = [predictions])

model.save("models/model.h5")