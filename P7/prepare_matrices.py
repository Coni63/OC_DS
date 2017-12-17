import os
from PIL import Image
import pandas as pd
import numpy as np
import glob

from keras import applications

def get_train_matrices(network, model, path):
    X = []
    y = []
    i = 0
    for filename in glob.glob(path):
        name_img = os.path.basename(filename)[:-4]
        classe = label[label["id"] == name_img]["breed"].values[0]
        input_img = np.expand_dims(np.array(Image.open(filename)), 0)
        input_img = network.preprocess_input(input_img.astype('float32'))
        pred = model.predict(input_img)
        X.append(pred[0])
        y.append(classe)
        if i%100 == 0:
            print("Train Image :" ,i)
        i += 1

    X = np.array(X)
    y = np.array(y)
    return X, y

def get_test_matrices(network, model, path):
    X = []
    y = []
    i = 0
    for filename in glob.glob(path):
        name_img = os.path.basename(filename)[:-4]
        input_img = np.expand_dims(np.array(Image.open(filename)), 0)
        input_img = network.preprocess_input(input_img.astype('float32'))
        pred = model.predict(input_img)
        X.append(pred[0])
        y.append(name_img)
        if i%100 == 0:
            print("Test Image :" ,i)
        i += 1

    X = np.array(X)
    y = np.array(y)
    return X, y


label = pd.read_csv("label_augmented.csv", index_col=0)

#### Model 1

print("Loading InceptionResNetV2")
app = applications.inception_resnet_v2
model = app.InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

for layer in model.layers:
    layer.trainable = False

print("Generating Train Matrice...")
X, y = get_train_matrices(app, model, 'train/resized/*.jpg')
print("Saving Train Matrice...")
np.savez_compressed('preprocessed_train_InceptionResNetV2.npz', X=X, y=y)

print("Generating Test Matrice...")
X, y = get_test_matrices(app, model, 'test/resized/*.jpg')
print("Saving Test Matrice...")
np.savez_compressed('preprocessed_test_InceptionResNetV2.npz', X=X, y=y)

#### Model 2

print("Loading InceptionV3")
app = applications.inception_v3
model = app.InceptionV3(
    include_top=False,
    weights='imagenet',
    pooling="avg",
)

for layer in model.layers:
    layer.trainable = False

print("Generating Train Matrice...")
X, y = get_train_matrices(app, model, 'train/resized/*.jpg')
print("Saving Train Matrice...")
np.savez_compressed('preprocessed_train_InceptionV3.npz', X=X, y=y)

print("Generating Test Matrice...")
X, y = get_test_matrices(app, model, 'test/resized/*.jpg')
print("Saving Test Matrice...")
np.savez_compressed('preprocessed_test_InceptionV3.npz', X=X, y=y)
