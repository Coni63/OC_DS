import os
from PIL import Image
import pandas as pd
import numpy as np
import glob

from keras.preprocessing import image
from keras import applications

def get_train_matrices(network, model, path, resize=False):
    X = []
    y = []
    i = 0
    for filename in glob.glob(path):
        name_img = os.path.basename(filename)[:-4]
        classe = label[label["id"] == name_img]["breed"].values[0]
        if resize:
            img = image.load_img(filename, target_size=(224, 224))
            x = image.img_to_array(img)
        else:
            img = image.load_img(filename, target_size=(299, 299))
            x = image.img_to_array(img)
        input_img = np.expand_dims(np.array(x), 0)
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

def get_test_matrices(network, model, path, resize=False):
    X = []
    y = []
    i = 0
    for filename in glob.glob(path):
        name_img = os.path.basename(filename)[:-4]
        if resize:
            img = image.load_img(filename, target_size=(224, 224))
            x = image.img_to_array(img)
        else:
            img = image.load_img(filename, target_size=(299, 299))
            x = image.img_to_array(img)
        input_img = np.expand_dims(np.array(x), 0)
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


label = pd.read_csv("../labels.csv")
train_path =  "../train/resized/*.jpg"
test_path = "../test/resized/*.jpg"

#### Model 1

# print("Loading InceptionResNetV2")
# app = applications.inception_resnet_v2
# model = app.InceptionResNetV2(
#     include_top=False,
#     weights='imagenet',
#     pooling='avg'
# )
#
# print("Generating Train Matrice...")
# X, y = get_train_matrices(app, model, train_path)
# print("Saving Train Matrice...")
# np.savez_compressed('datas/preprocessed_train_InceptionResNetV2.npz', X=X, y=y)
#
# print("Generating Test Matrice...")
# X, y = get_test_matrices(app, model, test_path)
# print("Saving Test Matrice...")
# np.savez_compressed('datas/preprocessed_test_InceptionResNetV2.npz', X=X, y=y)

#### Model 2

# print("Loading InceptionV3")
# app = applications.inception_v3
# model = app.InceptionV3(
#     include_top=False,
#     weights='imagenet',
#     pooling="avg",
# )
#
# print("Generating Train Matrice...")
# X, y = get_train_matrices(app, model, train_path)
# print("Saving Train Matrice...")
# np.savez_compressed('datas/preprocessed_train_InceptionV3.npz', X=X, y=y)
#
# print("Generating Test Matrice...")
# X, y = get_test_matrices(app, model, test_path)
# print("Saving Test Matrice...")
# np.savez_compressed('datas/preprocessed_test_InceptionV3.npz', X=X, y=y)

#### Model 3

# print("Loading VGG16")
# app = applications.vgg16
# model = app.VGG16(
#     include_top=False,
#     weights='imagenet',
#     pooling="avg",
# )
#
# print("Generating Train Matrice...")
# X, y = get_train_matrices(app, model, train_path, True)
# print("Saving Train Matrice...")
# np.savez_compressed('datas/preprocessed_train_VGG16.npz', X=X, y=y)
#
# print("Generating Test Matrice...")
# X, y = get_test_matrices(app, model, test_path, True)
# print("Saving Test Matrice...")
# np.savez_compressed('datas/preprocessed_test_VGG16.npz', X=X, y=y)

#### Model 4

# print("Loading Xception")
# app = applications.xception
# model = app.Xception(
#     include_top=False,
#     weights='imagenet',
#     pooling="avg",
# )
#
# print("Generating Train Matrice...")
# X, y = get_train_matrices(app, model, train_path)
# print("Saving Train Matrice...")
# np.savez_compressed('datas/preprocessed_train_Xception.npz', X=X, y=y)
#
# print("Generating Test Matrice...")
# X, y = get_test_matrices(app, model, test_path)
# print("Saving Test Matrice...")
# np.savez_compressed('datas/preprocessed_test_Xception.npz', X=X, y=y)

#### Model 5

# print("Loading ResNet50")
# app = applications.resnet50
# model = app.ResNet50(
#     include_top=False,
#     weights='imagenet',
#     pooling="avg",
# )
#
# print("Generating Train Matrice...")
# X, y = get_train_matrices(app, model, train_path, True)
# print("Saving Train Matrice...")
# np.savez_compressed('datas/preprocessed_train_ResNet50.npz', X=X, y=y)
#
# print("Generating Test Matrice...")
# X, y = get_test_matrices(app, model, test_path, True)
# print("Saving Test Matrice...")
# np.savez_compressed('datas/preprocessed_test_ResNet50.npz', X=X, y=y)

#### Model 6

# print("Loading MobileNet")
# app = applications.mobilenet
# model = app.MobileNet(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights='imagenet',
#     pooling="avg",
# )
#
# print("Generating Train Matrice...")
# X, y = get_train_matrices(app, model, train_path, True)
# print("Saving Train Matrice...")
# np.savez_compressed('datas/preprocessed_train_MobileNet.npz', X=X, y=y)
#
# print("Generating Test Matrice...")
# X, y = get_test_matrices(app, model, test_path, True)
# print("Saving Test Matrice...")
# np.savez_compressed('datas/preprocessed_test_MobileNet.npz', X=X, y=y)