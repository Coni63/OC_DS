import os
import pandas as pd
import numpy as np
import glob
import random
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model

img_width, img_height = 256, 256
test_data_dir = "test/"
nb_train_samples = 10357
batch_size = 1
latest_save = "vgg19.16.h5"

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(120, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

model_final.load_weights(latest_save)

test_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = test_datagen.flow_from_directory(test_data_dir,
                                                        target_size = (img_height, img_width),
                                                        class_mode = "categorical")

y_pred = model_final.predict_generator(validation_generator, steps = 1)  #   nb_train_samples//batch_size + 1

print(y_pred.shape)
print(np.argmax(y_pred, axis=0))
print(np.max(y_pred, axis=0))