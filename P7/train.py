import os
import pandas as pd
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
train_data_dir = "train/resized"
validation_data_dir = "eval/"
nb_train_samples = 9022
nb_validation_samples = 1200
batch_size = 32
epochs = 100

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# for layer in model.layers[:5]:   # a tester
#     layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(120, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

if os.path.exists("vgg19.16.h5"):
    model_final.load_weights('vgg19.16.h5')
    print("Weight Loaded...")
    # model_final = load_model("vgg19_1.h5")
else:
    print("No save found...")

checkpoint = ModelCheckpoint("vgg19.{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=2)
early = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')

train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

val_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = val_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

model_final.fit_generator(
train_generator,
steps_per_epoch = nb_train_samples//batch_size + 1,
epochs = epochs,
validation_data = validation_generator,
validation_steps = nb_validation_samples//batch_size + 1,
callbacks = [checkpoint, early],
verbose = 2)