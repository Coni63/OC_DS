import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *

import tensorflow as tf

data = np.load('datas/dataset_RGB.npz')
X_train = data["X"]
y_train = data["y"]

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 512, 512, 3

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
# s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['binary_crossentropy'])

data_gen_args = {
    "rotation_range" : 90.,
    "width_shift_range" : 0.1,
    "height_shift_range" : 0.1,
    "zoom_range" : 0.2,
    "shear_range" : 0.2,
    "fill_mode" : "constant",
    "cval" : 0,
    "horizontal_flip" : True,
    "vertical_flip" : True,
    "data_format" : "channels_last",
	"rescale" : 1./255
}

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_datagen.fit(X_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

image_generator = image_datagen.flow(X_train, batch_size=10, seed=seed)
mask_generator = mask_datagen.flow(y_train, batch_size=10, seed=seed)

train_generator = zip(image_generator, mask_generator)

model.fit_generator(train_generator, steps_per_epoch=67, epochs=5)

image_list= [10, 25, 152, 200, 255]
f, axarr = plt.subplots(3, len(image_list), figsize=(20,12))
for idx, img_id in enumerate(image_list):
    y_pred = model.predict(np.expand_dims(X_train[img_id], 0))
    axarr[0, idx].imshow(X_train[img_id, :, : , 0].astype(np.float64), cmap='gray')
    axarr[1, idx].imshow(y_pred[0, :, : , 0], cmap='gray')
    axarr[2, idx].imshow(y_train[img_id, :, : , 0].astype(np.float64), cmap='gray')
plt.savefig("result.png")
#plt.show()