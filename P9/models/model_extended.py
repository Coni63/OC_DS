import numpy as np
import pickle
import imageio
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.optimizers import Adam, SGD
from keras import regularizers

import tensorflow as tf

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class prediction_history(Callback):
    def __init__(self, model, X_test, X2_test, X3_test, y_test, folder):
        self.img_id = [8, 11, 12, 24]
        self.epoch = 0
        self.val_loss = []
        self.loss = []
        self.X_test = [X_test, X2_test, X3_test]
        self.y_test = y_test
        self.model = model
        self.folder = folder
        for idx in self.img_id:
            self.generate_image(X_test[idx, :, :, 0], "Init", idx)
            self.generate_image(y_test[idx, :, :, 0], "Mask", idx)

    def on_epoch_end(self, epoch, logs={}, model=None):
        self.epoch += 1
        for idx in self.img_id:
            X = [
                np.expand_dims(self.X_test[0][idx], 0),
                np.expand_dims(self.X_test[1][idx], 0),
                np.expand_dims(self.X_test[2][idx], 0)
            ]
            y_pred = self.model.predict(X)
            self.generate_image(y_pred[0, :, :, 0], "Epoch", idx)
        self.val_loss.append(logs.get('val_loss'))
        self.loss.append(logs.get('loss'))

    def generate_image(self, img, name="Epoch", index=0):
        plt.imshow(img, cmap='gray')
        plt.title("Epoch {}".format(self.epoch))
        plt.savefig("img/training/{}/{}/{}{:03d}.png".format(self.folder, index, name, self.epoch))

    def on_train_end(self, logs={}):
        result = {"val_loss": self.val_loss, "loss": self.loss}
        save_obj(result, "datas/results_unet_extended_2")
        for img_id in self.img_id:
            images = []
            for i in range(1, self.epoch):
                images.append(imageio.imread("img/training/{}/{}/Epoch{:03d}.png".format(self.folder, img_id, i)))
            output_file = 'img/training_unet_extended-{}.gif'.format(img_id)
            imageio.mimsave(output_file, images, duration=0.5)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def create_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    K_REG = None #regularizers.l1(0.1)
    BIAS_INIT = "zeros" # "he_normal"
    KERNEL_INIT = "he_normal" # "glorot_uniform"
    ACTIVATION = "relu"

    # Convolution 1

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))

    c1 = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (inputs)
    c1 = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (p1)
    c2 = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (p2)
    c3 = Conv2D(32, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (p3)
    c4 = Conv2D(64, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    # Convolution 2

    inputs2 = Input((IMG_HEIGHT, IMG_WIDTH, 1))

    c12 = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (inputs2)
    c12 = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c12)
    p12 = MaxPooling2D((2, 2)) (c12)

    c22 = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (p12)
    c22 = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c22)
    p22 = MaxPooling2D((2, 2)) (c22)

    c32 = Conv2D(32, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (p22)
    c32 = Conv2D(32, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c32)
    p32 = MaxPooling2D((2, 2)) (c32)

    c42 = Conv2D(64, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (p32)
    c42 = Conv2D(64, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c42)
    p42 = MaxPooling2D(pool_size=(2, 2)) (c42)

    # Convolution 3

    inputs3 = Input((IMG_HEIGHT, IMG_WIDTH, 3))

    c13 = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (inputs3)
    c13 = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c13)
    p13 = MaxPooling2D((2, 2)) (c13)

    c23 = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (p13)
    c23 = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c23)
    p23 = MaxPooling2D((2, 2)) (c23)

    c33 = Conv2D(32, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (p23)
    c33 = Conv2D(32, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c33)
    p33 = MaxPooling2D((2, 2)) (c33)

    c43 = Conv2D(64, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (p33)
    c43 = Conv2D(64, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c43)
    p43 = MaxPooling2D(pool_size=(2, 2)) (c43)

    ### Deconvolution

    p5 = concatenate([p4, p42, p43])

    c5 = Conv2D(128, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (p5)
    c5 = Conv2D(128, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4, c42, c43])
    c6 = Conv2D(64, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (u6)
    c6 = Conv2D(64, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3, c32, c33])
    c7 = Conv2D(32, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (u7)
    c7 = Conv2D(32, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2, c22, c23])
    c8 = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (u8)
    c8 = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1, c12, c13], axis=3)
    c9 = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (u9)
    c9 = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same', kernel_initializer=KERNEL_INIT, bias_initializer = BIAS_INIT, kernel_regularizer=K_REG) (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs, inputs2, inputs3], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['binary_crossentropy'])

    return model

def batch_train(train_generator):
    img_gen, mask_gen = train_generator
    while True:
        x1 = next(img_gen[0])
        x2 = next(img_gen[1])
        x3 = next(img_gen[2])
        m =  next(mask_gen[0])
        #if x1.shape[0] == x2.shape[0] and x1.shape[0] == x3.shape[0] and x1.shape[0] == m.shape[0]:
        yield [x1, x2, x3], [m]

def batch_test(test_generator):
    img_gen, mask_gen = test_generator
    while True:
        x1 = next(img_gen[0])
        x2 = next(img_gen[1])
        x3 = next(img_gen[2])
        m =  next(mask_gen[0])
        #if x1.shape[0] == x2.shape[0] and x1.shape[0] == x3.shape[0] and x1.shape[0] == m.shape[0]:
        yield [x1, x2, x3], [m]


def create_generator(X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test, BATCH_SIZE, seed = 42):
    data_gen_args_train = {
        # "rotation_range" : 90.,
        # "width_shift_range" : 0.1,
        # "height_shift_range" : 0.1,
        # "zoom_range" : 0.2,
        # "shear_range" : 0.2,
        # "fill_mode" : "nearest",
        # "cval" : 0,
        "horizontal_flip" : True,
        "vertical_flip" : True,
        "data_format" : "channels_last",
        "rescale" : 1./255                      # convert uint8 to float32
    }

    data_gen_args_test = {
        "horizontal_flip" : False,
        "vertical_flip" : False,
        "data_format" : "channels_last",
        "rescale" : 1./255                      # convert uint8 to float32
    }

    # Creation of image data augmenter for train set
    image1_datagen_train = ImageDataGenerator(**data_gen_args_train)
    image2_datagen_train = ImageDataGenerator(**data_gen_args_train)
    image3_datagen_train = ImageDataGenerator(**data_gen_args_train)
    mask_datagen_train = ImageDataGenerator(**data_gen_args_train)

    # fit this generator to train images
    image1_datagen_train.fit(X1_train, augment=True, seed=seed)
    image2_datagen_train.fit(X2_train, augment=True, seed=seed)
    image3_datagen_train.fit(X3_train, augment=True, seed=seed)
    mask_datagen_train.fit(y_train, augment=True, seed=seed)

    # create a generator of train images augmented
    image1_generator_train = image1_datagen_train.flow(X1_train, batch_size=BATCH_SIZE, seed=seed)
    image2_generator_train = image2_datagen_train.flow(X2_train, batch_size=BATCH_SIZE, seed=seed)
    image3_generator_train = image3_datagen_train.flow(X3_train, batch_size=BATCH_SIZE, seed=seed)
    mask_generator_train = mask_datagen_train.flow(y_train, batch_size=BATCH_SIZE, seed=seed)

    # Creation of data augmenter for test set
    image1_datagen_test = ImageDataGenerator(**data_gen_args_test)
    image2_datagen_test = ImageDataGenerator(**data_gen_args_test)
    image3_datagen_test = ImageDataGenerator(**data_gen_args_test)
    mask_datagen_test = ImageDataGenerator(**data_gen_args_test)

    # fit this generator to test images
    image1_datagen_test.fit(X1_test, augment=True, seed=seed)
    image2_datagen_test.fit(X2_test, augment=True, seed=seed)
    image3_datagen_test.fit(X3_test, augment=True, seed=seed)
    mask_datagen_test.fit(y_test, augment=True, seed=seed)

    # create a generator of test images augmented
    image1_generator_test = image1_datagen_test.flow(X1_test, batch_size=BATCH_SIZE, seed=seed)
    image2_generator_test = image2_datagen_test.flow(X2_test, batch_size=BATCH_SIZE, seed=seed)
    image3_generator_test = image3_datagen_test.flow(X3_test, batch_size=BATCH_SIZE, seed=seed)
    mask_generator_test = mask_datagen_test.flow(y_test, batch_size=BATCH_SIZE, seed=seed)

    train_generator = [[image1_generator_train, image2_generator_train, image3_generator_train], [mask_generator_train]]
    test_generator = [[image1_generator_test, image2_generator_test, image3_generator_test], [mask_generator_test]]

    gen_train = batch_train(train_generator)
    gen_test = batch_test(test_generator)

    return gen_train, gen_test


if __name__ == "__main__":
    print("00")