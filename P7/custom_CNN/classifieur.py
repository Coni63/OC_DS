import time
import pickle

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import applications
from keras.models import Model

class History(Callback):
    def on_train_begin(self, logs={}):
        self.epoch_time_start = time.time()
        self.times = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.top3 = []
        self.top5 = []
        self.val_top3 = []
        self.val_top5 = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.top3.append(logs.get('top3'))
        self.top5.append(logs.get('top5'))
        self.val_acc.append(logs.get('val_acc'))
        self.val_top3.append(logs.get('val_top3'))
        self.val_top5.append(logs.get('val_top5'))
        self.save()

    def save(self):
        with open('history1', 'wb') as file:
            pickle.dump(self.convert_to_dict(), file)

    def convert_to_dict(self):
        return {
            "time": self.times,
            "acc": self.acc,
            "val_acc": self.val_acc,
            "loss": self.losses,
            "val_loss": self.val_losses,
            "top3": self.top3,
            "val_top3": self.val_top3,
            "top5": self.top5,
            "val_top5": self.val_top5
        }

def top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top4(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=4)

def top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

padding = "same"

# model = Sequential()
# model.add(Conv2D(8, (3, 3), input_shape=(299, 299, 3), padding=padding))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
#
# model.add(Conv2D(16, (3, 3), padding=padding))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
#
# model.add(Conv2D(32, (3, 3), padding=padding))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
#
# model.add(Conv2D(61, (3, 3), padding=padding))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(200))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(120))
# model.add(Activation('softmax'))

target_size = (224, 224)

app = applications.mobilenet
model = app.MobileNet(
    include_top=False,
    weights=None,
    pooling="avg",
)

x = model.output
x = Dense(200, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(120, activation='softmax')(x)

model = Model(inputs=model.input, outputs=predictions)

history = History()
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
checkpoint = ModelCheckpoint("chkpt.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=2)

model.compile(loss='categorical_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy', top3, top5])

batch_size = 16
nb_image_train = 10222-1200
nb_image_test = 1200

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train',  # this is the target directory
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    'eval',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical')

print(model.summary())

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_image_train // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=nb_image_test // batch_size,
    verbose=2,
    callbacks=[history, checkpoint]  # early_stop
)

model.save_weights('models.h5')
