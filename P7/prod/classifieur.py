import numpy as np
import pickle
from keras import applications
from keras.metrics import top_k_categorical_accuracy
from keras.models import load_model
from keras.preprocessing import image

from sklearn.externals import joblib

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

app = applications.xception
model = app.Xception(
    include_top=False,
    weights='imagenet',
    pooling="avg",
)

breed_dict = load_obj("breed_dict_train")
name = "1b85a40f3c79e9d94e975216792ef52d"
y_true = breed_dict["breed"].get(name, "Unknown")

img = image.load_img("train/resized/1b85a40f3c79e9d94e975216792ef52d.jpg", target_size=(299, 299))

x = image.img_to_array(img)
input_img = np.expand_dims(np.array(x), 0)
input_img = app.preprocess_input(input_img.astype('float32'))
pred = model.predict(input_img)

classifier = load_model('final_classifier.h5', custom_objects={'top3': top3, "top5" : top5})
pred2 = classifier.predict(pred)

index_list = pred2[0].argsort()[::-1][:5]
score_list = pred2[0][index_list]*100
rank_list = list(range(1,6))

Label_binarizer = joblib.load('Label_binarizer.pkl')
print("Vraie Race (si disponible) :", y_true)
for rank, index, score in zip(rank_list, index_list, score_list):
    label = Label_binarizer.classes_[index]
    print("Rank {} - {} ({:.3f}%)".format(rank, label, score))

	
	
# dhole predected dingo : 1b85a40f3c79e9d94e975216792ef52d