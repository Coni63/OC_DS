import sys
import numpy as np

from keras import applications
from keras.metrics import top_k_categorical_accuracy
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

from sklearn.externals import joblib

def swish(x):
    return x*K.sigmoid(x)
   
def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
    
def top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def load_extractor():
    app = applications.inception_resnet_v2
    model = app.InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        pooling="avg")
    return app, model
    
def preprocess_image(img_path = None):
    if img_path is None:
        print("Merci de fournir un chemin pour l'image")
        return None
    try:
        input_img = image.load_img(img_path, target_size=(299, 299))
        input_img = image.img_to_array(input_img)
        input_img = np.expand_dims(np.array(input_img), 0)
        input_img = app.preprocess_input(input_img.astype('float32'))
    except:
        print(img_path, "n'est pas un chemin valide vers une image. Passage au suivant")
        return None
    return input_img
    
def extract_features(X, model):
    feature = model.predict(X)
    return feature
    
def make_prediction(X):
    classifier = load_model("final_classifier.h5", custom_objects={'top3': top3, "top5" : top5})  # 
    prediction = classifier.predict(X)
    return prediction
    
def get_top_5(y_pred, idx=0, top=5):
    index_list = y_pred[idx].argsort()[::-1][:top]
    score_list = y_pred[idx][index_list]*100
    rank_list = list(range(1,top+1))
    
    Label_binarizer = joblib.load('Label_binarizer.pkl')
    for rank, index, score in zip(rank_list, index_list, score_list):
        label = Label_binarizer.classes_[index]
        print("Rank {} - {} ({:.3f}%)".format(rank, label, score))

		
if __name__ == "__main__":
    get_custom_objects().update({'swish': swish})

    print("Loading Model ...")
    app, mdl = load_extractor()
    
    files = sys.argv[1:]
    featured_img = []
    final_img = []
    for idx, file in enumerate(files):
        print("Pre-processing image", idx+1)
        prepared_img = preprocess_image(file)
        if prepared_img is not None:
            featured_img.append(prepared_img[0])
            final_img.append(file)
    
    featured_img = np.array(featured_img)
	
    if len(featured_img) == 0:
        sys.exit()
    
    print("Matrice d'images :", featured_img.shape)
    
    print("Extracting Features")
    features = extract_features(featured_img, mdl)
    
    print("Matrice de features :", features.shape)
	
    print("Making Prediction")
    classes = make_prediction(features)
    
    print("Results")
    for idx, file in enumerate(final_img):
        print("\n", file)
        get_top_5(classes, idx=idx, top=5)