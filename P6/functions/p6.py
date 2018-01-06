import numpy as np
import nltk
import pickle

from collections import Counter

from scipy.stats import entropy
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

class Classifier(SGDClassifier):
    def __init__(self, penalty=None, alpha=0.0001):
        super(Classifier, self).__init__(n_jobs=-1, 
                                         loss="log",
                                         penalty=penalty, 
                                         alpha=alpha, 
                                         max_iter=20,
                                         tol=1e-3)
    
    def predict_proba(self, X):
        return super(Classifier, self).predict_proba(X)
		
		
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#stemmer = nltk.stem.PorterStemmer()
#tokenizer = nltk.RegexpTokenizer(r'(\w+|\d+)')

#def stem_tokens(tokens, stemmer):
#    stemmed = []
#    for item in tokens:
#        stemmed.append(stemmer.stem(item))
#    return stemmed

#def tokenize(text):
#    tokens = tokenizer.tokenize(text)
#    stems = stem_tokens(tokens, stemmer)
#    return stems
	
def prepare_text(text):
    tokenizer = nltk.RegexpTokenizer(r'(\w+|\d+)')
    tokens = tokenizer.tokenize(text)
    stemmer = nltk.stem.PorterStemmer()
    stemmed = [stemmer.stem(item) for item in tokens]
    return stemmed

def get_tiidf_matrice(text):
    vectorizer = joblib.load("p6/TfidfVectorizer")
    return vectorizer.transform([text])

#def get_classes(X, qte=5):
#    model = joblib.load("p6/SGDR")
#    y_pred = model.predict_proba(X)
#    classes = np.argsort(y_pred, axis=1)[:, -qte:].tolist()[0][::-1]  # pour les avoir dans l'ordre de proba
#    return classes

def get_classes(X, qte=5):
    model = joblib.load("p6/MOC")
    y_pred = model.predict_proba(X)
    y_pred = np.delete(y_pred, 0, axis=2)[:, :, 0].T
    classes = np.argsort(y_pred, axis=1)[:, -qte:].tolist()[0][::-1]  # pour les avoir dans l'ordre de proba
    # print(classes)
    return classes

def classes_to_tags(c):
    mlb = joblib.load("p6/MultiLabelBinarizer")
    classname = mlb.classes_
    return [classname[classnum] for classnum in c]

def get_tf_matrice(text):
    vectorizer = joblib.load("p6/CountVectorizer")
    return vectorizer.transform([text])

def perform_lda(X):
    lda = joblib.load('p6/lda.pkl')
    topic = lda.transform(X)
    print(topic)
    return topic

def JS_Divergence(P, Q):
    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def get_tags_from_lda(X, nb_post_close=10):
    ref = np.load("p6/LDAMatrix.npy")
    js_result = np.apply_along_axis(JS_Divergence, axis=1, arr=ref, Q=X[0])
    closest_post = np.argsort(js_result)[1: nb_post_close + 1]
    y_train_clean = load_obj("p6/y_train")
    counter_train = Counter()
    for post in y_train_clean:
        for label in post:
            counter_train[label] += 1

    counter = Counter()
    for post_index in closest_post:
        for key in y_train_clean[post_index]:
            counter[key] += 1

    counter_norm = Counter()
    for tag, freq in counter.items():
        counter_norm[tag] = counter[tag] / counter_train[tag]

    result_non_norm = [tag for tag, freq in counter.most_common(5)]
    result_norm = [tag for tag, freq in counter_norm.most_common(5)]
    return result_non_norm, result_norm

def arr_to_string(arr):
    return "".join(["<code>"+tag+"</code>" for tag in arr])