import numpy as np
import glob 
import cv2
import pickle

from scipy.spatial.distance import cdist

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def extract_SIFT():
	sift = cv2.xfeatures2d.SIFT_create()
	t = None
	for index, images in enumerate(glob.glob("train/*.jpg")):
		if index % 50 == 0:
			print("{}/10357".format(index))
			if index % 1000 == 0 and index != 0:
				np.save("datas/SIFT_descriptor_{}.npy".format(index), t)
				t = None
		img = cv2.imread(images)
		gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		(kps, descs) = sift.detectAndCompute(gray, None)
		descs = descs.astype(np.uint8)
		if t is None:
			t = descs
		else:
			t = np.concatenate([t, descs], axis=0)
	np.save("datas/SIFT_descriptor_{}.npy".format(index), t)


def concatenate_matrices():
	arr = []
	for index, mtx in enumerate(glob.glob("datas/*.npy")):
		print("Loading:", mtx)
		arr.append(np.load(mtx))
	t = np.concatenate(arr, axis=0)
	np.save("datas/SIFT_descriptor_full.npy", t)
	
def evaluate_clustering():
	silhouette = load_obj("datas/silhouette")
	elbow = load_obj("datas/elbow")
	for n_clusters in [50*i for i in range(2, 10)]:
		print("Evaluating {} clusters".format(n_clusters))
		kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1500)
		for index, mtx in enumerate(glob.glob("datas/*.npy")):
			print("Loading :", mtx)
			t = np.load(mtx)
			kmeans.fit(t)
		clusters = kmeans.predict(t)
		elbow_score = sum(np.min(cdist(t, kmeans.cluster_centers_, 'euclidean'), axis=1)) / t.shape[0]
		silhouette_avg = silhouette_score(t, clusters, sample_size=1500)
		print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg, "The average elbow_score is :", elbow_score)
		silhouette[n_clusters] = silhouette_avg
		elbow[n_clusters] = elbow_score
	save_obj(silhouette, "datas/silhouette")
	save_obj(elbow, "datas/elbow")
	
if __name__ == "__main__":
	# extract_SIFT()
	# concatenate_matrices()
	evaluate_clustering()