import pandas as pd
import glob
import os
from shutil import copyfile
import random

# Mise en  place des folder pour le training
df = pd.read_csv("../labels.csv")

for filename in glob.glob('../train/resized/*.jpg'):
    name_img = os.path.basename(filename)[:-4]
    classe = df[df["id"] == name_img]["breed"].values[0]
    if not os.path.isdir("train/" + classe):
        os.mkdir("train/" + classe)
    copyfile(filename, os.path.join("train", classe, name_img+".jpg"))

# Deplacement de 10 images par classes pour l'eval
for folder in os.listdir("train/"):
    for i in range(10):
        img = random.choice(os.listdir(os.path.join("train/", folder)))
        img_path = os.path.join("train/", folder, img)
        to_path = os.path.join("eval", folder, img)
        if not os.path.isdir("eval/" + folder):
            os.mkdir("eval/" + folder)
        os.rename(img_path, to_path)