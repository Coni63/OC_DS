import os
import glob
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt

# def explore():
#     height_list = defaultdict(int)
#
#     for filename in glob.glob('test/*.jpg'):
#         im = Image.open(filename)
#         w, h = im.size
#         height_list[h] += 1
#
#     for filename in glob.glob('train/*.jpg'):
#         im = Image.open(filename)
#         w, h = im.size
#         height_list[h] += 1

def resize(basewidth = 300):
    # for filename in glob.glob('debug/*.jpg'):
    #     img = Image.open(filename)
    #     img = img.resize((256, 256), Image.ANTIALIAS)
    #     img.save(os.path.join("debug/resized/", os.path.basename(filename)))

    for filename in glob.glob('test/*.jpg'):
        img = Image.open(filename)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(os.path.join("test/resized/", os.path.basename(filename)))

    for filename in glob.glob('train/*.jpg'):
        img = Image.open(filename)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(os.path.join("train/resized/", os.path.basename(filename)))

    # for filename in glob.glob('test/*.jpg'):
    #     img = Image.open(filename)
    #     wpercent = (basewidth/float(img.size[0]))
    #     hsize = int((float(img.size[1])*float(wpercent)))
    #     img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    #     img.save(os.path.join("test/resized/", os.path.basename(filename)))
    #
    # for filename in glob.glob('train/*.jpg'):
    #     img = Image.open(filename)
    #     wpercent = (basewidth/float(img.size[0]))
    #     hsize = int((float(img.size[1])*float(wpercent)))
    #     img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    #     img.save(os.path.join("train/resized/", os.path.basename(filename)))

# def expand():
#     max_height = 0
#     for filename in glob.glob('test/resized/*.jpg'):
#         im = Image.open(filename)
#         w, h = im.size
#         max_height = max(max_height, h)
#
#     for filename in glob.glob('train/resized/*.jpg'):
#         im = Image.open(filename)
#         w, h = im.size
#         max_height = max(max_height, h)
#
#     for filename in glob.glob('test/resized/*.jpg'):
#         old_im = Image.open(filename)
#         old_size = im.size
#         new_size = (300, max_height)
#         new_im = Image.new("RGB", new_size)
#         new_im.paste(old_im, (0, (new_size[1] - old_size[1]) // 2))
#         new_im.save(os.path.join("test/resized/", os.path.basename(filename)))
#
#     for filename in glob.glob('train/resized/*.jpg'):
#         old_im = Image.open(filename)
#         old_size = im.size
#         new_size = (300, max_height)
#         new_im = Image.new("RGB", new_size)
#         new_im.paste(old_im, (0, (new_size[1] - old_size[1]) // 2))
#         new_im.save(os.path.join("train/resized/", os.path.basename(filename)))



if __name__ == "__main__":
    #explore()
    resize()
    #expand()