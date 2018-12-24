from skimage.transform import resize
from skimage.io import imread, imsave
from skimage.color import rgb2gray
import numpy as np
import random
import os

# hello
def create_test_set(category, proportion=0.10):
    if not os.path.isdir("./testset"):
        os.mkdir("./testset")
    if os.path.isdir("./testset/" + category):
        print("Test set for " + category + " already created. Aborted.")
        return
    os.mkdir("./testset/" + category)
    os.mkdir("./testset/" + category + "/bw")
    os.mkdir("./testset/" + category + "/predicted")
    os.mkdir("./testset/" + category + "/expected")
    image_names = os.listdir("./adjusted/" + category)
    for name in image_names:
        num = random.random()
        if num < proportion:
            img = imread("./adjusted/" + category + "/" + name)
            imsave("./testset/" + category + "/bw/" + name, rgb2gray(img))
            imsave("./testset/" + category + "/expected/" + name, img)


# resizes/shapes images of category to squares of side length size
def filter_and_adjust_images_in_category(category, size, proportion=0.10):
    image_names = os.listdir("./images/" + category)
    if not os.path.isdir("./testset"):
        os.mkdir("./testset")
    if os.path.isdir("./testset/" + category):
        print("Test set for " + category + " already created. Aborted.")
        return
    os.mkdir("./testset/" + category)
    os.mkdir("./testset/" + category + "/bw")
    os.mkdir("./testset/" + category + "/predicted")
    os.mkdir("./testset/" + category + "/expected")
    for name in image_names:
        img = imread("./images/" + category + "/" + name)

        # check that images are large enough
        if img.shape[0] < size or img.shape[1] < size:
            continue

        # make images square
        if img.shape[0] != img.shape[1]:
            # make even shaped for resizing
            if img.shape[0] % 2 == 1:
                img = img[1:,:,:]
            if img.shape[1] % 2 == 1:
                img = img[:,1:,:]
            # make sides equal length
            if img.shape[0] > img.shape[1]:
                img = img[abs(int((img.shape[0] - img.shape[1])/2)):abs(int((img.shape[0] - img.shape[1])/2)) + img.shape[1],:,:]
            elif img.shape[0] < img.shape[1]:
                img = img[:,abs(int((img.shape[0] - img.shape[1])/2)):abs(int((img.shape[0] - img.shape[1])/2)) + img.shape[0],:]

        # resizes image 
        img = resize(img, (size, size, 3))

        # saves image to either main training/validation set or to test set
        num = random.random()
        if num < proportion:
            imsave("./testset/" + category + "/bw/" + name, rgb2gray(img))
            imsave("./testset/" + category + "/expected/" + name, img)
        else:
            if not os.path.isdir("./adjusted"):
                os.mkdir("./adjusted")
            if not os.path.isdir("./adjusted/" + category):
                os.mkdir("./adjusted/" + category)
            imsave("./adjusted/" + category + "/" + name, img)
            print("saved to ", "./adjusted/" + category + "/" + name)

# create bw adjusted images 
def create_bw_images():
    categories = sorted(os.listdir("./adjusted/"))
    print(categories)

    if not os.path.isdir("./adjustedbw"):
        os.mkdir("./adjustedbw")

    for category in categories:
        all_images = os.listdir("./adjusted/" + category)
        if not os.path.isdir("./adjustedbw/" + category):
            os.mkdir("./adjustedbw/" + category)
        for name in all_images:
            img = imread("./adjusted/" + category + "/" + name)
            gray = rgb2gray(img)
            imsave('./adjustedbw/' + category + '/' + name, gray)
            print("saved to ", './adjustedbw/' + category + '/' + name)

# category = [
#     'autumn',  'bus',  'christmas', 'dog',  'dress',  'forest',  'fruit',  'house',  'mountain',  'road',  'smoldoggo',  'sunset',  'waterfall'
# ]

# for c in category:
#     print(filter_and_adjust_images_in_category(c, 256))
# # create_test_set("beach")
# # filter_and_adjust_images_in_category("beach", 256)


create_bw_images()
