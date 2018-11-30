from keras.preprocessing.image import img_to_array, array_to_img, load_img
from skimage.transform import resize
import numpy as np
import os

def filter_and_adjust_images_in_category(category, size):
    all_images = os.listdir("./images/" + category)
    for name in all_images:
        img = img_to_array(load_img("./images/" + category + "/" + name))
        img = np.array(img, dtype=float)

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
            img = resize(img, (size, size, 3))
            new_image = array_to_img(img)
            # saves image
            if not os.path.isdir("./adjusted"):
                os.mkdir("./adjusted")
            if not os.path.isdir("./adjusted/" + category):
                os.mkdir("./adjusted/" + category)
            with open("./adjusted/" + category + "/" + name, "w") as f:
                new_image.save(f)
            print("saved to ", "./adjusted/" + category + "/" + name)

print(filter_and_adjust_images_in_category("mountain", 256))