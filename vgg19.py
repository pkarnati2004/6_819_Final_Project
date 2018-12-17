from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications import VGG19
from keras.utils.training_utils import multi_gpu_model
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb, gray2rgb, rgb2gray
from skimage.transform import resize
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint
import utils
import numpy as np
import os

from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# initialize VGG19 model
image_size=(256, 256, 3)
vgg = VGG19(weights='imagenet', include_top = False, input_shape=image_size)



