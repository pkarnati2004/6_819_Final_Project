from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, UpSampling2D
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb, gray2rgb
import numpy as np
import os

# loads training and validation data
# split: proportion of all data that goes toward training
# 1 - split: proportion of all data that goes toward validation
def get_image_data(category, split=0.9):
    data = []
    all_images = os.listdir("./adjusted/" + category)
    for name in all_images:
        data.append(imread("./adjusted/" + category + "/" + name))
    data = np.array(data, dtype=float)
    training_data = data[int((1 - split) * len(data)):]
    validation_data = data[:int((1 - split) * len(data))]
    return (training_data, validation_data)

def get_model():
    # temporarily copied - MODIFY
    model = Sequential()
    model.add(InputLayer(input_shape=(256, 256, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.compile(optimizer='rmsprop', loss='mse')
    return model


# data preparation and augmentation - also adds randomness
def get_data_generator(zca_epsilon=0.005, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, dtype=float):
    return ImageDataGenerator(
        zca_epsilon=zca_epsilon,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        dtype=dtype)


def generate_next_batch(generator, training_data, batch_size):
    for batch in generator.flow(training_data, batch_size=batch_size):
        lab = rgb2lab(batch * 1.0/255)
        l_batch = lab[:,:,:,0]
        ab_batch = lab[:,:,:,1:] / 128
        yield (l_batch.reshape(l_batch.shape[0], l_batch.shape[1], l_batch.shape[2], 1), ab_batch)


def parse_validation_data(validation_data):
    lab = rgb2lab(validation_data * 1.0/255)
    l = lab[:,:,:,0]
    ab = lab[:,:,:,1:] / 128
    return (l.reshape(l.shape[0], l.shape[1], l.shape[2], 1), ab)

def model_by_category(category, existing_weights=None, batch_size=25, epochs=5, steps_per_epoch=25):
    training_data, validation_data = get_image_data(category)
    generator = get_data_generator()
    if existing_weights is None:
        model = get_model()
        model.fit_generator(generate_next_batch(generator, training_data, batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch)
        model.save_weights("./model.h5")
    else:
        model = get_model()
        model.load_weights(existing_weights)
    test_l, test_ab = parse_validation_data(validation_data)
    print(model.evaluate(test_l, test_ab, 10))
    predict_by_category(model, category)

def predict_by_category(model, category):
    all_images = []
    image_names = os.listdir("./testset/" + category + "/bw")
    for name in image_names:
        img = imread("./testset/" + category + "/bw/" + name)
        img = rgb2lab(gray2rgb(img))[:,:,0]
        print(img)
        all_images.append(img)
    all_images = np.array(all_images, dtype=float)
    all_images = all_images.reshape(all_images.shape + (1,))
    predicted = model.predict(all_images)
    predicted = predicted * 128 *3
    for i in range(len(predicted)):
        print(predicted[i])
        img = np.zeros((256,256,3))
        img[:,:,0] = all_images[i][:,:,0]
        img[:,:,1:] = predicted[i]
        imsave("./testset/" + category + "/predicted/" + image_names[i], lab2rgb(img))

model_by_category("beach")
# model_by_category("person", existing_weights="./model.h5")
# model_by_category("animal", existing_weights="./model.h5")

