from keras.preprocessing.image import img_to_array, array_to_img, load_img
import numpy as np
import os

# loads training and validation data
# split: proportion of all data that goes toward training
# 1 - split: proportion of all data that goes toward validation
def get_image_data(category, split=0.8):
	data = []
	all_images = os.listdir("./adjusted/" + category)
	for name in all_images:
		data.append(img_to_array(load_img("./adjusted/" + category + "/" + name)))
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


def get_data_generator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True):
	return ImageDataGenerator(
        rescale=rescale,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip)

def main():
	pass




