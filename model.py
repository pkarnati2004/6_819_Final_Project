from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, UpSampling2D
from skimage.io import imread, imsave, imshow
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
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

def model_by_category(category, existing_weights=None, continue_training=False, batch_size=25, epochs=80, steps_per_epoch=25, epoch_str=''):
    training_data, validation_data = get_image_data(category)
    generator = get_data_generator()
    if not os.path.isdir("./models/beta/" + category):
        os.mkdir("./models/beta/" + category)
    filepath = "models/beta/" + category + "/beta-weights-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period=1)
    if existing_weights is None:
        model = get_model()
        model.fit_generator(generate_next_batch(generator, training_data, batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint])
        model.save_weights("./model.h5")
    else:
        print('---- GETTING MODEL ----')
        model = get_model()
        model.load_weights(existing_weights)
        if continue_training:
            model.fit_generator(generate_next_batch(generator, training_data, batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint])
            model.save_weights("./model.h5")
    test_l, test_ab = parse_validation_data(validation_data)
    predict_by_category(model, category, epoch_str)

def predict_by_category(model, category, epoch_str=''):
    all_images = []
    image_names = os.listdir("./testset/" + category + "/bw")
    for name in image_names:
        img = imread("./testset/" + category + "/bw/" + name)
        img = rgb2lab(gray2rgb(img))[:,:,0]
        all_images.append(img)
    all_images = np.array(all_images, dtype=float)
    all_images = all_images.reshape(all_images.shape + (1,))
    predicted = model.predict(all_images)
    predicted = predicted * 128 * 3
    dirpath = "./testset/" + category + "/predicted/" 
    if len(epoch_str) > 1:
        dirpath += epoch_str + "/"
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    for i in range(len(predicted)):
        img = np.zeros((256,256,3))
        img[:,:,0] = all_images[i][:,:,0]
        img[:,:,1:] = predicted[i]
        print(dirpath + image_names[i])
        imsave(dirpath + image_names[i], lab2rgb(img))

def test_epoch(category, str_num):
    weight_path = "./models/best_beta/" + 'city' + "/beta-weights-" + str_num + ".hdf5"
    model_by_category(category, existing_weights=weight_path, epoch_str=str_num)

def continue_training_on_model(category, str_num):
    weight_path = "./models/beta/" + category + "/beta-weights-" + str_num + ".hdf5"
    model_by_category(category, existing_weights=weight_path, continue_training=True)  

def convert_to_lab(impath):
    img = imread(impath) / 255
    print(img)
    print(img.shape)
    lab = rgb2lab(gray2rgb(img))
    # imsave('lab.png', lab)
    l = lab[:,:,0]
    print(l.shape)
    ab = lab[:,:,1:] / 128
    print(ab.shape)
    imsave('L.png', l)
    imsave('ab.png', ab)

if __name__ == "__main__":

    # convert_to_lab('adjusted/mountain/00000375.jpg')

    category = "flower"
    existing_weights = None
    continue_training = False 
    batch_size = 25
    epochs = 500
    steps_per_epoch = 25

    # # if using existing weights:
    # str_num = "25"
    # existing_weights = "./models/beta/" + category + "/beta-weights-" + str_num + "./hdf5"
    # # if continuing training:
    # continue_training = True


    # model_by_category(category, existing_weights=existing_weights, continue_training=continue_training, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch)
    # epochs = ['1', '10', '50', '100', '150', '180', '200', '250', '300', '350', '400', '450', '500']
    epochs=['200']

    for epoch in epochs:
        print('------ PREDICTING FOR {} --------'.format(epoch))
        try:
            test_epoch(category, epoch)
        except:
            print('----- EPOCH {} DOES NOT EXIST --------'.format(epoch))
            continue
