from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.utils.training_utils import multi_gpu_model
from keras.layers import InputLayer, Conv2D, UpSampling2D, Flatten, BatchNormalization, Dropout, Dense
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb, gray2rgb, rgb2gray
from skimage.transform import resize
from sklearn.preprocessing import normalize
from collections import defaultdict
from ranutils import print_something
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import os
import random
import datetime

from keras import backend as K

# os.environ["CUDA_VISIBLE_DEVICES"]="2"

class DataEncoder:
    '''encode each category name into a one-hot vector encoding'''
    def __init__(self, categories):
       self.categories = categories

    def one_hot_index(self, category):
        return self.categories.index(category)

    def one_hot_decode(self, predicted_labels):
        return dict(zip(self.categories, predicted_labels))

    def one_hot_encode(self, category):
        encoding = np.zeros(len(self.categories))
        index = self.one_hot_index(category)
        encoding[index] = 1
        return encoding


class DataGenerator:
    ''' split into train, test, validate '''
    def __init__(self, paths):
        self.paths = paths
        self.train = defaultdict(list)
        self.test = defaultdict(list)
        self.validate = defaultdict(list)

        self.categories = []

        all_categories = sorted(self.paths) # sorted(os.listdir('./adjusted/'))

        for category in all_categories:
            all_images = os.listdir('./adjustedbw/' + category)
            self.categories.append(category)
            paths = []
            for filepath in all_images:
                path = './adjustedbw/' + category + '/' + filepath
            #     paths.append('./adjusted/' + category + '/' + filepath)
            # random.shuffle(paths)
            # self.train = paths[:int((0.7) * len(paths))]
            # self.validate = paths[int((0.7) * len(paths)):int((0.9) * len(paths))]
            # self.test = paths[int((0.9) * len(paths)):]
                if hash(path) % 10 < 7:
                    self.train[category].append(path)
                elif 7 <= hash(path) % 10 < 9:
                    self.validate[category].append(path)
                elif 9 == hash(path) % 10:
                    self.test[category].append(path)
                else:
                    raise Exception('no partition')

        self.partitions = {
            'train': self.train,
            'validate': self.validate,
            'test': self.test
        }

        self.encoder = DataEncoder(self.categories)

        self.image_data_gen = ImageDataGenerator(
            rotation_range = 15,
            width_shift_range = 0.15,
            height_shift_range = 0.15,
            shear_range = 0.15,
            zoom_range = 0.15,
            channel_shift_range = 1,
            horizontal_flip = True,
            vertical_flip = False
        )

    def _pair_generator(self, partition, augmented=True):
        while True:
            for category, paths in self.partitions[partition].items():
                random_path = random.choice(paths)
                data = imread(random_path)
                data = np.array(data, dtype=float)

                # # resize along 3 axes 
                # data_resized = []
                # for i in data:
                #     i = resize(i, (256, 256, 3), mode='constant')
                #     data_resized.append(i)
                # data_resized = np.array(data_resized)
                # data_resized = preprocess_input(data_resized)
                # data = data_resized

                data *= 255.0/data.max()
                # data = normalize(data, axis=1, norm='l1')
                data = np.repeat(data[:, :, np.newaxis], 3, axis=2)

                
                data = data.reshape((1,) + data.shape)

                encoding = self.encoder.one_hot_encode(category)
                if augmented:
                    augmented_data = next(self.image_data_gen.flow(data))[0].astype(np.uint8)
                    yield augmented_data, encoding
                else:
                    yield data, encoding

    def batch_generator(self, partition, batch_size, augmented=True):
        while True:
            data_gen = self._pair_generator(partition, augmented)
            data_batch, encoding_batch = zip(*[next(data_gen) for _ in range(batch_size)])
            data_batch = np.array(data_batch)
            encoding_batch = np.array(encoding_batch)
            yield data_batch, encoding_batch

    def get_data_split(self):
        return self.partitions

    def decode(self, predictions):
        return self.encoder.one_hot_decode(predictions)


img_shape = (256, 256)
input_shape = (*img_shape, 3)

def get_model(categories):
    model = InceptionResNetV2(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet'
    )

    if model.output.shape.ndims > 2:
        output = Flatten()(model.output)
    else:
        output = model.output

    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(len(categories), activation='softmax')(output)

    model = Model(model.input, output)

    for layer in model.layers:
        layer.trainable = False
    
    # model.summary(line_length=280)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def make_predictions(model, test, data_generator):
    all_predictions = {}
    categories = data_generator.categories
    errors = {}
    for category, paths in test.items():
        inc = 0
        print('----- CATEGORY {}-----'.format(category))
        predictions = []
        for p in paths:
            data = imread(p)
            data = np.array(data)
            # data = np.linalg.norm(data)
            data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
            data = data.reshape((1,) + data.shape)
            predicted_labels = model.predict(data, batch_size=1)

            # print(p, predicted_labels[0])
            zipped = list(zip(categories, predicted_labels[0]))
            sort = sorted(zipped,key=lambda x:(-x[1],x[0]))
            # print(sort)

            # label = np.argmax(predicted_labels[0])
            # labeled = categories[label]
            labels = sort[:3]
            # print(labels)
            top3 = [label[0] for label in labels]
            if category not in top3:
                inc += 1
            generated_data = data_generator.decode(predicted_labels[0])
            # print(label, categories[label])
            # print('generated data: ', generated_data)

            predictions.append((p, sort))
        print(category, inc, len(paths))

        inc /= len(paths)
        errors[category] = inc

        all_predictions[category] = predictions

    print(errors)
            
    with open("res.txt") as myfile:
        myfile.write("-------------------\n")
        myfile.write("-------------------\n")
        myfile.write("PREDICTIONS {}\n".format(datetime.datetime.now()))
        for category, predictions in all_predictions.items():
            myfile.write('CATEGORY: {}\n'.format(category))
            for impath, preds in predictions:
                myfile.write('impath: {}, predictions: {}\n'.format(impath, str(preds)))
        for categroy, inc in errors.items():
            myfile.write('CATEGORY: {}, ERROR: {}'.format(category, inc))



if __name__ == "__main__":
    batch_size = 25
    epochs = 500
    steps_per_epoch = 100

    valid_paths = sorted(['autumn', 'beach', 'city', 'mountain', 'sunset', 'waterfall'])

    print_something('GETTING MODEL')

    # categories = sorted(os.listdir('./adjusted/'))
    model = get_model(valid_paths) # get_model(categories)

    print_something('CREATING GENERATOR')

    data_generator = DataGenerator(valid_paths) # DataGenerator(categories)

    print_something('SPLITTING DATA')

    data_split = data_generator.get_data_split()

    print_something('STUFF')

    train = data_split['train']
    validation = data_split['validate']
    test = data_split['test']

    print_something('GETTING MODEL EPOCHS')

    filepath = "models/classifier-inception-2/classifier-weights-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period=50)


    # model.fit_generator(
    #     data_generator.batch_generator('train', batch_size=batch_size),
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     callbacks=[checkpoint],
    #     use_multiprocessing=True,
    #     workers=3
    # )

    print_something('LOADING WEIGHTS')

    model.load_weights('models/classifier-3/classifier-weights-150.hdf5')

    print_something('MAKING PREDICTIONS')

    # all_predictions = {}

    # for path in valid_paths:
    #     all_images = os.listdir('./testset/' + path + '/bw/')
    #     predictions = []
    #     for name in all_images:
    #         p = './testset/' + path + '/bw/' + name

    #         data = imread(p)
    #         data = np.array(data)
    #         # data = np.linalg.norm(data)
    #         data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    #         data = data.reshape((1,) + data.shape)
    #         predicted_labels = model.predict(data, batch_size=1)

    #         # print(p, predicted_labels[0])
    #         zipped = list(zip(valid_paths, predicted_labels[0]))
    #         sort = sorted(zipped,key=lambda x:(-x[1],x[0]))
    #         print(sort)

    #         label = np.argmax(predicted_labels[0])
    #         generated_data = data_generator.decode(predicted_labels[0])
    #         print(label, valid_paths[label])
    #         print('generated data: ', generated_data)

    #         predictions.append((p, sort))

    #     all_predictions[path] = predictions
    
    # with open("predictions.txt") as myfile:
    #     myfile.write("-------------------\n")
    #     myfile.write("-------------------\n")
    #     myfile.write("PREDICTIONS {}\n".format(datetime.datetime.now()))
    #     for category, predictions in all_predictions.items():
    #         myfile.write('CATEGORY: {}\n'.format(category))
    #         for impath, preds in predictions:
    #             myfile.write('impath: {}, predictions: {}\n'.format(impath, str(preds)))


    # KDSJKFJDSJFDSK

    make_predictions(model, test, data_generator)

    # categories = data_generator.categories

    # impath = './testset/mountain/bw/00000128.jpg'

    # data = imread(impath)
    # data = np.array(data)
    # data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    # data = data.reshape((1,) + data.shape)

    # predicted_labels = model.predict(data, batch_size=1)

    # zipped = list(zip(valid_paths, predicted_labels[0]))
    # sort = sorted(zipped,key=lambda x:(-x[1],x[0]))
    # print(impath)
    # print(sort)

    # print(predicted_labels)



    # validation_data=data_generator.batch_generator(
    #         'validate',
    #         batch_size=batch_size,
    #         augmented=False
    #     ),
    #     validation_steps=10,