from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import InputLayer, Conv2D, UpSampling2D, Input, RepeatVector, Reshape, concatenate
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb, gray2rgb, rgb2gray
from skimage.transform import resize
import tensorflow as tf
import numpy as np
import os

# use an encoder - decoder method 
# reference paper: https://arxiv.org/pdf/1712.03400.pdf

class InceptionModel:
    '''
    Full encoder - decoder + inception model
    '''
    def __init__(self, input_shape):
        '''
        Uses Keras Functional API to create model
        '''
        # create encoder
        encoder_inpt = Input(shape=(256, 256, 1, ))
        encoder = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')(encoder_inpt)
        encoder = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(encoder)
        encoder = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same')(encoder)
        encoder = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same')(encoder)
        encoder = Conv2D(256, (3, 3), strides=2, activation='relu', padding='same')(encoder)
        encoder = Conv2D(512, (3, 3), strides=1, activation='relu', padding='same')(encoder)
        encoder = Conv2D(512, (3, 3), strides=1, activation='relu', padding='same')(encoder)
        encoder = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same')(encoder)

        # create inception embed layer
        inception_inpt = Input(shape=(1000, ))

        # fusion 
        fusion = RepeatVector((256 * 256)/(8 ** 2))(inception_inpt)
        fusion = Reshape([32, 32, 1000])(fusion)
        fusion = concatenate([encoder, fusion])
        fusion = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same')(fusion)

        # create decoder
        decoder = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(fusion)
        decoder = UpSampling2D((2, 2))(decoder)
        decoder = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(decoder)
        decoder = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(decoder)
        decoder = UpSampling2D((2, 2))(decoder)
        decoder = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(decoder)
        decoder = Conv2D(2, (3, 3), strides=1, activation='relu', padding='same')(decoder)
        decoder = UpSampling2D((2, 2))(decoder)

        model = Model(inputs=[encoder, inception_inpt], outputs=[decoder])

        self.model = model

        # inception_inpt = Input(shape=(299, 299, 1))
        # inception = self.load_inception_res_net()

    def load_inception_res_net(self):
        '''
        Load inception resnet with pretrained weights for 
        high level feature extraction
        '''
        inception = InceptionResNetV2(weights=None, include_top=True)
        inception.load_weights('local/data/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
        inception.graph = tf.get_default_graph()
        return inception

    def create_inception_embedding(self, gray):
        '''
        Pass image through inception and obtain high level 
        features before softmax
        '''
        inception = self.load_inception_res_net()

        gray_resized = []
        for i in gray:
            i = resize(i, (299, 299, 3), mode='constant')
            gray_resized.append(i)
        gray_resized = np.array(gray_resized)
        gray_resized = preprocess_input(gray_resized)
        with inception.graph.as_default():
            embed = inception.predict(gray_resized)
        return embed

    def get_data_generator(self, shear_range=0.4, zoom_range=0.4, horizontal_flip=True, rotation_range=40):
        '''
        data generator
        '''
        return ImageDataGenerator(
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            rotation_range=rotation_range)
    
    def generate_next_batch(self, generator, training_data, batch_size):
        # batch_size = 20
        for batch in generator.flow(training_data, batch_size=batch_size):
            gray = gray2rgb(rgb2gray(batch))
            embed = self.create_inception_embedding(gray)
            lab_batch = rgb2lab(batch)
            x_batch = lab_batch[:,:,:,0]
            x_batch = x_batch.reshape(x_batch.shape+(1,))
            y_batch = lab_batch[:,:,:,1:] / 128
            yield ([x_batch, embed, y_batch])

    def train(self, batch_size, epochs, training_data, steps):
        '''
        Train Model
        '''
        generator = self.get_data_generator()
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit_generator(self.generate_next_batch(generator, training_data, batch_size), epochs=epochs, steps_per_epoch=steps)

    def predict(self, validation_data, category):
        '''
        Make predictions
        '''
        test = gray2rgb(rgb2gray(validation_data))
        test_embed = self.create_inception_embedding(validation_data)
        test = rgb2lab(test)[:,:,:,0]
        test = test.reshape(test.shape+(1,))

        # test
        output = self.model.predict([test, test_embed])
        output = output * 128

        for i in range(len(output)):
            cur = np.zeros((256, 256, 3))
            cur[:,:,0] = test[i][:,:,0]
            cur[:,:,1:] = output[i]

            imsave("./testset/" + category + "/predicted/" + image_names[i], lab2rgb(img))



def get_image_data(category, split=0.8):
    data = []
    all_images = os.listdir("./adjusted/" + category)
    for name in all_images:
        data.append(imread("./adjusted/" + category + "/" + name))
    data = np.array(data, dtype=float)
    training_data = data[int((1 - split) * len(data)):]
    training_data = 1.0/255 * training_data
    validation_data = data[:int((1 - split) * len(data))]
    validation_data = 1.0/255 * validation_data
    return (training_data, validation_data)