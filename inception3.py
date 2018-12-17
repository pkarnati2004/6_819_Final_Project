from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import InputLayer, Conv2D, UpSampling2D, Input, RepeatVector, Reshape, concatenate
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
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

print('------ LOADING INCEPTION -----')
#Load weights
inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('local/data/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
inception.graph = tf.get_default_graph()

print('------ CREATING MODEL -----')

embed_input = Input(shape=(1000,))

#Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

#Fusion
fusion_output = RepeatVector(32 * 32)(embed_input) 
fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

#Decoder
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

print('------ FINISHED MODEL -----')

#Create embedding
def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True)

#Generate training data
batch_size = 20

def image_a_b_gen(Xtrain, batch_size):
    print('---- GENERATING BATCH -----')
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

#Train model      
# tensorboard = TensorBoard(log_dir="/output")

def get_images(category):
    # Get images
    print('------ GETTING IMAGES ------')
    X = []
    for filename in os.listdir("./adjusted/" + category):
        X.append(imread("./adjusted/" + category + "/" + filename))
    X = np.array(X, dtype=float)
    Xtrain = 1.0/255*X

    return Xtrain

def train_model(train):
    print('------ TRAINING ---------')
    # model = multi_gpu_model(model, gpus=4)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    filepath="models/inception_beach/inception-weights-{epoch:02d}.hdf5"
    # model.load_weights('./models/inception2/inception-weights-80.hdf5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period=1)
    model.fit_generator(image_a_b_gen(train, batch_size), epochs=500, steps_per_epoch=20, callbacks=[checkpoint])
    model.save_weights("./models/inception_beach/bestmodel.h5")

def predict(category, weights_path):
    model.load_weights(weights_path)

    #Make a prediction on the unseen images
    names = []
    test_x = []
    for filename in os.listdir("./testset/" + category + "/expected"):
        names.append(filename)
        test_x.append(imread("./testset/" + category + "/expected/" + filename))

    test_x = np.array(test_x, dtype=float)
    test_x = 1.0/255*test_x
    test_x = gray2rgb(rgb2gray(test_x))
    test_x_embed = create_inception_embedding(test_x)
    test_x = rgb2lab(test_x)[:,:,:,0]
    test_x = test_x.reshape(test_x.shape+(1,))

    print('-------- PREDICTING -------')
    # Test model
    output = model.predict([test_x, test_x_embed])
    output = output * 128

    print('SAVING')
    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = test_x[i][:,:,0]
        cur[:,:,1:] = output[i]
        filepath = "./testset/" + category + "/predicted3/" + names[i]
        print(filepath)
        imsave(filepath, lab2rgb(cur))


if __name__ == "__main__":
    category = 'landscape'

    trainx = get_images(category)
    # train_model(trainx)

    predict(category, 'models/inception_landscape/inception-weights-99.hdf5')
