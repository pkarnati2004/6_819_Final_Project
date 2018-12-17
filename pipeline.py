import classifier
import model
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb, gray2rgb, rgb2gray
from skimage.transform import resize
import numpy as np


if __name__ == "__main__":
    
    valid_paths = sorted(['autumn', 'beach', 'city', 'mountain', 'sunset', 'waterfall'])
    classifer_model = classifer.get_model(valid_paths)
    classifer_model.load_weights('models/classifier-2/classifier-weights-300.hdf5')

    colorizer = model.get_model()

    impath = 'testset/city/expected/00000029.jpg'

    data = imread(impath)
    data = np.array(data)
    data = data.reshape((1,) + data.shape)

    predicted_labels = classifer_model.predict(data, batch_size=1)

    print(predicted_labels)

    