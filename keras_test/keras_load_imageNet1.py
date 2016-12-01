from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import PIL
from scipy.ndimage import imread
import numpy as np
import os
import h5py


def imageNet_model(weights_path=None):
    # build the VGG16 network
    img_width, img_height = 128, 128
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten(input_shape=(512, 4, 4)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='softmax')) # Was sigmoid

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    files = None
    #cwd = os.getcwd()  #Took out the cwd for now 
    myPath = '/Users/sam/Google Drive/School/2016-17/CSC411/a3/keras_test/data/test/'
    files = os.listdir(myPath)
    #print(files)
    if(len(files) == 0):
        raise ValueError('There were no pictures to be read')

    file = files[0]
    x = imread(myPath + file, flatten=False, mode='RGB')
    imageL = x.shape[0]
    imageW = x.shape[1]
    numImages = len(files)
    data = np.zeros((numImages, imageL, imageW, 3))     # Channels = 3

    for i in range(0, numImages): 
        fileName = files[i]
        x = imread(myPath + fileName, flatten=False, mode='RGB')    # returns ndarray (L x W x 3)
        data[i,:,:,:] = x

    # Get all the images and store them in the data array 

    #np.save('test_mini_data', data)
    #currently [n, 128, 128, 3] we want [n, 3, 128, 128]
    np.transpose(data, [0,3,1,2])    

    model = imageNet_model('bottleneck_fc_model.h5')  
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    for i in range(0, numImages):
        #image = np.resize(data[i], [0, data[i].shape[0], data[i].shape[1], data[i].shape[2]])
        image = np.resize(data[i], [1, 3, 128, 128])
        out = model.predict(image)
        print np.argmax(out)

