from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import PIL
from scipy.ndimage import imread
import numpy as np
import os
import h5py


def first_model(weights_path=None):
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(3, 128, 128)))
    # now model.output_shape == (None, 32, 128, 128)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

    # the model so far outputs 3D feature maps (height, width, features)

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    files = None
    #cwd = os.getcwd()  #Took out the cwd for now 
    myPath = '/Users/sam/Google Drive/School/2016-17/CSC411/a3/keras_test/data/prediction-valid/'
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

    # for i in range(0, numImages): 
    #     fileName = files[i]
    #     x = imread(myPath + fileName, flatten=False, mode='RGB')    # returns ndarray (L x W x 3)
    #     data[i,:,:,:] = x
    for i in range(0, numImages): 
        fileName = files[i]
        img = PIL.Image.open(myPath + fileName)
        img = img.convert('RGB')
        img = img.resize((128, 128))
        data[i,:,:,:] = img


    # Get all the images and store them in the data array 

    #np.save('test_mini_data', data)
    #currently [n, 128, 128, 3] we want [n, 3, 128, 128]
    #np.transpose(data, [0,3,1,2])    

    model = first_model('second_try.h5')  
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    for i in range(0, numImages):
        fileName = files[i]
        # data[i] is (128, 128, 3)
        image = np.resize(data[i], [1, data[i].shape[0], data[i].shape[1], data[i].shape[2]])
        print(data[i].shape)
        #image = np.transpose(data[i], [1, 3, 128, 128])
        out = model.predict(image)
        print np.argmax(out)

