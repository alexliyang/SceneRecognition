from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import PIL
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np
import os
import h5py

def imageNet_model(vgg_weights_path=None, fc_weights_path=None):
    """
        This function takes in the path two two .h5 files which contain the weights
        which were previously trained.
        @arguments:
        vgg_weights_path: path to the VGG-16 weights previously trained for us.
        fc_weights_path: path to the fully-connected weights which we trained ourselves.
    """
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))

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

    vgg_layers = len(model.layers)

    model.add(Flatten())
    model.add(Dense(256, activation='relu', name='fc_1'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax', name='fc_2')) # Was sigmoid

    total_layers = len(model.layers)
    fc_layers = total_layers - vgg_layers
    print(vgg_layers)
    print(fc_layers)

    # Load the VGG weights into the model.
    f = h5py.File(vgg_weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= vgg_layers:
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()

    # # Load the FC weights into the model.
    # f = h5py.File(fc_weights_path)
    # for k in range(f.attrs['nb_layers']):
    #     g = f['layer_{}'.format(k)]
    #     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    #     model.layers[k + vgg_layers].set_weights(weights)
    # f.close()
    model.load_weights(fc_weights_path, by_name=True)


    return model

if __name__ == "__main__":
    vgg_path = 'vgg16_weights.h5'
    fc_path = 'bottleneck_fc_model_2.h5'
    cwd = os.getcwd()
    #myPath = '/Users/sam/Google Drive/School/2016-17/CSC411/a3/keras_test/data/test/'
    # small_path = cwd + '/data/test/'
    # big_path = cwd + '/data/test_128/'
    small_path = cwd + '/data/prediction-valid/'
    big_path = cwd + '/data/prediction-valid/'
    
    # Load the test image data into a numpy array: data[m,x,y,c]
    files = os.listdir(big_path)
    num_big = len(files)
    files = os.listdir(small_path)
    if(len(files) == 0):
        raise ValueError('There were no pictures to be read')
    file = files[0]
    x = imread(small_path + file, flatten=False, mode='RGB')
    x = imresize(x, (224,224))
    imageL = x.shape[0]
    imageW = x.shape[1]
    num_small = len(files)

    data = np.zeros((num_small + num_big, imageL, imageW, 3))     # Channels = 3
    for i in range(0, num_small): 
        fileName = files[i]
        x = imread(small_path + fileName, flatten=False, mode='RGB')    # returns ndarray (L x W x 3)
        x = imresize(x, (224,224))
        data[i,:,:,:] = x
    files = os.listdir(big_path)
    for i in range(num_small, num_small + num_big):
        fileName = files[i - num_small]
        x = imread(big_path + fileName, flatten=False, mode='RGB')    # returns ndarray (L x W x 3)
        x = imresize(x, (224,224))
        data[i,:,:,:] = x  

    data = data.transpose(0,3,1,2)       # [n, 224, 224, 3] --> [n, 3, 224, 224]

    # Load VGG16 weights and trained FC weights into the model
    model = imageNet_model(vgg_path, fc_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    identifier = 1
    predictions = np.zeros((num_small + num_big, 2))
    p_file = 'predictions' + str(identifier) + '.csv'
    for i in range(0, num_small + num_big):
        image = data[i,:,:,:]
        image = image.reshape([1, 3, 224, 224])
        out = model.predict(image)
        predict = np.argmax(out)
        print(predict)
        predictions[i,:] = [int(i + 1), int(predict + 1)]

    np.savetxt(p_file, predictions, fmt='%d', delimiter=",")
    with open(p_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('Id,Prediction' + '\n' + content)
