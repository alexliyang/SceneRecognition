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

    model.load_weights(fc_weights_path, by_name=True)
    
    return model

if __name__ == "__main__":
    vgg_path = 'vgg16_weights.h5'
    fc_path = 'bottleneck_fc_model_2.h5'
    cwd = os.getcwd()
    small_path = cwd + '/data/test-1/'
    big_path = cwd + '/data/test_128-1/'
    num_small = 970
    num_big = 2000

    # Load VGG16 weights and trained FC weights into the model
    model = imageNet_model(vgg_path, fc_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    predictions = np.zeros((num_small + num_big, 2))

    test_datagen = ImageDataGenerator(rescale=1./255)
    small_generator = test_datagen.flow_from_directory(
                small_path,
                target_size=(224,224),
                batch_size=16,
                class_mode='categorical',
                shuffle=False)
    big_generator = test_datagen.flow_from_directory(
                big_path,
                target_size=(224,224),
                batch_size=16,
                class_mode='categorical',
                shuffle=False)
    small_output = model.predict_generator(small_generator,num_small)
    big_output = model.predict_generator(big_generator,num_big)

    for i in range(0, num_small):
        predict = small_output[i,:]
        predict = np.argmax(predict)
        predictions[i,:] = [int(i + 1), int(predict + 1)]
    for i in range(num_small, num_small + num_big):
        predict = big_output[i,:]
        predict = np.argmax(predict)
        predictions[i,:] = [int(i + 1), int(predict + 1)]        

    identifier = 3
    p_file = 'predictions' + str(identifier) + '.csv'
    np.savetxt(p_file, predictions, fmt='%d', delimiter=",")
    with open(p_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('Id,Prediction' + '\n' + content)
