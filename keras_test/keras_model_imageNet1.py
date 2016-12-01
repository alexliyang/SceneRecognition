'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
'''
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

# path to the model weights file.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'

train_data_dir = 'data/train'
validation_data_dir = 'data/valid'
nb_train_samples = 11418
nb_validation_samples = 1867
nb_epoch = 50

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)

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

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    #assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')


    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
 
    train_generator = train_datagen.flow_from_directory(
            'data/train',  # this is the target directory
            target_size=(224, 224),  # all images will be resized to 128x128
            batch_size=16,
            class_mode='categorical',
            shuffle=False)  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            'data/valid',
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            shuffle=False)
    print('Finished the model generators')

    # In order to use this code, I'll probably have to modify this predict_generator
    #bottleneck_features_validation = model.predict_generator(validation_generator, nb_valid_samples)
    
    #bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples)
    bottleneck_features_validation = model.predict_generator(validation_generator, 1867)
    print('done predicting the train generator')
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    print('Saved the validation predict_generator outputs')

    bottleneck_features_train = model.predict_generator(train_generator, 11418)
    print('done predicting the valid generator')
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    print('Saved the train predict_generator outputs')

def train_top_model():
    print("Loading prevously generated data from VGG-16...")
    # Load the output of our model when we it put through the previously trained layers of VGG
    # We simply use this as data to train our fully-connected layers.
    train_data = np.load(open('bottleneck_features_train.npy'))
    validation_data = np.load(open('bottleneck_features_validation.npy'))
    print('the shape of training features is ', train_data.shape)
    print('the shape of validation features is ', validation_data.shape)
    
    # Create two arrays for the validation and training labels based on how many images we have in these folders.
    cwd = os.getcwd()
    folder_names = ['1-structures/','2-indoor/','3-people/','4-animals/','5-plantlife/','6-food/','7-car/','8-sea/']
    train_labels = np.array([])
    validation_labels = np.array([])
    for i in range(0,8):
        folder_path = cwd + '/data/train/' + folder_names[i]
        files = os.listdir(folder_path)
        np.append(train_labels, [i] * len(files))
    for i in range(0,8):
        folder_path = cwd + '/data/valid/' + folder_names[i]
        files = os.listdir(folder_path)
        np.append(validation_labels, [i] * len(files))

    train_labels_cat = to_categorical(train_labels)
    validation_labels_cat = to_categorical(validation_labels)
    print('the shape of training labels is ', train_labels_cat.shape)
    print('the shape of validation labels is ', validation_labels.shape)

    print("Data and Labels Loaded")

    # Create model of the fully-connected layers
    model = Sequential()
    model.add(Flatten(input_shape=(512, 4, 4)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='softmax')) # Was sigmoid
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #was sparce_cat...

    print('Training the model now...')
    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=50,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)  # top_model = FC weights.

if __name__ == "__main__":
    #save_bottlebeck_features()  # Don't we only need to run this once!?
    train_top_model()

