from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Builds the model.

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

# How random are these batches?

train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(128, 128),  # all images will be resized to 128x128
        batch_size=32,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/valid',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(                    # Starts the training
        train_generator,
        samples_per_epoch=12284,        # Why entire training set per epoch?
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=1001)

model.save_weights('second_try.h5')  # always save your weights after training or during training