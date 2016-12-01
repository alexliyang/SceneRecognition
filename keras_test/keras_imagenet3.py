from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg19 import VGG19
from keras.layers import Input

# Try to load in the base model of VGG19
input_tensor = Input(shape=(3, 128, 128))  # this assumes K.image_dim_ordering() == 'tf'

base_model = VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)
for layer in base_model.layers:
    layer.trainable = False
# x = base_model.output

# # Re-add our desired fully connected output layers
# x = Flatten(name='flatten')(x)
# x = Dense(16, activation='relu', name='fc1')(x)
# x = Dense(1, activation='softmax', name='predictions')(x)

# x = Flatten(name='flatten')(x)  # this converts our 3D feature maps to 1D feature vector
# x = Dense(64, activation='relu', name='fc1')(x)
# x = Dropout(0.5)
# x = Dense(8, activation='sigmoid', name='predictions')(x)

base_model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
base_model.add(Dense(64))
base_model.add(Activation('relu'))
base_model.add(Dropout(0.5))
base_model.add(Dense(8))
base_model.add(Activation('sigmoid'))

# Compile our model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('The model has compiled')
# Switch to training things 
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
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(128, 128),  # all images will be resized to 128x128
        batch_size=32,
        class_mode='sparse')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/valid',
        target_size=(128, 128),
        batch_size=32,
        class_mode='sparse')

model.fit_generator(
        train_generator,
        samples_per_epoch=5999,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=1001)

model.save_weights('third_try.h5')  # always save your weights after training or during training