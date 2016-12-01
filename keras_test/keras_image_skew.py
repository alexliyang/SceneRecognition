from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import PIL
import os

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


for i in range (2,7001):
	file_name = str(i).zfill(5) + '.jpg'

	cwd = os.getcwd()
	file_path = cwd + '/411a3/train/' + file_name
	img = load_img(file_path)  # this is a PIL image
	x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `preview/` directory
	j = 0
	## As is this only made 10000 images because of how it's indexed. I need to change the save prefix to 
	## Vary with i so that we get 20 skewed images per training example 
	for batch in datagen.flow(x, batch_size=1,
	                          save_to_dir='skewed_train', save_prefix='train', save_format='jpg'):
	    j += 1
	    if j > 20:
	        break  # otherwise the generator would loop indefinitely