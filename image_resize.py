import os, sys
import Image
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.misc import imsave

if __name__ == "__main__":	
	new_size = 224, 224
	for infile in sys.argv[1:]:
	    image = imread(infile, flatten=False, mode='RGB')
	    image = imresize(image,(224,224))
	    imsave('test.jpg', image)