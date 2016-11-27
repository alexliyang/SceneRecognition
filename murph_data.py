from scipy.misc import imread
import cPickle as pickle
import os
import numpy as np
import gzip

x = []
for filename in os.listdir('./train/'):
	img = imread('./train/'+filename)
	x.append(img)
x = np.asarray(x).astype(np.uint8)
with open('train.csv') as f:
	y = [(line.split(','))[1] for line in f]
y = np.asarray(y)[1:].astype(np.int32) # starts at 1
with gzip.open('test_data.bin', 'wb') as f:
	pickle.dump([x, y], f)