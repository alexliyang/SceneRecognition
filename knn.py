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
from keras.utils.np_utils import to_categorical
from keras_test.keras_load_imageNet1 import imageNet_model

def l2_distance(a, b):
    """Computes the Euclidean distance matrix between a and b.

    Inputs:
        A: D x M array.
        B: D x N array.

    Returns:
        E: M x N Euclidean distances between vectors in A and B.

    Author   : Roland Bunschoten
               University of Amsterdam
               Intelligent Autonomous Systems (IAS) group
               Kruislaan 403  1098 SJ Amsterdam
               tel.(+31)20-5257524
               bunschot@wins.uva.nl
    Last Rev : Wed Oct 20 08:58:08 MET DST 1999
    Tested   : PC Matlab v5.2 and Solaris Matlab v5.3

    Copyright notice: You are free to modify, extend and distribute 
       this code granted that the author of the original code is 
       mentioned as the original author of the code.

    Fixed by JBT (3/18/00) to work for 1-dimensional vectors
    and to warn for imaginary numbers.  Also ensures that 
    output is all real, and allows the option of forcing diagonals to
    be zero.  

    Basic functionality ported to Python 2.7 by JCS (9/21/2013).
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should be of same dimensionality")

    aa = np.sum(a**2, axis=0)
    bb = np.sum(b**2, axis=0)
    ab = np.dot(a.T, b)

    return np.sqrt(aa[:, np.newaxis] + bb[np.newaxis, :] - 2*ab)

def run_knn(k, train_data, train_labels, valid_data):
    valid = valid_data.transpose()
    train = train_data.transpose()
    d = l2_distance(valid, train)
    sorted = d.argsort(axis=1)
    nearest = sorted[:, :k]
    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]
    # note this only works for binary labels
    valid_lbls = np.zeros(len(valid_labels))
    for i in range(0,len(valid_labels)):
        x = valid_labels[i].astype(int)
        counts = np.bincount(x)
        valid_lbls[i] = np.argmax(counts)
    return valid_lbls

if __name__ == "__main__":
    
    # Get data and flatten it out for use in knn.
    train_data = np.load(open('keras_test/bottleneck_features_train_3.npy'))
    valid_data = np.load(open('keras_test/bottleneck_features_validation_3.npy'))
    small_test_data = np.load(open('keras_test/knn_features_test_small.npy'))
    large_test_data = np.load(open('keras_test/knn_features_test_large.npy'))
    test_data = np.append(small_test_data, large_test_data,axis=0)
    test_shape = test_data.shape
    train_shape = train_data.shape
    valid_shape = valid_data.shape
    num_test = test_shape[0]
    num_train = train_shape[0]
    num_valid = valid_shape[0]
    test_data = test_data.reshape(num_test, test_shape[1]*test_shape[2]*test_shape[3])
    train_data = train_data.reshape(num_train, train_shape[1]*train_shape[2]*train_shape[3])
    valid_data = valid_data.reshape(num_valid, valid_shape[1]*valid_shape[2]*valid_shape[3])

    # Create two arrays for the validation and training labels based on how many images we have in these folders.
    cwd = os.getcwd()
    folder_names = ['1-structures/','2-indoor/','3-people/','4-animals/','5-plantlife/','6-food/','7-car/','8-sea/']
    train_labels = np.array([0] * 1816 + [1] * 1686 + [2] * 1848 + [3] * 1668 + [4] * 1432 + [5] * 1041 + [6] * 1024 + [7] * 903)
    valid_labels = np.array([0] * 298 + [1] * 265 + [2] * 214 + [3] * 229 + [4] * 239 + [5] * 214 + [6] * 205 + [7] * 203)
    train_labels = to_categorical(train_labels)
    valid_labels = to_categorical(valid_labels)

    # Reformat Labels
    train_lbl = np.zeros(num_train)
    for i in range(0,num_train):
        train_lbl[i] = np.argmax(train_labels[i])

    # Run KNN
    predict_valid_labels = run_knn(10,train_data,train_lbl,valid_data)
    predict_test_labels = run_knn(10,train_data,train_lbl,test_data)

    # Test Accuracy of KNN Method
    valid_acc = 0
    for i in range(0,num_valid):
        correct = float(np.argmax(valid_labels[i]))
        predict = float(predict_valid_labels[i])
        if correct == predict:
            valid_acc += 1
    valid_acc /= float(num_valid)
    valid_acc *= 100
    print("Validation Accuracy: %g" %valid_acc)

    predictions = np.zeros((num_test,2))
    for i in range(0,num_test):
        predictions[i,:] = [i+1,int(predict_test_labels[i])+1]

    p_file = 'predictions_knn' + '.csv'
    np.savetxt(p_file, predictions, fmt='%d', delimiter=",")
    with open(p_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('Id,Prediction' + '\n' + content)
