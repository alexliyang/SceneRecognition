import cPickle as pickle
import os
import numpy as np

(X_train, Y_train), X_test

(X_train, Y_train), X_test = pickle.dump(hello, open('vgg_data.pkl', 'rb'))


print('Size of X_Train is ', X_Train.shape)
print('X_Train is ', X_Train)

print('Size of X_Train is ', Y_Train.shape)
print('Y_Train is ', Y_Train)

print('Size of X_Train is ', X_Test.shape)
print('X_Test is ', X_Test)