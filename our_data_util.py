"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

#Includes from download_data.py
import PIL
from scipy.ndimage import imread
import numpy as np
import os

# ****Code taken from TensorFlow Github *****
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
# *******************************************


def extract_images(folderPath):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    folderPath: A path to a folder containing the dataset of images
  Returns:
    data: A 4D unit8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  files = os.listdir(folderPath)
  if(len(files) == 0):
    raise ValueError('There were no pictures to be read')

  file = files[0]
  x = imread(folderPath + file, flatten=False, mode='RGB')
  imageL = x.shape[0]
  imageW = x.shape[1]
  numImages = len(files)
  data = np.zeros((numImages, imageL, imageW, 3))   # Channels = 3

  for i in range(0, numImages):
    fileName = files[i]
    x = imread(folderPath + fileName, flatten=False, mode='RGB')  # returns ndarray (L x W x 3)
    data[i,:,:,:] = x
  
  # The data is now a 4D numpy array [index, y, x, depth]
  return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(nameOfCSVFile, one_hot=False, num_classes=8):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    nameOfCSVFile: Name of CSV file containing the labels.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D unit8 numpy array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  f = open(nameOfCSVFile)
  labels = [(line.split(','))[1] for line in f]
  labels = np.asarray(labels)[1:].astype(np.uint8) # starts at 1
  if one_hot:
    return dense_to_one_hot(labels, num_classes)
  return labels



class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        #assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=1000):
  if fake_data:
    raise ValueError("We're not dealing with Fake Data\n")

  cwd = os.getcwd()
  folderPath = cwd + '/411a3/train/'
  train_images = extract_images(folderPath)
  train_file = cwd + '/411a3/train.csv'
  train_labels = extract_labels(train_file, one_hot=one_hot)
  total_size = len(train_images)
  test_size = 1000
  train_size = total_size - validation_size - test_size

  test_images = None
  test_labels = None

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[train_size:train_size + validation_size - 1]
  validation_labels = train_labels[train_size:train_size + validation_size - 1]
  test_images = train_images[train_size+validation_size:]
  test_labels = train_labels[train_size+validation_size:]
  train_images = train_images[:train_size]
  train_labels = train_labels[:train_size]

  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_images,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape)
  test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test)

