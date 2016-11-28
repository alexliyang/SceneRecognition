# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
# Import data
from our_input_data import *
import tensorflow as tf
FLAGS = None
def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')
def main(_):
  mnist = read_data_sets(FLAGS.data_dir, one_hot=True, dtype=dtypes.float32)
  # Create the model
  x = tf.placeholder(tf.float32, [None, 49152])
  W = tf.Variable(tf.zeros([49152, 8]))
  b = tf.Variable(tf.zeros([8]))
  y = tf.matmul(x, W) + b
  # Define the first convolutional layer
  W_conv1 = weight_variable([5, 5, 3, 16])
  b_conv1 = bias_variable([16])
  x_image = tf.reshape(x, [-1,128,128,3])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  
  # Define the second convolutional layer
  W_conv2 = weight_variable([5, 5, 16, 32])
  b_conv2 = bias_variable([32])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  # Define a fully connected layer

  # Define the second convolutional layer
  W_conv3 = weight_variable([5, 5, 32, 32])
  b_conv3 = bias_variable([32])
  h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
  h_pool3 = max_pool_2x2(h_conv3)
  # Define a fully connected layer


  W_fc1 = weight_variable([5 * 5 * 32, 512])
  b_fc1 = bias_variable([512])
  h_pool3_flat = tf.reshape(h_pool3, [-1, 5*5*32])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
  # Dropout 
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  # Readout layer
  W_fc2 = weight_variable([512, 8])
  b_fc2 = bias_variable([8])
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 8])
  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  
  # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  sess = tf.InteractiveSession()
  # # Train
  # tf.initialize_all_variables().run()
  # for _ in range(1000):
  #   batch_xs, batch_ys = mnist.train.next_batch(100)
  #   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # # Test trained model
  # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
  #                                     y_: mnist.test.labels}))
  # Code from multiple neural nets import 
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess.run(tf.initialize_all_variables())

  test_length = 5000
  
  # Arrays for Statistics
  train_ce_list = []
  train_acc_list = []

  # Shortened from 20000 to 1000 for now 
  for i in range(test_length):
    batch = mnist.train.next_batch(50)
    #print(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)))
    acc = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    ce = cross_entropy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
    train_ce_list.append((i,ce))
    train_acc_list.append((i,acc))

    if i%100 == 0:
      print("step %d,\tacc: %g \tce: %g"%(i, acc, ce))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  print("test accuracy %g"%accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

  stats = {'train_acc' : train_acc_list,
           'train_ce' : train_ce_list}
  identifier = 0
  stats_name = 'cnn_stats' + str(identifier) + '.npz'
  print('Writing to ' + stats_name)
  np.savez_compressed(stats_name, **stats)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()