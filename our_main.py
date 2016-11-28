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


def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)
def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')
def norm(x):
  return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def main(_):
  data = read_data_sets(FLAGS.data_dir, one_hot=True, dtype=dtypes.float32)
  # Create the model
  learning_rate = tf.placeholder(tf.float32, shape=[])
  x = tf.placeholder(tf.float32, [None, 49152])
  W = tf.Variable(tf.zeros([49152, 8]))
  b = tf.Variable(tf.zeros([8]))
  y = tf.matmul(x, W) + b

  # Conv Layer 1
  W_conv1 = weight_variable([5, 5, 3, 64], name="W_conv1")
  b_conv1 = bias_variable([64], name="b_conv1")
  x_image = tf.reshape(x, [-1,128,128,3])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling Layer 1
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

  # Normalized Layer 1
  h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
  
  # Conv Layer 2
  W_conv2 = weight_variable([5, 5, 64, 64], name="W_conv2")
  b_conv2 = bias_variable([64], name="b_conv2")
  h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2) + b_conv2)

  # Normalized Layer 2
  h_norm2 = norm(h_conv2)

  # Pooling Layer 2
  h_pool2 = tf.nn.max_pool(h_norm2, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

  dim = 1
  for d in h_pool2.get_shape()[1:].as_list():
    dim *= d
  #h_pool2_flat = tf.reshape(h_pool2, [FLAGS.batch_size, dim])

  # FC Layer 1
  W_fc1 = weight_variable([dim, 384], name="W_fc1")
  b_fc1 = bias_variable([384], name="b_fc1")
  h_pool2_flat = tf.reshape(h_pool2, [-1, dim])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # FC Layer 2
  W_fc2 = weight_variable([384, 192], name="W_fc2")
  b_fc2 = bias_variable([192], name="b_fc2")
  h_fc1_flat = tf.reshape(h_fc1, [-1, 384])
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc2) + b_fc2)

  # Dropout 
  # keep_prob = tf.placeholder(tf.float32)
  # h_fc2_drop = tf.nn.dropout(h_fc1, keep_prob)
  # Readout layer
  W_fc3 = weight_variable([192, 8], name="W_fc3")
  b_fc3 = bias_variable([8], name="b_fc3")
  y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 8])
  init_op = tf.initialize_all_variables()
  saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3])
  sess = tf.InteractiveSession()

  # Code from multiple neural nets import
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess.run(init_op)

  test_length = 10
  
  # Arrays for Statistics
  train_ce_list = []
  train_acc_list = []
  lr = 0.1
  decay_rate = 0.1
  decay_iters = 350
  # Shortened from 20000 to 1000 for now 
  for i in range(test_length):
    batch = data.train.next_batch(50)
    #print(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)))
    acc = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    ce = cross_entropy.eval(feed_dict={x:batch[0], y_:batch[1]})
    train_ce_list.append((i,ce))
    train_acc_list.append((i,acc))

    if i%100 == 0:
      print("step %d,\tacc: %g \tce: %g"%(i, acc, ce))

    if i%decay_iters == 0:
      lr = lr * decay_rate

    train_step.run(feed_dict={x: batch[0], y_: batch[1], learning_rate: lr})

  # Just so we can keep track of different models / statistics
  identifier = 1

  # Save the Model to Memory
  save_path = saver.save(sess, "/tmp/model" + str(identifier) + ".ckpt")
  print("Model saved in file: %s" % save_path)

  stats = {'train_acc' : train_acc_list,
           'train_ce' : train_ce_list}

  stats_name = 'cnn_stats' + str(identifier) + '.npz'
  print('Writing to ' + stats_name)
  np.savez_compressed(stats_name, **stats)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()