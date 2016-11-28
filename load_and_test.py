from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
# Import data
from our_input_data import *
import tensorflow as tf

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
  saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3])
  sess = tf.InteractiveSession()

  # Code from multiple neural nets import
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  saver.restore(sess, "/tmp/model1.ckpt")
  print("Model restored.")
  #sess.run()

  m = data.test.num_examples
  test_acc = 0
  for i in range(0, m):
	test_acc += accuracy.eval(feed_dict={x:data.test.images[i:i+1], y_: data.test.labels[i:i+1]})
  test_acc /= m
  print("Testing Accuracy: %g" %test_acc)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()