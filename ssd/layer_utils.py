from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def conv2d(net, filter_size, in_channel, out_channel, namescope, reuse, is_training):
  with tf.contrib.framework.arg_scope(resue=reuse, is_training = is_training):
    with tf.variable_scope(namescope):
      w = tf.get_variable(
        name="W", 
        shape = [filter_size, filter_size, in_channel, out_channel],
        )
      net = tf.nn.conv2d(net, filter=w, strides=1, padding="SAME")
      net = tf.contrib.layers.batch_norm(net,scale=True)
      net = tf.nn.relu(net)
      return net

def fc(net, in_channel, out_channel, namescope, reuse, is_training):
  with tf.contrib.framework.arg_scope(reuse=reuse, is_training=is_training):
    with tf.variable_scope(namescope):
      w = tf.get_variable(
        name = "W"
        shape = [in_channel, out_channel]
        )
      net = tf.contrib.layers.batch_norm(tf.matmul(net, w), scale=True)
      return net
