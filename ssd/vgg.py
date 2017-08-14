from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import layer_utils


class VGG:
  reuse = None
  is_training = True
  endpoints = {}
  def __init__(self, reuse, is_training, input_tensor=None):
    self.reuse= reuse
    self.is_training = is_training
    if input_tensor is None:
      input_tensor = tf.placeholder(types=tf.float32, shape=[None, 224, 224, 3])
    
    self.endpoints=self.create_network(input_tensor)

  def creat_network(self, net):
    with tf.variable_scope("VGG"):
      # conv_1 224->112
      net = layer_utils.conv2d(net, 3, 3, 64, "conv_1_1", self.reuse, self.is_training)
      net = layer_utils.conv2d(net, 3, 64, 64, "conv_1_2", self.reuse, self.is_training)
      net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

      #conv_2 112->56
      net = layer_utils.conv2d(net, 3, 64, 128, "conv_2_1", self.reuse, self.is_training)
      net = layer_utils.conv2d(net, 3, 128, 128, "conv_2_2", self.reuse, self.is_training)
      net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

      #conv_3 56->28
      net = layer_utils.conv2d(net, 3, 128, 256, "conv_3_1", self.reuse, self.is_training)
      net = layer_utils.conv2d(net, 3, 256, 256, "conv_3_2", self.reuse, self.is_training)
      net = layer_utils.conv2d(net, 3, 256, 256, "conv_3_3", self.reuse, self.is_training)
      net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

      #conv_4 28->14
      net = layer_utils.conv2d(net, 3, 256, 512, "conv_4_1", self.reuse, self.is_training)
      net = layer_utils.conv2d(net, 3, 512, 512, "conv_4_2", self.reuse, self.is_training)
      net = layer_utils.conv2d(net, 3, 512, 512, "conv_4_3", self.reuse, self.is_training)
      net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
      endpoints["conv4"] = net

      #conv_5 14->7
      net = layer_utils.conv2d(net, 3, 512, 512, "conv_5_1", self.reuse, self.is_training)
      net = layer_utils.conv2d(net, 3, 512, 512, "conv_5_2", self.reuse, self.is_training)
      net = layer_utils.conv2d(net, 3, 512, 512, "conv_5_3", self.reuse, self.is_training)
      net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
      endpoints["conv5"] = net

      #fc
      net = tf.reshape(net, [-1, 7*7*512])
      net = layer_utils.fc(net, 7*7*512, 4096, "fc_1", self.reuse, self.is_training)
      net = tf.nn.relu(net)
      net = layer_utils.fc(net, 4096, 4096, "fc_2", self.reuse, self.is_training)
      net = tf.nn.relu(net)
      net = layer_utils.fc(net, 4096, 4096, "fc_3", self.reuse, self.is_training)
      endpoints["fc"] = net

      return endpoints

