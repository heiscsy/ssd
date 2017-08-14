from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


class SSD:
  def __init__(self):
    self.create_network()

  def creat_network(self):
    with tf.variable_scope("ssd")