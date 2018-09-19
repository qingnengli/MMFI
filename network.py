"""Networks for MNIST example using TFGAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from utils import group_conv
from utils import channel_shuffle
layers = tf.contrib.layers

def unconditional_generator(img, weight_decay=2.5e-5,keep_prob=0.4, is_training=True):
  """Generator to produce unconditional MNIST images.

  Args:
    noise: A single Tensor representing noise.
    keep_prob: The keep probability of dropout layer.
    weight_decay: The value of the l2 weight decay.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A generated image in the range [-1, 1].
  """
  _,h,w,c = img.get_shape().as_list()
  repeat = int(np.log2(h//16))
  with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d_transpose],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.contrib.framework.arg_scope(
        [layers.batch_norm], is_training=is_training):
      net = layers.flatten(img)
      net = layers.fully_connected(net,100)
      net = layers.fully_connected(net, 4 * 4* 64)
      net = tf.reshape(net, [-1, 4, 4, 64])
      net = layers.dropout(net,keep_prob= keep_prob,is_training=is_training)
      net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      net = layers.repeat(net, repeat, layers.conv2d_transpose, 32, [4, 4], stride=2)
      net = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)
      return (net+1)/2

_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)
def unconditional_discriminator(img, unused_conditioning, weight_decay=2.5e-5):
  """Discriminator network on unconditional MNIST digits.

  Args:
    img: Real or generated MNIST digits. Should be in the range [-1, 1].
    unused_conditioning: The TFGAN API can help with conditional GANs, which
      would require extra `condition` information to both the generator and the
      discriminator. Since this example is not conditional, we do not use this
      argument.
    weight_decay: The L2 weight decay.

  Returns:
    Logits for the probability that the image is real.
  """
  with tf.contrib.framework.arg_scope(
      [layers.conv2d, layers.fully_connected],
      activation_fn=_leaky_relu, normalizer_fn=None,
      weights_regularizer=layers.l2_regularizer(weight_decay),
      biases_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.conv2d(img, 64, [4, 4], stride=2)
    net = layers.conv2d(net, 64, [4, 4], stride=2)
    net = layers.conv2d(net, 64, [4, 4], stride=2)
    net = layers.conv2d(net, 64, [4, 4], stride=2)
    net = layers.flatten(net)
    net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
  return layers.linear(net, 1)

def SRMatrix(img,weight_decay=2.5e-5,keep_prob =0.5, is_training=True,use_aux = False):
  b,h,w,c = img.get_shape().as_list()
  with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d_transpose, layers.conv2d],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.contrib.framework.arg_scope(
        [layers.batch_norm], is_training=is_training):
      encoder1 = layers.conv2d(img, 64, [4, 4], stride=2)
      encoder2 = layers.conv2d(encoder1, 64, [4, 4], stride=2)
      encoder3 = layers.conv2d(encoder2, 64, [4, 4], stride=2)

      net = layers.conv2d(encoder3, 1, [1, 1])
      net = layers.flatten(net)
      net = layers.fully_connected(net, int(h*w//64), normalizer_fn=layers.layer_norm)
      net = tf.reshape(net,[-1,h//8,w//8,1])
      Auxlogit_1 = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      decoder3 = layers.conv2d(encoder3, 1, [4, 4])
      net = layers.conv2d_transpose(net+decoder3, 64, [4, 4], stride=2)
      net = layers.conv2d(net, 1, [4, 4])
      Auxlogit_2 = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      decoder2 = layers.conv2d(encoder2, 1, [4, 4])
      net = layers.conv2d_transpose(net+decoder2, 64, [4, 4], stride=2)
      net = layers.conv2d(net, 1, [4, 4])
      Auxlogit_3 = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      decoder1 = layers.conv2d(encoder1, 1, [4, 4])
      net = layers.conv2d_transpose(net+decoder1, 64, [4, 4], stride=2)
      net = layers.conv2d(net, 1, [4, 4])

      net = layers.conv2d(net, 1, [4, 4])
      Logit = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

      Logit = (Logit + 1)/2
      Auxlogit_1 = (Auxlogit_1 + 1) / 2
      Auxlogit_2 = (Auxlogit_2 + 1) / 2
      Auxlogit_3 = (Auxlogit_3 + 1) / 2

      if use_aux:
        return [Logit, [Auxlogit_1,Auxlogit_2,Auxlogit_3]]
      else:
        return Logit
