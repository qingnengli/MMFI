"""Networks for MNIST example using TFGAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
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
  with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d_transpose],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.contrib.framework.arg_scope(
        [layers.batch_norm], is_training=is_training):
      net = layers.flatten(img)
      net = layers.fully_connected(net,64)
      net = layers.fully_connected(net, 1024)
      net = layers.fully_connected(net, 4 * 4* 128)
      net = tf.reshape(net, [-1, 4, 4, 128])
      net = layers.dropout(net,keep_prob= keep_prob,is_training=is_training)
      net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
      net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
      net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
      net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
      net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
      # Make sure that generator output is in the same range as `inputs`
      # ie [-1, 1].
      net = layers.conv2d(
          net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

      return net

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
    net = layers.conv2d(net, 128, [4, 4], stride=2)
    net = layers.conv2d(net, 128, [4, 4], stride=2)
    net = layers.conv2d(net, 128, [4, 4], stride=2)
    net = layers.flatten(net)
    net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)

  return layers.linear(net, 1)

def SRMatrix(img,weight_decay=2.5e-5,keep_prob=0.5, is_training=True):
  b,h,w,c = img.get_shape().as_list()
  with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d_transpose, layers.conv2d],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.contrib.framework.arg_scope(
        [layers.batch_norm], is_training=is_training):
      net = tf.image.resize_images(img, [32, 32], method=0)
      net = layers.flatten(net)
      net = layers.fully_connected(net, 32*32,normalizer_fn=layers.layer_norm)
      net = tf.reshape(net,[-1,32,32,1])

      Auxlogit = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

      net = layers.repeat(net, (h//32)-1, layers.conv2d_transpose,64, [4, 4], stride=2)
      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      # Make sure that generator output is in the same range as `inputs`, ie [-1, 1].
      Logit = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

      return (Logit + 1)/2 , (Auxlogit + 1)/2

def Onlydecoder(img, weight_decay=2.5e-5,keep_prob=0.5, is_training=True):
  """Core MNIST generator.

  This function is reused between the different GAN modes (unconditional,
  conditional, etc).

  Args:
    img: A 4D Tensor of shape [batch size, height, width ,channels].
    weight_decay: The value of the l2 weight decay.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A generated image in the range [-1, 1].
  """
  shape = img.get_shape().as_list()
  with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d_transpose, layers.conv2d],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.contrib.framework.arg_scope(
        [layers.batch_norm], is_training=is_training):
      # net = layers.conv2d(img, 32, [4, 4])
      # net = layers.conv2d(net, 1, [4, 4])
      net = tf.reshape(img,[shape[0], 4, 4,-1])
      net = layers.conv2d(net, 64, [4, 4]) # reduce the paraments
      net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      net = layers.conv2d(net, 64, [4, 4])
      net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      net = layers.conv2d(net, 64, [4, 4])
      net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
      net = layers.conv2d(net, 32, [4, 4])
      net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
      net = layers.conv2d(net, 32, [4, 4])
      net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
      # Make sure that generator output is in the same range as `inputs`, ie [-1, 1].
      net = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

      return net

def Vanillia(img,group_channel =32, pointwise_filter =32, weight_decay=2.5e-5,keep_prob=0.5, is_training=True):
  shape = img.get_shape().as_list()
  channel = int(shape[3] * shape[2] *shape[1]/16)
  with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.contrib.framework.arg_scope(
        [layers.batch_norm], is_training=is_training):
      net = layers.conv2d(img, 32, [3, 3])
      net = layers.conv2d(net, 1, [3, 3])
      net = tf.reshape(net,[shape[0],4,4,-1])
      net = layers.conv2d(net, channel, [1, 1])
      net = layers.conv2d(net, channel, [1, 1])

      # net = layers.separable_conv2d(net, channel, [3, 3], 5,normalizer_fn=layers.batch_norm)

      # net = channel_shuffle(net,group_channel)
      # net = group_conv(net, group_channel=group_channel, pointwise_filter = pointwise_filter,kernel_size=[3,3])
      # net = layers.separable_conv2d(net, None, [3, 3], group_channel//pointwise_filter, normalizer_fn=layers.batch_norm)
      # net = channel_shuffle(net,group_channel)
      # net = group_conv(net, group_channel=group_channel, pointwise_filter = pointwise_filter,kernel_size=[3,3])
      # net = layers.separable_conv2d(net, None, [3, 3], group_channel//pointwise_filter, normalizer_fn=layers.batch_norm)

      net = tf.reshape(net,shape)
      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      net = layers.conv2d(net, 64, [3, 3])
      net = layers.conv2d(net, 64, [3, 3])
      net = layers.conv2d(net, 64, [3, 3])
      net = layers.conv2d(net, 32, [3, 3])
      net = layers.conv2d(net, 32, [3, 3])
      net = layers.conv2d(net, 32, [3, 3])
      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      # Make sure that generator output is in the same range as `inputs` ie [-1, 1].
      net = layers.conv2d(net, 1, [3, 3], normalizer_fn=None, activation_fn=tf.tanh)
      return net
