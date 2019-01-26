import tensorflow as tf
import numpy as np

from utils import group_conv
from utils import channel_shuffle
from slim.nets import cyclegan
from slim.nets import pix2pix
from utils import circle

layers = tf.contrib.layers
# ----------------------------Generator --------------------------------

def MLP(img, weight_decay=2.5e-5,keep_prob=0.4, is_training=True):
  with tf.contrib.framework.arg_scope([layers.fully_connected],
    weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.flatten(img)
    net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
    net = layers.fully_connected(net, 64 * 64 * 1,
                                 normalizer_fn=None,
                                 activation_fn=tf.tanh)
    net = tf.reshape(net, [-1, 64, 64,1])
    return (net + 1) / 2

def Unet(img, weight_decay=2.5e-5, keep_prob=0.4, is_training=True):
  with tf.contrib.framework.arg_scope(
      [layers.conv2d, layers.conv2d_transpose],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.contrib.framework.arg_scope(
        [layers.batch_norm], is_training=is_training):
      net = layers.conv2d(img, 64, [3, 3], stride=1)  # 64*64
      net = layers.conv2d(net, 64, [3, 3], stride=1)
      encode_1 = net
      net = layers.max_pool2d(net, [2, 2], padding='SAME')
      net = layers.conv2d(net, 128, [3, 3], stride=1)  # 32*32
      net = layers.conv2d(net, 128, [3, 3], stride=1)
      net = layers.conv2d(net, 128, [3, 3], stride=1)
      encode_2 = net
      net = layers.max_pool2d(net, [2, 2], padding='SAME')
      net = layers.conv2d(net, 256, [3, 3], stride=1)  # 16*16
      net = layers.conv2d(net, 256, [3, 3], stride=1)
      net = layers.conv2d(net, 256, [3, 3], stride=1)
      encode_3 = net
      net = layers.max_pool2d(net, [2, 2], padding='SAME')
      net = layers.conv2d(net, 512, [3, 3], stride=1)  # 8*8
      net = layers.conv2d(net, 512, [3, 3], stride=1)
      net = layers.conv2d(net, 512, [3, 3], stride=1)
      encode_4 = net
      net = layers.max_pool2d(net, [2, 2], padding='SAME')
      net = layers.conv2d(net, 1024, [3, 3], stride=1)  # 4*4

      net = layers.conv2d(net, 1024, [3, 3], stride=1)
      net = layers.conv2d_transpose(net, 512, [2, 2], stride=2)
      net = tf.concat((net,encode_4),axis=-1)
      net = layers.conv2d(net, 512, [3, 3], stride=1)  # 8*8
      net = layers.conv2d(net, 512, [3, 3], stride=1)
      net = layers.conv2d(net, 512, [3, 3], stride=1)
      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      net = layers.conv2d_transpose(net, 256, [2, 2], stride=2)
      net = tf.concat((net,encode_3),axis=-1)
      net = layers.conv2d(net, 256, [3, 3], stride=1)  # 16*16
      net = layers.conv2d(net, 256, [3, 3], stride=1)
      net = layers.conv2d(net, 256, [3, 3], stride=1)
      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      net = layers.conv2d_transpose(net, 128, [2, 2], stride=2)
      net = tf.concat((net, encode_2), axis=-1)
      net = layers.conv2d(net, 128, [3, 3], stride=1)  # 32*32
      net = layers.conv2d(net, 128, [3, 3], stride=1)
      net = layers.conv2d(net, 128, [3, 3], stride=1)
      net = layers.dropout(net, keep_prob=keep_prob, is_training=is_training)
      net = layers.conv2d_transpose(net, 64, [2, 2], stride=2)
      net = tf.concat((net, encode_1), axis=-1)
      net = layers.conv2d(net, 64, [3, 3], stride=1)
      net = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)  # 64*64
      return (net+1)/2


def pix2pix_G(input_images,is_training=True):

    blocks = [pix2pix.Block(64, 0.5), pix2pix.Block(128, 0.5),pix2pix.Block(256, 0.5),
              pix2pix.Block(512, 0), pix2pix.Block(512, 0), ]

    with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
      output_images, _ = pix2pix.pix2pix_generator(input_images, num_outputs=1,
                                                   blocks=blocks,
                                                   upsample_method='nn_upsample_conv',
                                                   is_training=is_training)

    return (tf.tanh(output_images)+1)/2


# ----------------------------Discriminator ------------------------------
def pix2pix_D(image_batch, unused_conditioning=None):

  with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
    logits_4d, _ = pix2pix.pix2pix_discriminator(
      image_batch, num_filters=[64, 128, 256, 512])
    logits_4d.shape.assert_has_rank(4)

  # Output of logits is 4D. Reshape to 2D, for TFGAN.
  net = layers.flatten(logits_4d)
  # net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
  net = layers.fully_connected(net, 1024, activation_fn=None) # default: normalizer_fn=None
  return net

# ---------------------------- Combination ------------------------------
from utils import circle
def generator_fn(input_images,is_training=True):
  with tf.variable_scope('G1'):
    generated_input = pix2pix_G(input_images, is_training) * circle(64,64)
  with tf.variable_scope('G2'):
    generated_data = pix2pix_G(generated_input, is_training)* circle(64,64)

  return tf.concat((generated_data, generated_input), axis=-1)