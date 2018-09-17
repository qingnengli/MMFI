"""
(Conditional) Discriminator network is built.
G_loss, D_loss and L1_loss are added as total loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import numpy as np
import tensorflow as tf

from model import tfgan_model
from model import slim_model

import data_loader
import config as cfg

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

def main(_):
  with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    # Read data from disk
    # images: input, targets: ground truth images, label: Multi-shapes
    images, targets,optics = data_loader.read_inputs(is_training = True)

    logdir = os.path.join(FLAGS.path_prefix,FLAGS.logdir)
    logdir = os.path.join(logdir, "{:%m%d-%H%M}".format(datetime.datetime.now()))
    traindir = os.path.join(logdir,'train')
    if not tf.gfile.Exists(traindir):
    #   tf.gfile.DeleteRecursively(traindir)
      tf.gfile.MakeDirs(traindir)

    # Storary Memory is reduced with development of Network
    # tfgan.model:425MB --> slim.model: 30MB --> slim(FCN):3MB

    if FLAGS.use_tfgan:
      tfgan_model(inputs=images, targets=optics)
    else:
      slim_model(inputs=images, targets=optics)


if __name__ == '__main__':
    tf.app.run()
