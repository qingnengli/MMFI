"""
(Conditional) Discriminator network is built.
G_loss, D_loss and L1_loss are added as total loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

from train_util import get_learning_rate
from train_util import get_optimizer
from train_util import get_init_fn
from train_util import get_gan_model
from train_util import get_generator_model
from train_util import get_loss

import data_loader
import config as cfg
FLAGS = tf.app.flags.FLAGS

import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

EPS = 1e-12

def main(_):
  with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    # Read data from disk
    # images: input, targets: ground truth images, label: Multi-shapes
    images, targets,conds = data_loader.read_inputs(FLAGS.trainable)
    tf.summary.image('Images',images,FLAGS.max_summary_images)
    tf.summary.image('Targets',targets,FLAGS.max_summary_images)
    tf.summary.image('Condition', conds, FLAGS.max_summary_images)

    logdir = os.path.join(FLAGS.path_prefix,FLAGS.logdir)
    traindir = os.path.join(logdir, "{:%m%d-%H%M}".format(datetime.datetime.now()))
    if not tf.gfile.Exists(traindir):
        tf.gfile.MakeDirs(traindir)

    # # Select train_mode to train model
    # output = get_generator_model((0, images), mode=FLAGS.train_mode)
    # tf.summary.image('Output', output, FLAGS.max_summary_images)
    #
    # total_loss = get_loss(output, FLAGS.train_mode)
    # train_op = slim.learning.create_train_op(total_loss,
    #                                          optimizer=get_optimizer(get_learning_rate(FLAGS.learning_rate)),
    #                                          update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    # slim.learning.train(train_op=train_op, logdir=traindir,
    #                     save_summaries_secs=FLAGS.save_summaries_secs,
    #                     save_interval_secs=FLAGS.save_interval_secs,
    #                     number_of_steps=FLAGS.max_iter,
    #                     init_fn=get_init_fn() if FLAGS.restore_variable else None
    #                     )
    input = tf.concat((images,conds),axis=-1)
    get_gan_model(inputs = input,targets = targets )


if __name__ == '__main__':
    tf.app.run()
