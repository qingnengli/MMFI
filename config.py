from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

###############################################################
#                     Data loader                             #
###############################################################
tf.app.flags.DEFINE_string(
    'path_prefix', 'E:\GitHub\MMFI',
    'the prefix address for images.')

tf.app.flags.DEFINE_string(
    'delimiter', ' ', 'Delimiter of the input files.')
###############################################################
#                     Train Model                             #
###############################################################
tf.app.flags.DEFINE_string(
    'logdir', 'log',
    'The directory of saved checkpoints and logging events.')

tf.app.flags.DEFINE_integer(
    'output_channel', 1, 'The total channels of output from backbone')

tf.app.flags.DEFINE_integer(
    'input_size',64,
    'The size of images and labels for input.')

tf.app.flags.DEFINE_integer(
    'batch_size',64,'The training batch size.')

tf.app.flags.DEFINE_integer(
    'grid_size', 4, 'Grid size for image visualization.')

tf.app.flags.DEFINE_float(
    'generator_lr', 0.001, 'Initial Generator learning rate.')

tf.app.flags.DEFINE_float(
    'discriminator_lr', 0.0001, 'Initial Discriminator learning rate.')

tf.app.flags.DEFINE_integer(
    'max_iter', 45000,
    'The max iteration to train, the total train step')

###############################################################
#                        Loss Flags                           #
###############################################################
tf.app.flags.DEFINE_float(
    'l1_loss_weight', 0.0,
    'The weight of absolute error loss.')

tf.app.flags.DEFINE_float(
    'SSIM_loss_weight', 10.0,
    'How much to weight the cross entropy loss.')

tf.app.flags.DEFINE_float(
    'mmae_loss_weight', 1.0,
    'The weight of modified absolute error loss.')

tf.app.flags.DEFINE_float(
    'dice_loss_weight', 0.0,
    'How much to weight the dice loss.')

tf.app.flags.DEFINE_float(
    'adversarial_loss_weight', 0.5,
    'How much to weight the adversarial loss relative to L1 loss.')

###############################################################
#                     Saver and Summary                       #
###############################################################
tf.app.flags.DEFINE_integer(
    'log_n_steps', 10,
    'The steps of logging information')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 300,
    'The seconds of summary image')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 1800,
    'The seconds for saving trained checkpoint model')

tf.app.flags.DEFINE_integer(
    'save_summaries_steps', 300,
    'The train steps of summary image')

tf.app.flags.DEFINE_integer(
    'save_interval_steps', 2000,
    'The steps for saving checkpoint')
###############################################################
#                     Optimizer FLAG                          #
###############################################################
tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type', 'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')


FLAGS = tf.app.flags.FLAGS
