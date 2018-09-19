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

tf.app.flags.DEFINE_integer(
    'input_size',256,
    'The size of images and labels for input.')

tf.app.flags.DEFINE_integer(
    'crop_size',800,
    'The size of images and labels for effective spots.')

tf.app.flags.DEFINE_integer(
    'image_num_channels',1,
    'Auto select the color channel of the decoded image.')

tf.app.flags.DEFINE_integer(
    'label_num_channels',1,
    'Auto select the color channel of the decoded label.')

tf.app.flags.DEFINE_integer(
    'cond_num_channels',1,
    'Auto select the color channel of the decoded image.')

tf.app.flags.DEFINE_integer(
    'num_threads',4,
    'The number of threads for loading data.')
###############################################################
#                     Train Model                             #
###############################################################
tf.app.flags.DEFINE_string(
    'logdir', 'log',
    'The directory of saved checkpoints and logging events.')

tf.app.flags.DEFINE_integer(
    'batch_size',16,'The training batch size.')

tf.app.flags.DEFINE_float('weight_decay', 1e-3,
                          'The weight decay of generator.')

tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'keep_prob', 0.4,
    'The keep probability of dropout layer.')

tf.app.flags.DEFINE_integer(
    'output_channel', 1,
    'The total channels of output from backbone')

tf.app.flags.DEFINE_integer(
    'max_iter', 15000,
    'The max iteration to train, the total train step')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.6, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'decay_steps', 10.0*100,
    'Number of epochs after which learning rate decays.')
###############################################################
#                        Loss Flags                           #
###############################################################
tf.app.flags.DEFINE_float(
    'mae_loss_weight', 2.0,
    'The weight of MS-SSIM loss.')

tf.app.flags.DEFINE_float(
    'entropy_loss_weight', 2.0,
    'How much to weight the cross entropy loss.')

tf.app.flags.DEFINE_float(
    'dice_loss_weight', 1.0,
    'How much to weight the dice loss.')

tf.app.flags.DEFINE_float(
    'l1_loss_weight', 0.0,
    'How much to weight the dice loss.')

tf.app.flags.DEFINE_boolean(
    'useaux', True,
    'How much to weight the Aux loss.')
###############################################################
#                     Restore and Init                        #
###############################################################
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of '
    'variables to exclude when restoring from a checkpoint.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', 'log/train',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_boolean(
    'restore_variable', False,
    'Whether to use saved variable '
    'from checkpoints to restore.')
###############################################################
#                     Saver and Summary                       #
###############################################################
tf.app.flags.DEFINE_integer(
    'max_summary_images', 4,
    'The max_number of summuaried images')

tf.app.flags.DEFINE_integer(
    'log_n_steps', 10,
    'The steps of logging information')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 300,
    'The seconds of summary image')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 300,
    'The seconds for saving trained checkpoint model')

tf.app.flags.DEFINE_integer(
    'eval_interval_secs', 300,
    'The seconds of summary image')

tf.app.flags.DEFINE_integer(
    'save_summaries_steps', 500,
    'The train steps of summary image')

tf.app.flags.DEFINE_integer(
    'save_interval_steps', 500,
    'The seconds of summary image')
###############################################################
#                     Optimizer FLAG                          #
###############################################################
tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.5,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1e-08, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.99,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.99, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.99, 'Decay term for RMSProp.')
###############################################################
#                     Learning Rate Flags                     #
###############################################################
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type', 'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 1e-10,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

###############################################################
#                        TFGAN Flags                          #
###############################################################
tf.app.flags.DEFINE_boolean('use_tfgan',True,
                            'Whether to use TFGAN as main model')

tf.app.flags.DEFINE_integer(
    'grid_size', 4, 'Grid size for image visualization.')

tf.app.flags.DEFINE_float('generator_lr', 0.001,
                          'Initial Generator learning rate.')

tf.app.flags.DEFINE_float('discriminator_lr', 0.001,
                          'Initial Discriminator learning rate.')

tf.app.flags.DEFINE_float(
    'adversarial_loss_weight', 0.001,
    'How much to weight the adversarial loss relative to L1 loss.')

FLAGS = tf.app.flags.FLAGS

