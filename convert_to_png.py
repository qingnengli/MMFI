from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

###############################################################
#                     Data loader                             #
###############################################################
tf.app.flags.DEFINE_string(
    'path_prefix', '/home/amax/SIAT/MMFI',
    'the prefix address for images.')

tf.app.flags.DEFINE_string(
    'data_file', 'train.txt',
    'Name of the file containing addresses and labels of training images.')

tf.app.flags.DEFINE_string(
    'delimiter', ' ', 'Delimiter of the input files.')

tf.app.flags.DEFINE_integer(
    'input_size',128,
    'The size of images and labels for input.')

tf.app.flags.DEFINE_integer(
    'crop_size',800,
    'The size of images and labels for effective spots.')

tf.app.flags.DEFINE_integer(
    'label_num_channels',1,
    'Auto select the color channel of the decoded label.')

tf.app.flags.DEFINE_integer(
    'image_num_channels',3,
    'Auto select the color channel of the decoded image.')

tf.app.flags.DEFINE_integer(
    'num_threads',32,
    'The number of threads for loading data.')

tf.app.flags.DEFINE_boolean(
    'shuffle', True,'Shuffle training data or not.')

tf.app.flags.DEFINE_integer(
    'batch_size',32,'The training batch size.')

# tf.app.flags.DEFINE_integer(
#     'num_sample',3000,'The number of samples per epoch.')

# tf.app.flags.DEFINE_integer(
#     'num_gpus',1,'Number of GPUs.')

###############################################################
#                     Train Model                             #
###############################################################
tf.app.flags.DEFINE_boolean(
    'trainable', True,'Whether is train or not.')

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00005, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'final_endpoint', 'Conv2d_7b_1x1', 'The final endpoint of backbone.')

tf.app.flags.DEFINE_integer(
    'output_stride', 16,
    'The total stride of output from backbone')

tf.app.flags.DEFINE_integer(
    'num_output_channel', 1,
    'The total channels of output from backbone')

tf.app.flags.DEFINE_boolean(
    'align_feature_maps', True,
    'When true, changes all the VALID paddings in the network '
    'to SAME padding so that the feature maps are aligned')

tf.app.flags.DEFINE_float(
    'keep_prob', 0.8,
    'The keep probability of dropout layer.')

tf.app.flags.DEFINE_integer(
    'num_classes', 100,
    'The number of classes for classifier network')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'logdir', 'log',
    'The directory of saved checkpoints and logging events.')

tf.app.flags.DEFINE_integer(
    'max_summary_images', 4,
    'The max_number of summuaried images')

tf.app.flags.DEFINE_string(
    'train_mode', 'inception_resnet_v2',
    'The mode for training different sub-network.')

tf.app.flags.DEFINE_float(
    'classifier_loss_weight', 1.0,
    'The weight of classifier_loss.')

tf.app.flags.DEFINE_float(
    'L1_loss_weight', 1.0,
    'The weight of l1 loss between outputs and targets.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The seconds of summary image')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 1200,
    'The seconds for saving trained checkpoint model')

tf.app.flags.DEFINE_integer(
    'max_iter', 300000,
    'The max iteration to train, the total train step')

tf.app.flags.DEFINE_boolean(
    'restore_variable', False, 'Whether to use saved variable from checkpoints to restore.')

tf.app.flags.DEFINE_boolean('vanilla', False,'Whether to use vanilla model')
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
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

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

tf.app.flags.DEFINE_float('learning_rate', 0.002,
                          'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'decay_steps', 2.0*100,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average. '
    'If left as None, then moving averages are not used.')

###############################################################
#                       GAN  Flags                            #
###############################################################
tf.app.flags.DEFINE_boolean(
    'GAN', True, 'Whether to use GAN architecture or not.')

tf.app.flags.DEFINE_float('generator_lr', 0.001,
                          'Initial Generator learning rate.')

tf.app.flags.DEFINE_float('discriminator_lr', 0.001,
                          'Initial Discriminator learning rate.')

tf.app.flags.DEFINE_float(
    'adversarial_loss_weight', 1.0,
    'How much to weight the adversarial loss relative to pixel loss.')

tf.app.flags.DEFINE_integer(
    'save_summaries_steps', 100,
    'The train steps of summary image')

tf.app.flags.DEFINE_boolean(
    'patch_level', True, 'Whether to use patch_level logits to predict or not.')

FLAGS = tf.app.flags.FLAGS
