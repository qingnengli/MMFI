from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import config
FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim
layers = tf.contrib.layers


def get_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer

def get_learning_rate(init_lr):
  """Configures the learning rate.

  Args:
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(learning_rate=init_lr,
                                      global_step=tf.train.get_or_create_global_step(),
                                      decay_steps=FLAGS.decay_steps,
                                      decay_rate=FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(learning_rate=init_lr,
                                     global_step=tf.train.get_or_create_global_step(),
                                     decay_steps=FLAGS.decay_steps,
                                     decay_rate=FLAGS.end_learning_rate,
                                     power=0.9,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)

def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    if not FLAGS.checkpoint_exclude_scopes:
        checkpoint_exclude_scopes = FLAGS.checkpoint_exclude_scopes
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
        variables_to_restore = []
        for var in slim.get_model_variables():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
                variables_to_restore.append(var)
    else:
        variables_to_restore = slim.get_model_variables()

    checkpoint_path = FLAGS.checkpoint_path if FLAGS.checkpoint_path \
        else tf.train.latest_checkpoint(FLAGS.traindir)

    # In common tensorflow, Restore the variables: init_fn(sess)
    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)


def group_conv(tensor, group_channel, pointwise_filter, kernel_size):
  shape = tensor.get_shape().as_list()
  split_num = int(shape[-1]//group_channel)
  net_splits = tf.split(tensor,split_num,axis=-1)
  net = [layers.conv2d(net_split, pointwise_filter, kernel_size,
                       normalizer_fn=layers.batch_norm
                       ) for net_split in net_splits]
  net = tf.concat(net, axis=-1)
  return net

def channel_shuffle(x, group_channel):
  """The first and last channel is fixed, 
  but the others are random shuffled."""
  n, h, w, c = x.shape.as_list()
  x_reshaped = tf.reshape(x, [-1, h, w,  c // group_channel, group_channel])
  x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
  output = tf.reshape(x_transposed, [-1, h, w, c])
  return output


def upsample(net, num_outputs, resize_shape=None, ksize = 2,
             stride = 2, method='deconv2d', name = 'Upsample'):
    """Upsamples the given inputs."""
    net_shape = tf.shape(net)
    height = net_shape[1]
    width = net_shape[2]
    if method == 'bilinear':
        net = slim.conv2d(net, num_outputs, 1)
        net = tf.image.resize_bilinear(net, resize_shape, name=name) if resize_shape else \
            tf.image.resize_bilinear(net, [stride * height, stride * width],name=name)
    elif method == 'deconv2d':
        net = layers.conv2d_transpose(net, num_outputs, kernel_size= ksize, stride = stride,
                                           padding='SAME', activation_fn=tf.nn.relu, scope = name)
    else:
        raise ValueError('Unknown method: [%s]', method)
    return net


def SPP(net, patch_size=[3, 5, 10], num_channel=8,scope = None):
    """Spatial Pyramid Pooling """
    with tf.variable_scope(scope,'SPP_map',[net]):
      _, h, w, c = net.get_shape().as_list()
      feature = tf.reduce_mean(net,axis=[1,2])
      feature = slim.flatten(feature)
      for psize in patch_size:
          h_wid = int(np.ceil(float(h) / psize))
          w_wid = int(np.ceil(float(w) / psize))
          h_pad = (h_wid * psize - psize + 1) / 2
          w_pad = (w_wid * psize - psize + 1) / 2
          # padding must be shape=[n,2],n means rank of tensor net
          padding = [[0, 0], [int(np.ceil(float(h_pad) / 2)), int(np.floor(float(h_pad) / 2))],
                     [int(np.ceil(float(w_pad) / 2)), int(np.floor(float(w_pad) / 2))], [0, 0]]
          padding_net = tf.pad(net, padding, mode='CONSTANT')
          patch = layers.max_pool2d(padding_net, [h_wid, w_wid], [h_wid, w_wid])
          patch = layers.conv2d(patch,num_channel)
          patch_pixel = layers.flatten(patch)
          feature = tf.concat((feature,patch_pixel),axis=-1)
    return feature
