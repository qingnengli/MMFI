from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import datetime
import tensorflow as tf

slim = tf.contrib.slim
tfgan = tf.contrib.gan

from network import _arg_scope
from network import backbone
from network import Inception_resnet_v2_model as IRV2
from network import vanilla_CNN_model as VCNN
from network import discriminator

import config as cfg
FLAGS = tf.app.flags.FLAGS


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

    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

def get_loss(outputs):
    """
    Save total losses into tf.GraphKeys.Losses collection
    Use TFlosses to obtain the list of tf.GraphKeys.Losses
    """
    with tf.variable_scope('losses'):
        tf.summary.image('Output', outputs, FLAGS.max_summary_images)
        # Absolute difference between outputs and targets
        # abs(targets - outputs) => 0
        l1_loss = tf.losses.absolute_difference(targets, outputs)
        l1_loss = tf.losses.compute_weighted_loss(l1_loss, weights=FLAGS.L1_loss_weight)
        tf.summary.scalar('L1_loss', l1_loss)

        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
        tf.summary.scalar('total_loss', total_loss)

        return total_loss

def get_generator_model(inputs):
    """
    Args:
        inputs: A 2-tuple of Tensors (noise, conditions).

    Returns:
        A generated image in the range [0, 1].

    Raises:
        ValueError: if mode of train model is not recognized.

    """
    noise , input = inputs
    model_arg_scope = _arg_scope(is_training=FLAGS.trainable,weight_decay=FLAGS.weight_decay)
    if FLAGS.train_mode == 'inception_resnet_v2':
        with slim.arg_scope(model_arg_scope):
            final_endpoint, end_points = backbone(input, final_endpoint=FLAGS.final_endpoint,
                                                output_stride=FLAGS.output_stride,
                                                align_feature_maps=FLAGS.align_feature_maps)
            net , _ = IRV2(end_points, output_stride=FLAGS.output_stride,
                                  dropout_keep_prob=FLAGS.keep_prob)
    elif FLAGS.train_mode == 'vanilla':
        with slim.arg_scope(model_arg_scope):
            net, _ = VCNN(input)
    else:
        raise ValueError('The train mode [%s] was not supported for GAN',FLAGS.train_mode)

    tf.summary.image('Output',net,FLAGS.max_summary_images)
    return net


def get_discriminator_model(input,condition):
    """Conditional discriminator network on MNIST digits.

    Args:
        input: Real or generated MNIST digits. Should be in the range [0, 1].
        condition: A 2-tuple of Tensors representing (noise, conditional_image),
                   where conditional_image is a two-channel tensor, identical to generator_input.

    Returns:
        Logits for the probability that the image is real.
    """
    _, conditional_image = condition
    model_arg_scope = _arg_scope(is_training=FLAGS.trainable,weight_decay=FLAGS.weight_decay)
    with slim.arg_scope(model_arg_scope):
        pred = discriminator(input,conditional_image,FLAGS.patch_level)
        pred = slim.flatten(pred)

    return pred

def get_gan_model(inputs,targets):
    """Configure Conditional GAN based on TFGAN

  Args:
    inputs: generator_inputs tensor, which are detected images.
    targets: real_data tensor, which are ground truth imaegs.

  Returns:
    Run GAN network (similar to pix2pix).

    """
    generator_fn = get_generator_model
    discriminator_fn = get_discriminator_model
    # Create a GANModel tuple.
    gan_model = tfgan.gan_model(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        real_data=targets,
        generator_inputs=(tf.zeros_like(inputs),inputs))

    # Define the GANLoss tuple using standard library functions,
    # including regularization loss
    with tf.name_scope('GAN_losses'):
      gan_loss = tfgan.gan_loss(
          gan_model,
          generator_loss_fn=tfgan.losses.least_squares_generator_loss,
          discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss)

      # Define the standard L1 pixel loss.
      l1_pixel_loss = tf.losses.absolute_difference(gan_model.real_data, gan_model.generated_data)
      tf.summary.scalar('L1_pixel_loss',l1_pixel_loss)

      # Modify the loss tuple to include the pixel loss. Add summaries as well.
      # Assume the GANLoss.generator_loss is the adversarial loss.
      # The non_adversarial loss is the main loss
      gan_loss = tfgan.losses.combine_adversarial_loss(
          gan_loss = gan_loss, non_adversarial_loss = l1_pixel_loss,
          gan_model = gan_model, # Used to access the generator's variables.
          weight_factor=FLAGS.adversarial_loss_weight,
          scalar_summaries = True, gradient_summaries = False)

    with tf.name_scope('train_ops'):
      # Get the GANTrain ops using the custom optimizers and optional
      # discriminator weight clipping.
      gen_lr = get_learning_rate(FLAGS.generator_lr)
      dis_lr = get_learning_rate(FLAGS.discriminator_lr)
      gen_opt = get_optimizer(gen_lr)
      dis_opt = get_optimizer(dis_lr)
      train_ops = tfgan.gan_train_ops(
          gan_model,
          gan_loss,
          generator_optimizer=gen_opt,
          discriminator_optimizer=dis_opt,
          summarize_gradients=False,
          colocate_gradients_with_ops=True,
          aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
          transform_grads_fn=tf.contrib.training.clip_gradient_norms_fn(1e3))
      tf.summary.scalar('generator_lr', gen_lr)
      tf.summary.scalar('discriminator_lr', dis_lr)

    # Use GAN train step function if using adversarial loss, otherwise
    # only train the generator.
    train_steps = tfgan.GANTrainSteps(
        generator_train_steps=1,
        discriminator_train_steps=int(FLAGS.GAN))

    # Run the alternating training loop. Skip it if no steps should be taken
    # (used for graph construction tests).
    status_message = tf.string_join(
        ['Starting train step: ',
         tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')
    if FLAGS.max_iter == 0: return

    logdir = os.path.join(FLAGS.path_prefix,FLAGS.logdir)
    traindir = os.path.join(logdir, "{:%m%d-%H%M}".format(datetime.datetime.now()))
    return tfgan.gan_train(train_ops, logdir = traindir,
                           get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
                           hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_iter),
                                  tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
                           save_checkpoint_secs=FLAGS.save_interval_secs,
                           save_summaries_steps = FLAGS.save_summaries_steps)

