from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import time
import os

import tensorflow as tf

slim = tf.contrib.slim
tfgan = tf.contrib.gan
layers = tf.contrib.layers

from network import unconditional_generator
from network import unconditional_discriminator
from network import SRMatrix

from utils import get_optimizer
from utils import get_learning_rate
from utils import get_init_fn

from loss import cross_entropy_loss
from loss import combine_loss
from loss import dice_loss
from loss import mae_loss
from loss import l1_loss

import config
FLAGS = tf.app.flags.FLAGS

logdir = os.path.join(FLAGS.path_prefix, FLAGS.logdir)
# logdir = os.path.join(logdir, "{:%m%d-%H%M}".format(datetime.datetime.now()))
traindir = os.path.join(logdir,'train')

def slim_model(inputs,targets):
  generated_data,Aux_data = SRMatrix(inputs, FLAGS.weight_decay, FLAGS.keep_prob, use_aux=FLAGS.useaux)
  with tf.name_scope('Train_summary'):
    summeried_num = FLAGS.grid_size * FLAGS.grid_size
    reshaped_images = tfgan.eval.image_reshaper(inputs[:summeried_num, ...], num_cols=FLAGS.grid_size)
    reshaped_generated_data = tfgan.eval.image_reshaper(generated_data[:summeried_num, ...], num_cols=FLAGS.grid_size)
    reshaped_targets = tfgan.eval.image_reshaper(targets[:summeried_num, ...], num_cols=FLAGS.grid_size)
    tf.summary.image('Inputs', reshaped_images, FLAGS.max_summary_images)
    tf.summary.image('Generated_data', reshaped_generated_data, FLAGS.max_summary_images)
    tf.summary.image('Real_data', reshaped_targets, FLAGS.max_summary_images)


  with tf.name_scope('Aux_summary'):
    b, h, w, c = targets.get_shape().as_list()
    Aux_data_1 = Aux_data[0]
    Aux_data_2 = Aux_data[1]
    Aux_data_3 = Aux_data[2]
    reshaped_auxdata_1 = tfgan.eval.image_reshaper(Aux_data_1[:summeried_num, ...], num_cols=FLAGS.grid_size)
    reshaped_auxdata_2 = tfgan.eval.image_reshaper(Aux_data_2[:summeried_num, ...], num_cols=FLAGS.grid_size)
    reshaped_auxdata_3 = tfgan.eval.image_reshaper(Aux_data_3[:summeried_num, ...], num_cols=FLAGS.grid_size)
    tf.summary.image('Aux_data_1', reshaped_auxdata_1, FLAGS.max_summary_images)
    tf.summary.image('Aux_data_2', reshaped_auxdata_2, FLAGS.max_summary_images)
    tf.summary.image('Aux_data_3', reshaped_auxdata_3, FLAGS.max_summary_images)
    aux_loss_1 = combine_loss(Aux_data_1, tf.image.resize_images(targets, [h // 8, w // 8]), name='Aux_loss_1')
    aux_loss_2 = combine_loss(Aux_data_2, tf.image.resize_images(targets, [h // 4, w // 4]), name='Aux_loss_2')
    aux_loss_3 = combine_loss(Aux_data_3, tf.image.resize_images(targets, [h // 2, w // 2]), name='Aux_loss_3')
    aux_loss = aux_loss_1 + aux_loss_2 + aux_loss_3

  with tf.name_scope('Train_Loss'):
    main_loss = combine_loss(generated_data,targets,add_summary=True,name='Main_loss')
    reg_loss = tf.losses.get_regularization_loss()
    total_loss = main_loss + reg_loss + aux_loss * int(FLAGS.useaux)
    total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
    tf.summary.scalar('Regularization_loss',reg_loss)
    tf.summary.scalar('Total_loss',total_loss)

  lr = get_learning_rate(FLAGS.learning_rate)
  optimizer = get_optimizer(lr)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  # with tf.control_dependencies(update_ops):
  train_op = slim.learning.create_train_op(total_loss, optimizer, update_ops =update_ops)
  tf.summary.scalar('Learning_rate', lr)

  slim.learning.train(train_op, traindir,
                      number_of_steps =FLAGS.max_iter,
                      log_every_n_steps=FLAGS.log_n_steps,
                      init_fn= None, # init_fn = get_init_fn()
                      save_summaries_secs=FLAGS.save_summaries_secs,
                      save_interval_secs = FLAGS.save_interval_secs)

def tfgan_model(inputs,targets):
    """Configure Conditional GAN based on TFGAN

  Args:
    inputs: generator_inputs tensor, which are detected images.
    targets: real_data tensor, which are ground truth imaegs.
    cond ：Conditional GAN is applied.
  Returns:
    Run GAN network.
    """
    # Create a GANModel tuple.
    gan_model = tfgan.gan_model(
        generator_fn=unconditional_generator,
        discriminator_fn=unconditional_discriminator,
        real_data=targets,
        generator_inputs=inputs)

    with tf.name_scope('Train_summary'):
      summeried_num = FLAGS.grid_size * FLAGS.grid_size
      reshaped_images = tfgan.eval.image_reshaper(inputs[:summeried_num, ...], num_cols=FLAGS.grid_size)
      reshaped_targets = tfgan.eval.image_reshaper(targets[:summeried_num, ...], num_cols=FLAGS.grid_size)
      reshaped_generated_data = tfgan.eval.image_reshaper(gan_model.generated_data[:summeried_num, ...],
                                                          num_cols=FLAGS.grid_size)
      tf.summary.image('Inputs', reshaped_images, FLAGS.max_summary_images)
      tf.summary.image('Generated_data', reshaped_generated_data, FLAGS.max_summary_images)
      tf.summary.image('Real_data', reshaped_targets, FLAGS.max_summary_images)
      # tfgan.eval.add_gan_model_image_summaries(gan_model, FLAGS.grid_size)

    # Define the GANLoss tuple using standard library functions,
    # return namedtuples.GANLoss(gen_loss + gen_reg_loss, dis_loss + dis_reg_loss)
    with tf.name_scope('GAN_loss'):
      gan_loss = tfgan.gan_loss(
        gan_model,
        gradient_penalty_weight=1.0,
        mutual_information_penalty_weight=0.0,
        add_summaries=True)
      tfgan.eval.add_regularization_loss_summaries(gan_model)

    # Modify the loss tuple to include the pixel loss. Add summaries as well.
    # return gan_loss._replace(generator_loss=combined_loss).
    with tf.name_scope('G_loss'):
      main_loss = combine_loss(gan_model.generated_data, targets, add_summary=True)
      gan_loss = tfgan.losses.combine_adversarial_loss(
          gan_loss = gan_loss, non_adversarial_loss = main_loss,
          gan_model = gan_model, # Used to access the generator's variables.
          weight_factor=FLAGS.adversarial_loss_weight, # weight * adversarial_loss
          scalar_summaries = True, gradient_summaries = False)

    with tf.name_scope('train_ops'):
      gen_lr = get_learning_rate(FLAGS.generator_lr)
      dis_lr = get_learning_rate(FLAGS.discriminator_lr)
      train_ops = tfgan.gan_train_ops(
          gan_model,
          gan_loss,
          generator_optimizer=get_optimizer(gen_lr),
          discriminator_optimizer=get_optimizer(dis_lr),
          summarize_gradients=False,
          colocate_gradients_with_ops=True,
          aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
          transform_grads_fn=tf.contrib.training.clip_gradient_norms_fn(1e3))
      tf.summary.scalar('generator_lr', gen_lr)
      tf.summary.scalar('discriminator_lr', dis_lr)

    train_steps = tfgan.GANTrainSteps(generator_train_steps=2, discriminator_train_steps=1)
    status_message = tf.string_join(['Train step: ', tf.as_string(tf.train.get_or_create_global_step())],
                                     name='status_message')
    if FLAGS.max_iter == 0:
      raise ValueError('The number of iteration must be positive integer')
    else:
      tfgan.gan_train(train_ops, logdir = traindir,
                      get_hooks_fn=tfgan.get_joint_train_hooks(train_steps),
                      hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_iter),
                             tf.train.LoggingTensorHook([status_message], every_n_iter=FLAGS.log_n_steps)],
                      save_checkpoint_secs=FLAGS.save_interval_secs, save_summaries_steps = FLAGS.save_summaries_steps)

