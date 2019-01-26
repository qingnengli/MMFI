import datetime,time,os
import tensorflow as tf
import config, data_loader
from network import MLP,Unet,pix2pix_G,pix2pix_D,generator_fn
from utils import circle,get_optimizer,get_lr,get_summary_image
from utils import get_multimodel_init_fn, get_init_fn, get_tfgan_init_fn
from loss import combine_loss,correlation,discriminator_loss,generator_loss

slim = tf.contrib.slim
tfgan = tf.contrib.gan
layers = tf.contrib.layers
FLAGS = tf.app.flags.FLAGS

logdir = os.path.join(FLAGS.path_prefix, FLAGS.logdir)
# logdir = os.path.join(logdir, "{:%m%d\%H%M}".format(datetime.datetime.now()))


def Generator_1(inputs,targets):

  traindir = os.path.join(logdir, 'G1\\pix2pix_G')
  if tf.gfile.Exists(traindir):
    tf.gfile.DeleteRecursively(traindir)
  tf.gfile.MakeDirs(traindir)

  fiber_output,fiber_input = inputs
  encoder, label = targets

  with tf.variable_scope('Generator'):
    with tf.variable_scope('G1'):
      generated_input = pix2pix_G(fiber_output) * circle(FLAGS.input_size,FLAGS.input_size)

  with tf.name_scope('Train_summary'):
    reshaped_fiber_output = get_summary_image(fiber_output,FLAGS.grid_size)
    reshaped_fiber_input = get_summary_image(fiber_input,FLAGS.grid_size)
    reshaped_generated_input = get_summary_image(generated_input,FLAGS.grid_size)
    tf.summary.image('Fiber_Output', reshaped_fiber_output)
    tf.summary.image('Fiber_Input', reshaped_fiber_input)
    tf.summary.image('Generated_Input', reshaped_generated_input)

  with tf.name_scope('g1_loss'):
    G1_loss = combine_loss(generated_input, fiber_input, add_summary=True)
  with tf.name_scope('Train_Loss'):
    reg_loss = tf.losses.get_regularization_loss()
    total_loss = G1_loss + reg_loss
    total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
    tf.summary.scalar('Regularization_loss',reg_loss)
    tf.summary.scalar('G1_loss', G1_loss)
    tf.summary.scalar('Total_loss',total_loss)

  lr = get_lr(FLAGS.generator_lr)
  optimizer = get_optimizer(lr)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  train_op = slim.learning.create_train_op(total_loss, optimizer, update_ops =update_ops,
                                           variables_to_train=
                                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                             scope='Generator/G1')
                                           )

  with tf.name_scope('Train_ops'):
    psnr = tf.reduce_mean(tf.image.psnr(generated_input, fiber_input, max_val=1.0))
    ssim = tf.reduce_mean(tf.image.ssim(generated_input, fiber_input, max_val=1.0))
    corr = correlation(generated_input, fiber_input)
    tf.summary.scalar('PSNR', psnr)
    tf.summary.scalar('SSIM', ssim)
    tf.summary.scalar('Relation', corr)
    tf.summary.scalar('Learning_rate', lr)

  slim.learning.train(train_op, traindir,
                      number_of_steps =FLAGS.max_iter,
                      log_every_n_steps=FLAGS.log_n_steps,
                      # init_fn=get_init_fn('E:\GitHub\MMFI\log\\G1\\Unet'),
                      save_summaries_secs=FLAGS.save_summaries_secs,
                      save_interval_secs = FLAGS.save_interval_secs)

def Generator_2(inputs,targets):

  traindir = os.path.join(logdir, 'G2\\pix2pix_G')
  if tf.gfile.Exists(traindir):
    tf.gfile.DeleteRecursively(traindir)
  tf.gfile.MakeDirs(traindir)

  fiber_output,fiber_input = inputs
  encoder, label = targets

  with tf.variable_scope('Generator'):
    with tf.variable_scope('G2'):
      generated_data = pix2pix_G(fiber_input) * circle(FLAGS.input_size,FLAGS.input_size)

  with tf.name_scope('Train_summary'):
    reshaped_fiber_input = get_summary_image(fiber_input,FLAGS.grid_size)
    reshaped_label = get_summary_image(label,FLAGS.grid_size)
    reshaped_generated_data = get_summary_image(generated_data,FLAGS.grid_size)
    tf.summary.image('Fiber_Input', reshaped_fiber_input)
    tf.summary.image('Fiber_Label', reshaped_label)
    tf.summary.image('Generated_Data', reshaped_generated_data)

  with tf.name_scope('g2_loss'):
    G2_loss = combine_loss(generated_data, label, add_summary=True)
  with tf.name_scope('Train_Loss'):
    reg_loss = tf.losses.get_regularization_loss()
    total_loss = G2_loss + reg_loss
    total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
    tf.summary.scalar('Regularization_loss',reg_loss)
    tf.summary.scalar('G2_loss', G2_loss)
    tf.summary.scalar('Total_loss',total_loss)

  lr = get_lr(FLAGS.generator_lr)
  optimizer = get_optimizer(lr)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  train_op = slim.learning.create_train_op(total_loss, optimizer, update_ops =update_ops,
                                           variables_to_train=
                                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                             scope='Generator/G2')
                                           )

  with tf.name_scope('Train_ops'):
    psnr = tf.reduce_mean(tf.image.psnr(generated_data, label, max_val=1.0))
    ssim = tf.reduce_mean(tf.image.ssim(generated_data, label, max_val=1.0))
    corr = correlation(generated_data, label)
    tf.summary.scalar('PSNR', psnr)
    tf.summary.scalar('SSIM', ssim)
    tf.summary.scalar('Relation', corr)
    tf.summary.scalar('Learning_rate', lr)

  slim.learning.train(train_op, traindir,
                      number_of_steps =FLAGS.max_iter,
                      log_every_n_steps=FLAGS.log_n_steps,
                      # init_fn=get_init_fn('E:\GitHub\MMFI\log\\G2\\pix2pix_G'),
                      save_summaries_secs=FLAGS.save_summaries_secs,
                      save_interval_secs = FLAGS.save_interval_secs)

def Discriminator(inputs,targets):

  traindir = os.path.join(logdir, 'G2\\pix2pix_D')
  if tf.gfile.Exists(traindir):
    tf.gfile.DeleteRecursively(traindir)
  tf.gfile.MakeDirs(traindir)

  fiber_output,fiber_input = inputs
  encoder, label = targets

  with tf.variable_scope('Generator'):
    with tf.variable_scope('G2'):
      generated_data = pix2pix_G(fiber_input) * circle(FLAGS.input_size,FLAGS.input_size)

  with tf.variable_scope('Discriminator',reuse=tf.AUTO_REUSE):
      discriminator_gen_outputs = pix2pix_D(tf.concat((generated_data,fiber_input),-1))
      discriminator_real_outputs = pix2pix_D(tf.concat((label, fiber_input), -1))

  with tf.name_scope('Train_summary'):
    reshaped_label = get_summary_image(label,FLAGS.grid_size)
    reshaped_fiber_input = get_summary_image(fiber_input,FLAGS.grid_size)
    reshaped_generated_data = get_summary_image(generated_data,FLAGS.grid_size)
    tf.summary.image('Fiber_Label', reshaped_label)
    tf.summary.image('Fiber_Input', reshaped_fiber_input)
    tf.summary.image('Generated_Data', reshaped_generated_data)

  with tf.name_scope('Train_Loss'):
    predict_real = discriminator_real_outputs
    predict_fake = discriminator_gen_outputs
    discrim_real_loss = tf.reduce_mean(tf.abs(1-predict_real))
    discrim_gen_loss = tf.reduce_mean(tf.abs(-1-predict_fake))
    discrim_loss = discrim_real_loss + discrim_gen_loss
    total_loss = discrim_loss +  tf.losses.get_regularization_loss()
    total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
    tf.summary.scalar('Total_loss',total_loss)
    tf.summary.scalar('discrim_loss', discrim_loss)
    tf.summary.scalar('discrim_real_loss',discrim_real_loss)
    tf.summary.scalar('discrim_gen_loss',discrim_gen_loss)

  with tf.name_scope('Train_OP'):
    tf.summary.scalar('predict_real', tf.reduce_mean(predict_real))
    tf.summary.scalar('predict_fake', tf.reduce_mean(predict_fake))
    tf.summary.scalar('discrim_lr', get_lr(FLAGS.discriminator_lr,decay_steps=5000))

  train_op = slim.learning.create_train_op(total_loss,
                                           get_optimizer(get_lr(FLAGS.discriminator_lr,decay_steps=5000)),
                                           update_ops =tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                                           variables_to_train=
                                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                          scope='Discriminator')
                                           )

  slim.learning.train(train_op, traindir,
                      number_of_steps =FLAGS.max_iter,
                      log_every_n_steps=FLAGS.log_n_steps,
                      init_fn=get_init_fn('E:\GitHub\MMFI\log\\G2\\pix2pix_G',
                                          inclusion_scope=['Generator/G2']),
                      save_summaries_secs=FLAGS.save_summaries_secs,
                      save_interval_secs = FLAGS.save_interval_secs)

def Generator_all(inputs,targets):

  traindir = os.path.join(logdir, 'GG12\\CNN')
  if tf.gfile.Exists(traindir):
    tf.gfile.DeleteRecursively(traindir)
  tf.gfile.MakeDirs(traindir)

  fiber_output,fiber_input = inputs
  encoder, label = targets

  with tf.variable_scope('Generator'):
    with tf.variable_scope('G1'):
      generated_input = pix2pix_G(fiber_output) * circle(FLAGS.input_size,FLAGS.input_size)
    with tf.variable_scope('G2'):
      generated_data = pix2pix_G(generated_input) * circle(FLAGS.input_size,FLAGS.input_size)

  with tf.name_scope('Train_summary'):
    reshaped_fiber_input = get_summary_image(fiber_input,FLAGS.grid_size)
    reshaped_label = get_summary_image(label,FLAGS.grid_size)
    reshaped_generated_input = get_summary_image(generated_input,FLAGS.grid_size)
    reshaped_generated_data = get_summary_image(generated_data,FLAGS.grid_size)
    tf.summary.image('Input_Fiber', reshaped_fiber_input)
    tf.summary.image('Input_Generator', reshaped_generated_input)
    tf.summary.image('Data_Real', reshaped_label)
    tf.summary.image('Data_Generator', reshaped_generated_data)

  with tf.name_scope('g1_loss'):
    G1_loss = combine_loss(generated_input, fiber_input, add_summary=True)
  with tf.name_scope('g2_loss'):
    G2_loss = combine_loss(generated_data, label, add_summary=True)
  with tf.name_scope('Train_Loss'):
    reg_loss = tf.losses.get_regularization_loss()
    total_loss = G1_loss + G2_loss + reg_loss
    total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
    tf.summary.scalar('Regularization_loss',reg_loss)
    tf.summary.scalar('G1_loss', G1_loss)
    tf.summary.scalar('G2_loss', G2_loss)
    tf.summary.scalar('Total_loss',total_loss)

  lr = get_lr(1e-5,decay_steps=5000)
  optimizer = get_optimizer(lr)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  train_op = slim.learning.create_train_op(total_loss, optimizer, update_ops =update_ops,
                                           variables_to_train=
                                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                             scope='Generator')
                                           )

  with tf.name_scope('Train_ops'):
    psnr = tf.reduce_mean(tf.image.psnr(generated_data, label, max_val=1.0))
    ssim = tf.reduce_mean(tf.image.ssim(generated_data, label, max_val=1.0))
    corr = correlation(generated_data, label)
    tf.summary.scalar('PSNR', psnr)
    tf.summary.scalar('SSIM', ssim)
    tf.summary.scalar('Relation', corr)
    tf.summary.scalar('Learning_rate', lr)

  slim.learning.train(train_op, traindir,
                      number_of_steps =FLAGS.max_iter,
                      log_every_n_steps=FLAGS.log_n_steps,
                      init_fn=get_multimodel_init_fn(ckpt1='E:\GitHub\MMFI\log\\G1\\pix2pix_G',
                                                     include1='Generator/G1',
                                                     ckpt2='E:\GitHub\MMFI\log\\G2\\pix2pix_G',
                                                     include2='Generator/G2'),
                      save_summaries_secs=FLAGS.save_summaries_secs,
                      save_interval_secs = FLAGS.save_interval_secs)

def TFGAN(inputs,targets):

    traindir = os.path.join(logdir, 'GG12\\PIX2PIX_MINMAX_1024')
    if tf.gfile.Exists(traindir):
      tf.gfile.DeleteRecursively(traindir)
    tf.gfile.MakeDirs(traindir)

    # Create a GANModel tuple.
    fiber_output, fiber_input = inputs
    encoder, label = targets
    real_data = tf.concat((label,fiber_input),-1)
    #######################################################################
    ##########################  GAN MODEL #################################
    #######################################################################
    gan_model = tfgan.gan_model(
        generator_fn=generator_fn,
        discriminator_fn=pix2pix_D,
        real_data=real_data,
        generator_inputs=fiber_output,
        generator_scope='Generator',
        discriminator_scope='Discriminator')

    #######################################################################
    ##########################  GAN SUMMARY ###############################
    #######################################################################
    with tf.name_scope('Train_summary'):
      generated_data, generated_input = tf.split(gan_model.generated_data,2,-1)
      reshaped_fiber_input = get_summary_image(fiber_input, FLAGS.grid_size)
      reshaped_label = get_summary_image(label, FLAGS.grid_size)
      reshaped_generated_input = get_summary_image(generated_input, FLAGS.grid_size)
      reshaped_generated_data = get_summary_image(generated_data, FLAGS.grid_size)
      tf.summary.image('Input_Fiber', reshaped_fiber_input)
      tf.summary.image('Input_Generator', reshaped_generated_input)
      tf.summary.image('Data_Real', reshaped_label)
      tf.summary.image('Data_Generator', reshaped_generated_data)

    #######################################################################
    ##########################  GAN LOSS  #################################
    #######################################################################
    with tf.name_scope('pixel_loss'):
      pixel_loss = combine_loss(gan_model.generated_data,
                                gan_model.real_data,
                                add_summary=True)
    with tf.name_scope('gan_loss'):
      gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=tfgan.losses.modified_generator_loss,
        discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
        gradient_penalty_weight=1.0, # only in wassertein_loss
      )
      tfgan.eval.add_regularization_loss_summaries(gan_model)
    with tf.name_scope('Train_Loss'):
      gan_loss = tfgan.losses.combine_adversarial_loss(
          gan_loss, gan_model, pixel_loss,
          weight_factor=FLAGS.adversarial_loss_weight)

    #######################################################################
    ##########################   GAN OPS   ################################
    #######################################################################
    with tf.name_scope('Train_ops'):
      gen_lr = get_lr(1e-5,decay_steps=5000)
      dis_lr = get_lr(5e-5,decay_steps=5000)
      train_ops = tfgan.gan_train_ops(
          gan_model,  gan_loss,
          generator_optimizer=get_optimizer(gen_lr),
          discriminator_optimizer=get_optimizer(dis_lr),
          # summarize_gradients=False,
          # colocate_gradients_with_ops=True,
          # transform_grads_fn=tf.contrib.training.clip_gradient_norms_fn(1e3),
          # aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
          )
      psnr = tf.reduce_mean(tf.image.psnr(generated_data, label, max_val = 1.0))
      ssim = tf.reduce_mean(tf.image.ssim(generated_data, label, max_val = 1.0))
      corr = correlation(generated_data, label)
      tf.summary.scalar('PSNR', psnr)
      tf.summary.scalar('SSIM', ssim)
      tf.summary.scalar('Relation', corr)
      tf.summary.scalar('generator_lr', gen_lr)
      # tf.summary.scalar('discriminator_lr', dis_lr)

    #######################################################################
    ##########################   GAN TRAIN   ##############################
    #######################################################################
    train_steps = tfgan.GANTrainSteps(generator_train_steps=1, discriminator_train_steps=1)
    message = tf.string_join([' Train step: ', tf.as_string(tf.train.get_or_create_global_step()),
                              '   PSNR:', tf.as_string(psnr), '   SSIM:', tf.as_string(ssim),
                              '   Correlation:', tf.as_string(corr)
                              ], name='status_message')

    tfgan.gan_train(train_ops, logdir = traindir,  get_hooks_fn=tfgan.get_joint_train_hooks(train_steps),
                    hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_iter),
                           tf.train.LoggingTensorHook([message], every_n_iter=FLAGS.log_n_steps),
                           get_tfgan_init_fn('E:\GitHub\MMFI\log\\GG12\\CNN', 'Generator'),
                           # get_tfgan_init_fn('E:\GitHub\MMFI\log\\G2\\pix2pix_D', 'Discriminator'),
                           ],
                    save_summaries_steps = FLAGS.save_summaries_steps*2,
                    save_checkpoint_secs = FLAGS.save_interval_secs)

def main(_):
  with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    # Read data from disk
    fiber_output, fiber_input, encoder, label = data_loader.read_inputs('train.txt', True)

    # Generator_1(inputs=[fiber_output,fiber_input], targets=[encoder,label])
    # Generator_2(inputs=[fiber_output,fiber_input], targets=[encoder,label])
    # Discriminator(inputs=[fiber_output,fiber_input], targets=[encoder,label])
    Generator_all(inputs=[fiber_output,fiber_input], targets=[encoder,label])
    # TFGAN(inputs=[fiber_output, fiber_input], targets=[encoder,label])

if __name__ == '__main__':
    tf.app.run()

