from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import numpy as np
import tensorflow as tf

from model import unconditional_generator
from model import Vanillia
from model import Onlydecoder
from model import SRMatrix
from loss import tf_ms_ssim
from loss import cross_entropy_loss
from loss import dice_loss
from loss import mae_loss
from loss import ssim_loss
from loss import l1_loss

import data_loader
import config as cfg

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim
tfgan = tf.contrib.gan

os.environ["CUDA_VISIBLE_DEVICES"]=''
def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)
	with tf.Graph().as_default():
		logdir = os.path.join(FLAGS.path_prefix, FLAGS.logdir)
		# logdir = os.path.join(logdir, "{:%m%d-%H%M}".format(datetime.datetime.now()))
		evaldir = os.path.join(logdir,'eval')
		if not tf.gfile.Exists(evaldir):
			# tf.gfile.DeleteRecursively(evaldir)
			tf.gfile.MakeDirs(evaldir)

		# images: input, targets: ground truth images, label: Multi-shapes
		with tf.name_scope('inputs'):
			images, mnist, targets = data_loader.read_inputs(is_training = False)

		if FLAGS.use_tfgan:
			with tf.variable_scope('Generator'):
				generated_data = SRMatrix(images,is_training = False)
		else:
			generated_data,Aux_data= SRMatrix(images,is_training = False)

		# Choose the metrics to compute:
		names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
			"eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(generated_data, targets),
			"eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(generated_data, targets),
		})

		# Create the summary ops such that they also print out to std output:
		with tf.name_scope('Valid_Loss'):
			ssim = ssim_loss(generated_data, targets)
			mae = mae_loss(generated_data, targets)
			dice = dice_loss(generated_data, targets)
			entropy_loss = cross_entropy_loss(generated_data, targets)
			total_loss = mae * FLAGS.mae_loss_weight \
									 + dice * FLAGS.dice_loss_weight \
									 + ssim * FLAGS.ssim_loss_weight\
									 +entropy_loss * FLAGS.entropy_loss_weight

			tf.summary.scalar('Total_loss', total_loss)
			tf.summary.scalar('Cross_entropy_loss', entropy_loss)
			tf.summary.scalar('Dice_loss', dice)
			tf.summary.scalar('MAE_loss', mae)
			tf.summary.scalar('SSIM_loss', ssim)

		with tf.name_scope('Valid_summary'):
			summeried_num = FLAGS.grid_size * FLAGS.grid_size
			reshaped_images = tfgan.eval.image_reshaper(images[:summeried_num, ...], num_cols=FLAGS.grid_size)
			reshaped_generated_data = tfgan.eval.image_reshaper(generated_data[:summeried_num, ...], num_cols=FLAGS.grid_size)
			reshaped_targets = tfgan.eval.image_reshaper(targets[:summeried_num, ...], num_cols=FLAGS.grid_size)
			tf.summary.image('Inputs', reshaped_images, FLAGS.max_summary_images)
			tf.summary.image('Generated_data', reshaped_generated_data, FLAGS.max_summary_images)
			tf.summary.image('Real_data', reshaped_targets, FLAGS.max_summary_images)

		num_examples = 1000
		num_batches = int(num_examples / FLAGS.batch_size)

		if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
			checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
		else:
			checkpoint_path = FLAGS.checkpoint_path

		tf.logging.info('Evaluating %s' % checkpoint_path)

		if FLAGS.use_tfgan:
			tf.contrib.training.evaluate_once(
				master=FLAGS.master,
				checkpoint_path = checkpoint_path,
				eval_ops=list(names_to_updates.values()),
				hooks=[tf.contrib.training.SummaryAtEndHook(evaldir),
							 tf.contrib.training.StopAfterNEvalsHook(1)])
		else:
			slim.evaluation.evaluate_once(
				master=FLAGS.master,
				checkpoint_path=checkpoint_path,
				logdir=evaldir,
				num_evals=num_batches,
				eval_op=list(names_to_updates.values()))

			# slim.evaluation.evaluation_loop(
			# 	master =FLAGS.master,
			# 	checkpoint_dir= checkpoint_path,
			# 	logdir = evaldir,
			# 	num_evals = num_batches,
			# 	eval_op = list(names_to_updates.values()),
			# 	eval_interval_secs=FLAGS.eval_interval_secs)

if __name__ == '__main__':
	tf.app.run()


