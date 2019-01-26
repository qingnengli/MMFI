import tensorflow as tf
import os,data_loader, config

from loss import correlation
from utils import circle,get_summary_image
from network import MLP,pix2pix_G

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim
tfgan = tf.contrib.gan
CURRENT_DIR=os.path.dirname(__file__)
# os.environ["CUDA_VISIBLE_DEVICES"]=''


def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)
	with tf.Graph().as_default():
		logdir = 'E:\GitHub\MMFI\log\GG12\\CNN'
		evaldir = os.path.join(logdir, 'eval')
		if not tf.gfile.Exists(evaldir):
			# tf.gfile.DeleteRecursively(evaldir)
			tf.gfile.MakeDirs(evaldir)

		with tf.name_scope('inputs'):
			fiber_output, fiber_input, encoder, label = data_loader.read_inputs('valid.txt', False)

		with tf.variable_scope('Generator'):
			with tf.variable_scope('G1'):
				generated_input = pix2pix_G(fiber_output, is_training=False) \
				                  * circle(FLAGS.input_size,FLAGS.input_size)
			with tf.variable_scope('G2'):
				generated_data = pix2pix_G(generated_input,is_training=False)\
				                 * circle(FLAGS.input_size,FLAGS.input_size)

		with tf.name_scope('Valid_summary'):
			reshaped_fiber_input = get_summary_image(fiber_input, FLAGS.grid_size)
			reshaped_label = get_summary_image(label, FLAGS.grid_size)
			reshaped_generated_input = get_summary_image(generated_input, FLAGS.grid_size)
			reshaped_generated_data = get_summary_image(generated_data, FLAGS.grid_size)
			tf.summary.image('Input_Fiber', reshaped_fiber_input)
			tf.summary.image('Input_Generator', reshaped_generated_input)
			tf.summary.image('Data_Real', reshaped_label)
			tf.summary.image('Data_Generator', reshaped_generated_data)

		with tf.name_scope('Valid_op'):
			psnr = tf.reduce_mean(tf.image.psnr(generated_data, label, max_val=1.0))
			ssim = tf.reduce_mean(tf.image.ssim(generated_data, label, max_val=1.0))
			corr = correlation(generated_data, label)
			# inception_score = get_inception_score(generated_data)

			tf.summary.scalar('PSNR', psnr)
			tf.summary.scalar('SSIM', ssim)
			tf.summary.scalar('Relation', corr)

			grate = tf.ones([1,FLAGS.grid_size*FLAGS.input_size,10,1],dtype=tf.float32)
			reshaped_images = tf.concat((reshaped_generated_input, grate,
			                             reshaped_fiber_input, grate,
			                             reshaped_label, grate,
			                             reshaped_generated_data, grate), 2)
			uint8_images = tf.cast(reshaped_images*255, tf.uint8)
			image_write_ops = tf.write_file('%s/%s' % (evaldir, 'Generator_is_training_False.png'),
			                                tf.image.encode_png(uint8_images[0]))

			status_message = tf.string_join([' PSNR: ', tf.as_string(psnr), ' ',
			                                 ' SSIM: ', tf.as_string(ssim), ' ',
			                                 ' Correlation: ', tf.as_string(corr)],
			                                name='status_message')


		checkpoint_path = tf.train.latest_checkpoint(logdir)
		tf.logging.info('Evaluating %s' % checkpoint_path)

		tf.contrib.training.evaluate_once(
			checkpoint_path,
			hooks=[tf.contrib.training.SummaryAtEndHook(evaldir),
			       tf.contrib.training.StopAfterNEvalsHook(50),
						 tf.train.LoggingTensorHook([status_message],every_n_iter=5)],
			eval_ops=image_write_ops)

if __name__ == '__main__':
	tf.app.run()


