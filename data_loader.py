from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from utils import circle
from six.moves import xrange

import tensorflow as tf
import config as cfg
FLAGS = tf.app.flags.FLAGS

CURRENT_DIR=os.path.dirname(__file__)

# Generate the train/validation set list
def generate_dataset_list(imagepath,labelpath,listname):
    file = os.path.join(CURRENT_DIR,listname)
    fd = open(file, 'w')
    images_list = os.listdir(imagepath)
    # condpath = imagepath.replace('/image','/cond')
    shape = 0
    for image_name in images_list:
        # label_name = image_name.replace('.jpg','.png')
        # fd.write('{}/{} {}/{} {}/{} {}\n'.format(imagepath,image_name,
        #                                    labelpath,label_name,
        #                                    condpath,image_name,
        #                                    shape))
        fd.write('{}/{} {}/{} {}\n'.format(imagepath, image_name, labelpath, image_name, shape))
    fd.close()

# Parse the input file name
def _read_label_file(file, delimiter):
  f = open(file, "r")
  fiber_outputpaths = []
  fiber_inputpaths = []
  encoder_paths =[]
  label_paths = []
  shape =[]
  for line in f:
    tokens = line.split(delimiter)
    fiber_outputpaths.append(tokens[0])
    fiber_inputpaths.append(tokens[1])
    encoder_paths.append(tokens[2])
    label_paths.append(tokens[3])
    shape.append(int(tokens[4]))
  return fiber_outputpaths, fiber_inputpaths, encoder_paths, label_paths, shape

def read_inputs(file = 'train.txt', is_training = True):

  shuffle = True if is_training else False

  data_file = os.path.join(FLAGS.path_prefix, file)
  fiber_outputpaths, fiber_inputpaths, encoder_paths, label_paths, shape= _read_label_file(data_file, FLAGS.delimiter)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.slice_input_producer([fiber_outputpaths, fiber_inputpaths,
                                                  encoder_paths, label_paths, shape],
                                                 shuffle= shuffle, capacity= 1024)
  # Data information
  fiber_outputname = filename_queue[0]
  fiber_inputname = filename_queue[1]
  encoder_name = filename_queue[2]
  label_name = filename_queue[3]

  # Read examples from files in the filename queue.
  fiber_output_content = tf.read_file(fiber_outputname)
  fiber_input_content = tf.read_file(fiber_inputname)
  encoder_content = tf.read_file(encoder_name)
  label_content = tf.read_file(label_name)

  # Read JPEG or PNG or GIF image from file
  fiber_output = tf.to_float(tf.image.decode_jpeg(fiber_output_content, channels=1))
  fiber_input = tf.to_float(tf.image.decode_jpeg(fiber_input_content, channels=1))
  encoder = tf.to_float(tf.image.decode_jpeg(encoder_content, channels=1))
  label = tf.to_float(tf.image.decode_jpeg(label_content, channels=1))

  # Pre-Process Image and Label data
  reshaped_fiber_output = _fiber_output_preprocess(fiber_output)
  reshaped_fiber_input = _fiber_input_preprocess(fiber_input)
  reshaped_encoder = _label_preprocess(encoder)
  reshaped_label = _label_preprocess(label)

  # Load images and labels with additional info and return batches
  fiber_output_batch, fiber_input_batch, encoder_batch, label_batch = tf.train.batch(
      [reshaped_fiber_output, reshaped_fiber_input, reshaped_encoder, reshaped_label],
      batch_size= FLAGS.batch_size, num_threads=4,
      capacity= 2000 + 3 * FLAGS.batch_size,
      allow_smaller_final_batch=True)
  fiber_output_batch = tf.reshape(fiber_output_batch, shape=[FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 1])
  fiber_input_batch = tf.reshape(fiber_input_batch, shape=[FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 1])
  encoder_batch = tf.reshape(encoder_batch, shape=[FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 1])
  label_batch = tf.reshape(label_batch, shape=[FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 1])

  return fiber_output_batch, fiber_input_batch, encoder_batch, label_batch


def _fiber_output_preprocess(image):
  # Extract the effective regions
  processed_image = tf.image.resize_image_with_crop_or_pad(image,768,768)
  processed_image = tf.image.resize_images(processed_image, [FLAGS.input_size,FLAGS.input_size])
  # Subtract off the mean and divide by the variance of the pixels.
  # processed_image = tf.image.per_image_standardization(processed_image)
  min, max = tf.reduce_min(processed_image), tf.reduce_max(processed_image)
  processed_image = 2*((processed_image - min)/ tf.maximum((max - min), 1))-1
  return processed_image

def _fiber_input_preprocess(image):
  # Extract the effective regions
  processed_image = tf.image.resize_image_with_crop_or_pad(image,768,768)
  processed_image = tf.image.resize_images(processed_image, [FLAGS.input_size,FLAGS.input_size])

  processed_image = processed_image * circle(FLAGS.input_size,FLAGS.input_size)

  min, max = tf.reduce_min(processed_image), tf.reduce_max(processed_image)
  processed_image = (processed_image - min)/tf.maximum((max-min),1)
  return processed_image

def _label_preprocess(label):
  processed_image = tf.image.resize_images(label, [FLAGS.input_size, FLAGS.input_size*4//3])
  processed_image = tf.image.resize_image_with_crop_or_pad(processed_image,  FLAGS.input_size, FLAGS.input_size)
  with tf.device('/cpu:0'):
    # Rotate image(s) counterclockwise(逆时针) by the passed angle(s) in radians.
    processed_image = tf.contrib.image.rotate(images=processed_image, angles=42*3.1415926/180)

  processed_image = processed_image * circle(FLAGS.input_size,FLAGS.input_size)

  min, max = tf.reduce_min(processed_image), tf.reduce_max(processed_image)
  processed_image = (processed_image - min)/tf.maximum((max-min),1)
  return processed_image


if __name__ == '__main__':
    for i in range(15):
      imagepath = os.path.join(CURRENT_DIR,'data\\11_16\ms\\aug') + '/' + str(i+1)
      labelpath = os.path.join(CURRENT_DIR,'data\\11_16\gt\\aug') + '/' + str(i+1)
      print(imagepath,labelpath)
      generate_dataset_list(imagepath,labelpath, 'DMD_mg_aug' + str(i+1) +'.txt')

    imagepath = os.path.join(CURRENT_DIR, 'data\\11_16\ms\\na')
    labelpath = os.path.join(CURRENT_DIR, 'data\\11_16\gt\\na')
    print(imagepath,labelpath)
    generate_dataset_list(imagepath, labelpath, 'DMD_mg_nature.txt')
