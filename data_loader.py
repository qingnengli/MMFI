from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
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
    condpath = imagepath.replace('/image','/cond')
    shape = 0
    for image_name in images_list:
        label_name = image_name.replace('.jpg','.png')
        fd.write('{}/{} {}/{} {}/{} {}\n'.format(imagepath,image_name,
                                           labelpath,label_name,
                                           condpath,image_name,
                                           shape))
    fd.close()

# Parse the input file name
def _read_label_file(file, delimiter):
  f = open(file, "r")
  imagepaths = []
  labelpaths = []
  condpaths = []
  shape =[]
  for line in f:
    tokens = line.split(delimiter)
    imagepaths.append(tokens[0])
    labelpaths.append(tokens[1])
    condpaths.append(tokens[2])
    shape.append(int(tokens[3]))
  return imagepaths, labelpaths,condpaths,shape

def read_inputs(is_training):
  if is_training:
    shuffle = True
    file = 'train.txt'
  else:
    shuffle = False
    file = 'eval.txt'

  data_file = os.path.join(FLAGS.path_prefix, file)
  imagepaths, labelpaths,condpaths,shape  = _read_label_file(data_file, FLAGS.delimiter)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.slice_input_producer([imagepaths, labelpaths,condpaths,shape], shuffle= shuffle,
                                                   capacity= 1024)
  # Data information
  imagename = filename_queue[0]
  labelname = filename_queue[1]
  condname = filename_queue[2]

  # Read examples from files in the filename queue.
  image_content = tf.read_file(imagename)
  label_content = tf.read_file(labelname)
  cond_content = tf.read_file(condname)

  # Read JPEG or PNG or GIF image from file
  image = tf.to_float(tf.image.decode_jpeg(image_content, channels=FLAGS.image_num_channels))
  label = tf.to_float(tf.image.decode_png(label_content,channels=FLAGS.label_num_channels))
  cond = tf.to_float(tf.image.decode_jpeg(cond_content, channels=FLAGS.image_num_channels))


  # Pre-Process Image and Label data
  reshaped_image = _image_preprocess(image)
  reshaped_label = _label_preprocess(label)
  reshaped_cond = _cond_preprocess(cond)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(5000*min_fraction_of_examples_in_queue)
  #print(batch_size)
  print ('Filling queue with %d images before starting to train. '
         'This may take some times.' % min_queue_examples)
  # batch_size = int(FLAGS.batch_size/FLAGS.num_gpus) if is_training else FLAGS.batch_size

  # Load images and labels with additional info and return batches
  images_batch, label_batch,cond_batch = tf.train.batch(
      [reshaped_image, reshaped_label,reshaped_cond],
      batch_size= FLAGS.batch_size,
      num_threads=FLAGS.num_threads,
      capacity= min_queue_examples + 3 * FLAGS.batch_size,
      allow_smaller_final_batch=True)
  images_batch = tf.reshape(images_batch, shape=[FLAGS.batch_size, FLAGS.input_size,
                                                 FLAGS.input_size, FLAGS.image_num_channels])
  label_batch = tf.reshape(label_batch, shape=[FLAGS.batch_size, FLAGS.input_size,
                                               FLAGS.input_size, FLAGS.label_num_channels])
  cond_batch = tf.reshape(cond_batch, shape=[FLAGS.batch_size, FLAGS.input_size,
                                                 FLAGS.input_size, FLAGS.cond_num_channels])
  return images_batch, label_batch, cond_batch


def _image_preprocess(image):
  # Extract the effective regions
  processed_image = tf.image.resize_image_with_crop_or_pad(image,FLAGS.crop_size,FLAGS.crop_size)
  processed_image = tf.image.resize_images(processed_image, [FLAGS.input_size,FLAGS.input_size])
  # Subtract off the mean and divide by the variance of the pixels.
  processed_image = tf.image.per_image_standardization(processed_image)
  return processed_image

def _cond_preprocess(cond):
  # Extract the effective regions
  processed_image = tf.image.resize_image_with_crop_or_pad(cond,FLAGS.crop_size,FLAGS.crop_size)
  processed_image = tf.image.resize_images(processed_image, [FLAGS.input_size,FLAGS.input_size])
  # Subtract off the mean and divide by the variance of the pixels.
  min = tf.reduce_min(cond)
  max = tf.reduce_max(cond)
  processed_image = (processed_image - min)/(max-min)
  return processed_image

import cv2
import numpy as np
def _label_preprocess(label):
  processed_label = tf.image.resize_image_with_crop_or_pad(label, FLAGS.crop_size, FLAGS.crop_size)
  processed_label = tf.image.resize_images(processed_label, [FLAGS.input_size, FLAGS.input_size])
  # Make mask to extract approriate regions

  mask = np.zeros([FLAGS.input_size,FLAGS.input_size,FLAGS.label_num_channels])
  radius = int(FLAGS.input_size/2)
  cv2.circle(mask,(radius,radius),radius,1,-1)
  mask = tf.convert_to_tensor(mask,dtype=tf.float32)
  processed_label = processed_label * mask

  # Ensure the label with 0&1 value
  processed_label = tf.cast(tf.greater(processed_label,0),dtype=tf.float32)
  return processed_label



if __name__ == '__main__':
    for i in range(5):
        imagepath = os.path.join(CURRENT_DIR,'input/image/mnist') + '/' + str(i)
        labelpath = os.path.join(CURRENT_DIR,'label/mnist') + '/' + str(i)
        if not os.path.exists('train.txt'):
            generate_dataset_list(imagepath,labelpath, 'train' + str(i) +'.txt')
