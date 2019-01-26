#-*- coding: UTF-8 -*-

# ==============================================================================
"""Convert Label(.BMP) and Image(.TIF) to PNG format

convert BMP(logic, 1 bit) and TIF to standard PNG format.
For Label,the initial data is BMP format and Logic dtype.
Actually,Label with only 0 and 1 value is expected, but
tf.image.decode_bmp can not parse these logic bmp images.

Image.save function can save PNG images with 0&1 values(
dtype=uint8), without visualization. Image.open function
only can load PNG Image with 0-255, instead of 0&1.
However Tensorflow and Matlab can read with 0&1.
Every converted result will be saved in a specific Label_dir.
"""
# ==============================================================================
import cv2
import glob
import os.path
import numpy as np
from PIL import Image

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('original_folder', 'E:\GitHub\MMFI\endoscopy\DMD_augmentor',
                           'Original ground truth annotations.')
tf.app.flags.DEFINE_string('original_format', 'bmp', 'Original format.')

tf.app.flags.DEFINE_string('label_format', 'jpg', 'Output format.')
tf.app.flags.DEFINE_string('label_dir','E:\GitHub\MMFI\data\\11_16\\label_binary_aug',
                           'folder to save modified ground truth annotations.')

# PIL.Image.save convert logic dtype to uint8(0 and 1), and save.
def _save_annotation(annotation, filename):
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  with tf.gfile.Open(filename, mode='w') as f:
    pil_image.save(f, 'JPEG')


def _source_preprocess(source):
  # Make mask to extract approriate regions
  shape = np.shape(source)
  mask = np.zeros([shape[0],shape[1]])
  radius = int(min(shape)//2)
  cv2.circle(mask,(shape[0]//2,shape[1]//2),radius,1,-1)
  processed_label = source * mask
  return processed_label

def main(unused_argv):
  # Create the output directory if not exists.
  if not tf.gfile.IsDirectory(FLAGS.label_dir):
    tf.gfile.MakeDirs(FLAGS.label_dir)

  annotations = glob.glob(os.path.join(FLAGS.original_folder, '*.' + FLAGS.original_format))
  for annotation in annotations:
    raw_annotation = np.array(Image.open(annotation))
    # raw_annotation = _source_preprocess(raw_annotation)
    # raw_annotation = np.equal(raw_annotation,1)

    # filename = FLAGS.original_folder[-7:-2] + '_'+ FLAGS.original_folder[-1] + '_'  + os.path.basename(annotation)[:-4]
    filename = os.path.basename(annotation)[:-4]
    print(filename)
    _save_annotation(raw_annotation*255, os.path.join(FLAGS.label_dir, filename + '.' + FLAGS.label_format))


if __name__ == '__main__':
    tf.app.run()
