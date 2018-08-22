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

import glob
import os.path
import numpy as np
from PIL import Image

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('original_folder',
                           '/media/amax/DISK_IMG/PASCALVOC',
                           'Original ground truth annotations.')
tf.app.flags.DEFINE_string('original_format', 'bmp', 'Original format.')

tf.app.flags.DEFINE_string('label_format', 'png', 'Output format.')
tf.app.flags.DEFINE_string('label_dir','/home/amax/SIAT/MMFI/label/PASCALVOC',
                           'folder to save modified ground truth annotations.')

# PIL.Image.save convert logic dtype to uint8(0 and 1), and save.
def _save_annotation(annotation, filename):
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  with tf.gfile.Open(filename, mode='w') as f:
    pil_image.save(f, 'PNG')

def main(unused_argv):
  # Create the output directory if not exists.
  if not tf.gfile.IsDirectory(FLAGS.label_dir):
    tf.gfile.MakeDirs(FLAGS.label_dir)

  annotations = glob.glob(os.path.join(FLAGS.original_folder,
                                       '*.' + FLAGS.original_format))
  for annotation in annotations:
    raw_annotation = np.array(Image.open(annotation))
    filename = os.path.basename(annotation)[:-4]
    _save_annotation(raw_annotation,
                     os.path.join(FLAGS.label_dir,
                                  filename + '.' + FLAGS.label_format))


if __name__ == '__main__':
    tf.app.run()
