import tensorflow as tf
import numpy as np
import config
import cv2

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim
layers = tf.contrib.layers
tfgan = tf.contrib.gan
###############################################################
#                     TFSLIM or TFGAN                         #
###############################################################
def get_optimizer(learning_rate,optimizer_type='adam'):

  if optimizer_type == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=0.95,
        epsilon=1e-08)
  elif optimizer_type == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=0.1)
  elif optimizer_type == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08)
  elif optimizer_type == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power = -0.5,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0)
  elif optimizer_type == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum= 0.99,
        name='Momentum')
  elif optimizer_type == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay= 0.99,
        momentum= 0.99,
        epsilon=1e-08)
  elif optimizer_type == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', optimizer_type)
  return optimizer

def get_lr(init_lr=1e-3, decay_steps=10000,
           learning_rate_decay_type='exponential',
           learning_rate_decay_factor=0.5):

  if learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(learning_rate=init_lr,
                                      global_step=tf.train.get_or_create_global_step(),
                                      decay_steps=decay_steps,
                                      decay_rate=learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif learning_rate_decay_type == 'fixed':
    return tf.constant(init_lr, name='fixed_learning_rate')
  elif learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(learning_rate=init_lr,
                                     global_step=tf.train.get_or_create_global_step(),
                                     decay_steps=decay_steps,
                                     decay_rate=1e-10,
                                     power=0.9,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     learning_rate_decay_type)

def get_init_fn(checkpoint_dir,exclusion_scope=None, inclusion_scope = None):
    """Returns a function run by the chief worker to warm-start the training.
    slim.get_model_variable do not include global_step and so on.
    """

    variables_to_restore = []
    if exclusion_scope is not None:
      exclusions = [scope.strip() for scope in exclusion_scope]
      for var in slim.get_model_variables():
        for exclusion in exclusions:
          if var.op.name.startswith(exclusion):
            break
        else:
          variables_to_restore.append(var)

    if inclusion_scope is not None:
      inclusions = [scope.strip() for scope in inclusion_scope]
      for var in slim.get_model_variables():
        for inclusion in inclusions:
          if var.op.name.startswith(inclusion):
            variables_to_restore.append(var)

    if exclusion_scope is None:
      if inclusion_scope is None:
        variables_to_restore = slim.get_model_variables()

    # variables_to_restore = [var for var in tf.trainable_variables() if var.op.name.startswith(inclusion)]

    # variables_to_restore = slim.get_variables_to_restore(include=inclusion_scope,
    #                                                      exclude=exclusion_scope)
    # variables_to_restore = slim.filter_variables(var_list, include_patterns=None,
    #                                              exclude_patterns=None, reg_search=True)

    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    # In common tensorflow, Restore the variables: init_fn(sess)
    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

def get_multimodel_init_fn(ckpt1,include1,ckpt2,include2):

  variables_to_restore_1 = []
  variables_to_restore_2 = []
  for var in slim.get_model_variables():
    if var.op.name.startswith(include1):
      variables_to_restore_1.append(var)
  for var in slim.get_model_variables():
    if var.op.name.startswith(include2):
      variables_to_restore_2.append(var)

  checkpoint_path_1 = tf.train.latest_checkpoint(ckpt1)
  checkpoint_path_2 = tf.train.latest_checkpoint(ckpt2)

  ########################### Var_list_1 #######################
  var_list_1 = variables_to_restore_1
  grouped_vars_1 = {}
  if isinstance(var_list_1, (tuple, list)):
    for var in var_list_1:
      ckpt_name = slim.get_variable_full_name(var)
      if ckpt_name not in grouped_vars_1:
        grouped_vars_1[ckpt_name] = []
      grouped_vars_1[ckpt_name].append(var)
  else:
    for ckpt_name, value in var_list_1.items():
      if isinstance(value, (tuple, list)):
        grouped_vars_1[ckpt_name] = value
      else:
        grouped_vars_1[ckpt_name] = [value]
  ########################### Var_list_2 #######################
  var_list_2 = variables_to_restore_2
  grouped_vars_2 = {}
  if isinstance(var_list_2, (tuple, list)):
    for var in var_list_2:
      ckpt_name = slim.get_variable_full_name(var)
      if ckpt_name not in grouped_vars_2:
        grouped_vars_2[ckpt_name] = []
      grouped_vars_2[ckpt_name].append(var)
  else:
    for ckpt_name, value in var_list_2.items():
      if isinstance(value, (tuple, list)):
        grouped_vars_2[ckpt_name] = value
      else:
        grouped_vars_2[ckpt_name] = [value]

  # Read each checkpoint entry. Create a placeholder variable and
  # add the (possibly sliced) data from the checkpoint to the feed_dict.
  feed_dict = {}
  assign_ops = []
  ########################### Assign_op_1 & Feed_dict_1 #######################
  reader_1 = tf.train.NewCheckpointReader(checkpoint_path_1)
  for ckpt_name in grouped_vars_1:
    if not reader_1.has_tensor(ckpt_name):
      log_str = 'Checkpoint is missing variable [%s]' % ckpt_name
      raise ValueError(log_str)
    ckpt_value = reader_1.get_tensor(ckpt_name)

    for var in grouped_vars_1[ckpt_name]:
      placeholder_tensor = tf.placeholder(
        dtype=var.dtype.base_dtype,
        shape=var.get_shape(),
        name='placeholder/' + var.op.name)
      assign_ops.append(var.assign(placeholder_tensor))

      if not var._save_slice_info:
        if var.get_shape() != ckpt_value.shape:
          raise ValueError(
            'Total size of new array must be unchanged for %s '
            'lh_shape: [%s], rh_shape: [%s]'
            % (ckpt_name, str(ckpt_value.shape), str(var.get_shape())))
        feed_dict[placeholder_tensor] = ckpt_value.reshape(ckpt_value.shape)
      else:
        slice_dims = zip(var._save_slice_info.var_offset,
                         var._save_slice_info.var_shape)
        slice_dims = [(start, start + size) for (start, size) in slice_dims]
        slice_dims = [slice(*x) for x in slice_dims]
        slice_value = ckpt_value[slice_dims]
        slice_value = slice_value.reshape(var._save_slice_info.var_shape)
        feed_dict[placeholder_tensor] = slice_value
  ########################### Assign_op_1 & Feed_dict_1 #######################
  reader_2 = tf.train.NewCheckpointReader(checkpoint_path_2)
  for ckpt_name in grouped_vars_2:
    if not reader_2.has_tensor(ckpt_name):
      log_str = 'Checkpoint is missing variable [%s]' % ckpt_name
      raise ValueError(log_str)
    ckpt_value = reader_2.get_tensor(ckpt_name)

    for var in grouped_vars_2[ckpt_name]:
      placeholder_tensor = tf.placeholder(
        dtype=var.dtype.base_dtype,
        shape=var.get_shape(),
        name='placeholder/' + var.op.name)
      assign_ops.append(var.assign(placeholder_tensor))

      if not var._save_slice_info:
        if var.get_shape() != ckpt_value.shape:
          raise ValueError(
            'Total size of new array must be unchanged for %s '
            'lh_shape: [%s], rh_shape: [%s]'
            % (ckpt_name, str(ckpt_value.shape), str(var.get_shape())))
        feed_dict[placeholder_tensor] = ckpt_value.reshape(ckpt_value.shape)
      else:
        slice_dims = zip(var._save_slice_info.var_offset,
                         var._save_slice_info.var_shape)
        slice_dims = [(start, start + size) for (start, size) in slice_dims]
        slice_dims = [slice(*x) for x in slice_dims]
        slice_value = ckpt_value[slice_dims]
        slice_value = slice_value.reshape(var._save_slice_info.var_shape)
        feed_dict[placeholder_tensor] = slice_value

  assign_op = tf.group(*assign_ops)

  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(assign_op, feed_dict)
  return InitAssignFn


def get_summary_image(image,grid_size):
  num = grid_size * grid_size
  summary_image = tfgan.eval.image_reshaper(image[:num, ...],
                                            num_cols=grid_size)
  return summary_image
###############################################################
#                  TFGAN HOOKS INIT_FN                         #
###############################################################
class RestoreCheckpointHook(tf.train.SessionRunHook):
  """A hook to run train ops a fixed number of times."""

  def __init__(self, checkpoint_path,
               # exclude_scope_patterns,
               include_scope_patterns
               ):
    tf.logging.info("Create RestoreCheckpointHook.")
    # super(IteratorInitializerHook, self).__init__()
    self.checkpoint_path = checkpoint_path
    # self.exclude_scope_patterns = None if (not exclude_scope_patterns) else exclude_scope_patterns.split(',')
    self.include_scope_patterns = None if (not include_scope_patterns) else include_scope_patterns.split(',')

  def begin(self):
    # You can add ops to the graph here.
    # 1. Create saver
    variables_to_restore = slim.filter_variables( slim.get_model_variables(),
                                                  include_patterns=self.include_scope_patterns,
                                                  reg_search=True)
    self.saver = tf.train.Saver(variables_to_restore)

  def after_create_session(self, session, coord):
    # When this is called, the graph is finalized and
    # ops can no longer be added to the graph.
    self.saver.restore(session, tf.train.latest_checkpoint(self.checkpoint_path))


def get_tfgan_init_fn(ckpt,include_scope):
  generator_hook = RestoreCheckpointHook(ckpt,include_scope)
  return generator_hook


###############################################################
#                       ShuffleNet                            #
###############################################################
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

###############################################################
#                          Tools                              #
###############################################################
def circle(m,n,r=None):
  mask = np.zeros([m,n])
  # cv2.circle(img, (50, 50), 10, (0, 0, 255), -1)
  # img:图像，圆心坐标(col,row)，圆半径，颜色，线宽度(-1：表示对封闭图像进行内部填满)
  if r is None:
    r = np.min([m//2,n//2])
  cv2.circle(mask,(n//2,m//2),r,(1,0),-1)
  circle_mask = tf.convert_to_tensor(mask, dtype=tf.float32)
  circle_mask = tf.reshape(circle_mask,shape=[m,n,1])
  return circle_mask

###############################################################
#                     Inception Score                         #
###############################################################
import os.path,scipy.misc
import sys,tarfile,glob,math
from six.moves import urllib

MODEL_DIR = 'E:/GitHub/MMFI/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.

def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 1
  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.set_shape(tf.TensorShape(new_shape))
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()