#-*- coding: UTF-8 -*-

"""
The model is similar to pix2pix net, with encoder-decoder architecture.
The Backbone is based on Inception_resnet_v2 model, sharing the main weight.

The Encoded feature map will be divided into Multi-task: Classification of shape,
which is an extra but useful function to take MMF's bending/motion into account;

And The Decoder Part follows a PSP merging with "concat" skip connnection.
Additionally, The loss layer contains L1 loss, G_loss, D_loss and so on.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net

def _arg_scope(is_training=True,
               weight_decay=0.00004,
               batch_norm_decay=0.9997,
               batch_norm_epsilon=0.001,
               activation_fn=tf.nn.relu):
  """Returns the scope with the default parameters for inception_resnet_v2.
    with slim.arg_scope(_arg_scope()):
        backbone/classification/reconstruction

  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: Activation function for conv2d.

  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected,
                       slim.convolution2d_transpose,
                       slim.separable_convolution2d], trainable=is_training,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):
      with slim.arg_scope([slim.dropout],is_training=is_training):

        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'fused': None,  # Use fused batch norm if possible.
        }
        # Set activation_fn and parameters for batch_norm.
        with slim.arg_scope([slim.conv2d,slim.convolution2d_transpose,
                             slim.separable_convolution2d],
                            activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as scope:
          return scope

def backbone(inputs, final_endpoint='Conv2d_7b_1x1',
             output_stride=16, align_feature_maps=True,
             scope=None, activation_fn=tf.nn.relu):
  """Inception model from  http://arxiv.org/abs/1602.07261.

  Constructs an Inception Resnet v2 network from inputs to the given final
  endpoint. This method can construct the network up to the final inception
  block Conv2d_7b_1x1.

  Args:
    inputs: a 4D-tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
    output_stride: A scalar that specifies the requested ratio of input to
      output spatial resolution. Only supports 8 and 16.
    align_feature_maps: When true, changes all the VALID paddings in the network
      to SAME padding so that the feature maps are aligned.
    scope: Optional variable_scope.
    activation_fn: Activation function for block scopes.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
      or if the output_stride is not 8 or 16, or if the output_stride is 8 and
      we request an end point after 'PreAuxLogits'.
  """
  if output_stride != 8 and output_stride != 16:
    raise ValueError('output_stride must be 8 or 16.')

  padding = 'SAME' if align_feature_maps else 'VALID'

  end_points = {}

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'Backbone', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # Output_shape
      # 299 x 299 x 1 --> 149 x 149 x 32
      net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding,
                        scope='Conv2d_1a_3x3')
      if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points

      # 149 x 149 x 32 --> 147 x 147 x 32
      net = slim.conv2d(net, 32, 3, padding=padding,
                        scope='Conv2d_2a_3x3')
      if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
      # 147 x 147 x 32 --> 147 x 147 x 64
      net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
      if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
      # 147 x 147 x 64 --> 73 x 73 x 64
      net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                            scope='MaxPool_3a_3x3')
      if add_and_check_final('MaxPool_3a_3x3', net): return net, end_points
      # 73 x 73 x 64 --> 73 x 73 x 80
      net = slim.conv2d(net, 80, 1, padding=padding,
                        scope='Conv2d_3b_1x1')
      if add_and_check_final('Conv2d_3b_1x1', net): return net, end_points
      # 73 x 73 x 80 --> 71 x 71 x 192
      net = slim.conv2d(net, 192, 3, padding=padding,
                        scope='Conv2d_4a_3x3')
      if add_and_check_final('Conv2d_4a_3x3', net): return net, end_points
      # 71 x 71 x 192 --> 35 x 35 x 192
      net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                            scope='MaxPool_5a_3x3')
      if add_and_check_final('MaxPool_5a_3x3', net): return net, end_points

      # 35 x 35 x 192 --> 35 x 35 x 320
      with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                      scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                      scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                       scope='AvgPool_0a_3x3')
          tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                     scope='Conv2d_0b_1x1')
        net = tf.concat(
            [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)

      if add_and_check_final('Mixed_5b', net): return net, end_points
      # TODO(alemi): Register intermediate endpoints
      net = slim.repeat(net, 10, block35, scale=0.17,
                        activation_fn=activation_fn)

      # 35 x 35 x 320 --> 17 x 17 x 1088 if output_stride == 16,
      # 35 x 35 x 320 --> 33 x 33 x 1088 if output_stride == 8.
      use_atrous = output_stride == 8

      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,
                                   padding=padding,
                                   scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                      stride=1 if use_atrous else 2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

      if add_and_check_final('Mixed_6a', net): return net, end_points

      # TODO(alemi): register intermediate endpoints
      with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):
        net = slim.repeat(net, 20, block17, scale=0.10,
                          activation_fn=activation_fn)
      if add_and_check_final('PreAuxLogits', net): return net, end_points

      if output_stride == 8:
        # TODO(gpapan): Properly support output_stride for the rest of the net.
        raise ValueError('output_stride==8 is only supported up to the '
                         'PreAuxlogits end_point for now.')

      # only excuted when output_stride=16
      # 17 x 17 x 1088 --> 8 x 8 x 2080
      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                     padding=padding,
                                     scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.max_pool2d(net, 3, stride=2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
        net = tf.concat(
            [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)

      if add_and_check_final('Mixed_7a', net): return net, end_points

      # TODO(alemi): register intermediate endpoints
      net = slim.repeat(net, 9, block8, scale=0.20, activation_fn=activation_fn)
      net = block8(net, activation_fn=None)

      # 8 x 8 x 1536
      net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
      if add_and_check_final('Conv2d_7b_1x1', net): return net, end_points

    raise ValueError('final_endpoint (%s) not recognized', final_endpoint)


# def classfication_of_shape(final_endpoint,end_points=None,
#                            num_classes=1001, dropout_keep_prob=0.8,
#                            reuse=None, scope='Classfication'):
#   """Use the Inception Resnet V2 model to classifier various shapes
#
#   Args:
#     final_endpoint: a 4-D tensor of size [batch_size, height, width, channel].
#       Dimension batch_size may be undefined.
#     end_points: some key nodes saved by the backbone
#     num_classes: number of predicted classes. If 0 or None, the logits layer
#       is omitted and the input features to the logits layer (before  dropout)
#       are returned instead.
#     is_training: whether is training or not.
#     dropout_keep_prob: float, the fraction to keep before final layer.
#     reuse: whether or not the network and its variables should be reused. To be
#       able to reuse 'scope' must be given.
#     scope: Optional variable_scope.
#
#   Returns:
#     net: the output of the logits layer (if num_classes is a non-zero integer),
#       or the non-dropped-out input to the logits layer (if num_classes is 0 or
#       None).
#     end_points: the set of end_points from the inception model.
#   """
#   if not end_points:
#     end_points = {}
#
#   with tf.variable_scope(scope, reuse=reuse):
#       with tf.variable_scope('Logits'):
#         # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
#         # can be set to False to disable pooling here (as in resnet_*()).
#         kernel_size = final_endpoint.get_shape()[1:3]
#         if kernel_size.is_fully_defined():
#           net = slim.avg_pool2d(final_endpoint, kernel_size, padding='VALID',
#                                 scope='AvgPool_1a_8x8')
#         else:
#           net = tf.reduce_mean(final_endpoint, [1, 2], keep_dims=True, name='global_pool')
#         end_points['global_pool'] = net
#         if not num_classes:
#           return net, end_points
#         net = slim.flatten(net)
#         net = slim.dropout(net, dropout_keep_prob, scope='Dropout')
#         end_points['PreLogitsFlatten'] = net
#         net = slim.fully_connected(net,300,activation_fn=None)
#         net = slim.dropout(net,dropout_keep_prob,scope='Dropout')
#         end_points['Feature_list'] = net
#         logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
#         end_points['Logits'] = logits
#         end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
#
#         return logits, end_points

def Inception_resnet_v2_model(end_points, add_feature=False,
                            output_stride=16, dropout_keep_prob=0.8,
                            reuse=None, spp_concat=False, scope='Inception_resnet_v2'):
    """Similar to pix2pix, the encoder-decoder architecture is built to
        reconstruct images, but the final_endpoint from backbone is merged
         with Spatial Pyramid Pooling layer.

    Args:
      end_points: some key nodes saved by the backbone, which is required.
      add_feature: Optional. Whether to add some learned features
        from classifier of shape into every step of decoder.
      num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before  dropout)
        are returned instead.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.

    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the non-dropped-out input to the logits layer (if num_classes is 0 or
        None).

      end_points: the set of end_points from the inception model.
    """

    def upsample(net, num_outputs, resize_shape=None,
                 kernel_size=2,  method='conv2d_transpose'):
      """Upsamples the given inputs."""
      net_shape = tf.shape(net)
      height = net_shape[1]
      width = net_shape[2]
      if method == 'bilinear_upsample':
        net = slim.conv2d(net, num_outputs)
        net = tf.image.resize_images(net,resize_shape) if resize_shape else \
            tf.image.resize_images(net, [kernel_size * height, kernel_size * width])
      elif method == 'conv2d_transpose':
        net = slim.convolution2d_transpose(net, num_outputs, [4, 4], stride=kernel_size,
                                         padding='SAME', activation_fn = tf.nn.relu)
      else:
        raise ValueError('Unknown method: [%s]', method)
      return net

    def SPP(net,patch_size=[1,2,3,6],num_channel=16):
        """Spatial Pyramid Pooling """
        _,h,w,c = net.get_shape().as_list()
        SPP_map = slim.conv2d(net,num_channel*4)

        for psize in patch_size:
            h_wid = int(np.ceil(float(h) / psize))
            w_wid = int(np.ceil(float(w) / psize))
            h_pad = (h_wid * psize - psize + 1) / 2
            w_pad = (w_wid * psize - psize + 1) / 2
            # padding must be shape=[n,2],n means rank of tensor net
            padding = [[0,0],[int(np.ceil(float(h_pad)/2)),int(np.floor(float(h_pad)/2))],
                       [int(np.ceil(float(w_pad)/2)),int(np.floor(float(w_pad)/2))],[0,0]]
            padding_net = tf.pad(net,padding,mode='CONSTANT')
            patch = slim.max_pool2d(padding_net,[h_wid,w_wid],[h_wid,w_wid])
            resized_patch = upsample(patch,num_channel,resize_shape=[h,w],method='bilinear_upsample')
            SPP_map = tf.concat((SPP_map,resized_patch),axis=3)

        return SPP_map

    if add_feature:
        feature_list = end_points['Feature_list']
        N, C = feature_list.get_shape().as_list()
        feature_list = tf.reshape(feature_list,shape=[N,1,1,C],name='feature_list')
        feature_list_1536 = slim.conv2d(feature_list,1536,[1,1],stride=1)
        feature_list_1088 = slim.conv2d(feature_list,1088,[1,1],stride=1)
        feature_list_320 = slim.conv2d(feature_list,320,[1,1],stride=1)
        feature_list_192 = slim.conv2d(feature_list,192,[1,1],stride=1)
        feature_list_64 = slim.conv2d(feature_list,64,[1,1],stride=1)

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], stride=1, padding='SAME'):
            with tf.variable_scope('Decoder'):
                if output_stride ==16:
                    # 8x8x1536 --> 8x8x(8*192) if output_stride=16
                    with tf.variable_scope('SPP'):
                        net = end_points['Conv2d_7b_1x1']
                        net = SPP(net, num_channel=192)
                        net = tf.add(net,feature_list_1536) if add_feature else net
                        end_points['SPP'] = net
                    # 8x8x1536-->17x17x1088-->17x17x(1088+1088)-->17x17x1088
                    net = upsample(net, 1088)
                    net = tf.concat((net,SPP(end_points['PreAuxLogits'],num_channel=136)),axis=3) \
                        if spp_concat else tf.concat((net,end_points['PreAuxLogits']),axis=3)
                    net = slim.conv2d(net,1088,kernel_size=[1,1])
                    net = tf.add(net, feature_list_1088) if add_feature else net
                    net = slim.dropout(net, dropout_keep_prob)
                    # 17x17x1088-->35x35x320-->35x35x(320+320)-->35x35x320
                    net = upsample(net, 320)
                    net = tf.concat((net,SPP(end_points['Mixed_5b'],num_channel=40)),axis=3) \
                        if spp_concat else tf.concat((net,end_points['Mixed_5b']),axis=3)
                    net = slim.conv2d(net, 320,kernel_size=[1,1])
                    net = tf.add(net, feature_list_320) if add_feature else net
                    net = slim.dropout(net, dropout_keep_prob)
                else:
                    with tf.variable_scope('SPP'):
                        net = end_points['PreAuxLogits']
                        net = SPP(net, num_channel=136)
                        net = tf.add(net,feature_list_1088) if add_feature else net
                        end_points['SPP'] = net
                # 35x35x320(35x35x1088)-->71x71x192-->71x71x(192+192)-->71x71x192
                net = upsample(net, 192)
                net = tf.concat((net, SPP(end_points['Conv2d_4a_3x3'], num_channel=24)), axis=3) \
                    if spp_concat else tf.concat((net, end_points['Conv2d_4a_3x3']), axis=3)
                net = slim.conv2d(net, 192, kernel_size=[1, 1])
                net = tf.add(net, feature_list_192) if add_feature else net
                net = slim.dropout(net, dropout_keep_prob)
                end_points['decoder_2'] = net
                # 71x71x192-->147x147x64--147x147x(64+64)-->147x147x64
                net = upsample(net, 64)
                net = tf.concat((net, SPP(end_points['Conv2d_2b_3x3'], num_channel=8)), axis=3) \
                    if spp_concat else tf.concat((net, end_points['Conv2d_2b_3x3']), axis=3)
                net = slim.conv2d(net, 32, kernel_size=[1, 1])
                net = tf.add(net, feature_list_64) if add_feature else net
                end_points['decoder_1'] = net
                # 147x147x64-->299x299x1
                net = upsample(net,10)
                net = slim.conv2d(net, 1, kernel_size=[1,1],activation_fn=tf.nn.sigmoid)
                end_points['output'] = net

    return net,end_points


def vanilla_CNN_model(input,scope=None):
    end_points={}
    with tf.variable_scope(scope,'Vanilla_CNN',[input]):
        with slim.arg_scope([slim.conv2d,slim.separable_convolution2d],
                            kernel_size=3 , stride=1):
            net = slim.conv2d(input, 32)
            net = slim.conv2d(net,64)
            net = slim.conv2d(net, 128)
            end_points['Conv2d_128a'] = net
            net = slim.separable_convolution2d(net,256,depth_multiplier=4)
            net = slim.separable_convolution2d(net,256,depth_multiplier=4)
            net = slim.separable_convolution2d(net,256,depth_multiplier=4)
            net = slim.separable_convolution2d(net,256,depth_multiplier=4)
            end_points['Separable_conv2d_256'] = net
            net = slim.conv2d(net, 128)
            end_points['Conv2d_128b'] = net
            net = slim.conv2d(net,64)
            net = slim.conv2d(net,32)
            net = slim.conv2d(net,1,activation_fn = tf.nn.sigmoid)
            end_points['output'] = net
        return net, end_points

def discriminator(input,conditioning,patch_level = True,scope='Discriminator'):
    """
    Conditional GAN. Original image and target or output
    are concated to feed into discriminator network.
    :param input: ground truth or output from generator network.
    :param conditioning: The generator_input, a 4D tensor [N,H,W,1] or [N,H,W,2].
    :param patch_level: Whether is patch level probability prediction.
    :param scope: The scope of discriminator network.
    :return: A patch-level score map
    """
    with tf.variable_scope(scope):
        net = tf.concat((input,conditioning),axis=3)
        with slim.arg_scope([slim.conv2d],kernel_size=[4,4],
                            stride=[2,2],activation_fn=tf.nn.relu):
            net = slim.conv2d(net, 64)
            net = slim.conv2d(net, 128)
            net = slim.conv2d(net, 256)
            net = slim.conv2d(net, 512)
            net = slim.conv2d(net, 1)
            if not patch_level:
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1, activation_fn=None)
    return net
