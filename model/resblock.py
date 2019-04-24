from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import model_helper

import pdb

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=-1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
      center=True, scale=True, training=training, fused=True)

def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  padded_inputs = tf.pad(tensor=inputs,
                         paddings=[[0, 0], [pad_beg, pad_end],
                                   [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, reg_func):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False, kernel_regularizer=reg_func,
      kernel_initializer=tf.variance_scaling_initializer())

def conv2d_fixed_padding_with_bias_and_activation(inputs, filters, kernel_size, strides, activation, reg_func):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), activation=activation, use_bias=True, kernel_regularizer=reg_func,
      kernel_initializer=tf.variance_scaling_initializer())

def _bottleneck_block_without_bn(inputs, filters, training, projection_shortcut, strides, reg_func):
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding_with_bias_and_activation(
      inputs=inputs, filters=filters, kernel_size=1, strides=1, activation=tf.nn.relu, reg_func=reg_func)

  inputs = conv2d_fixed_padding_with_bias_and_activation(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides, activation=tf.nn.relu, reg_func=reg_func)

  inputs = conv2d_fixed_padding_with_bias_and_activation(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, activation=None, reg_func=reg_func)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs

def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, reg_func):
  """A single block for ResNet v2, with a bottleneck.
  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1, reg_func=reg_func)

  inputs = batch_norm(inputs, training)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides, reg_func=reg_func)

  inputs = batch_norm(inputs, training)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, reg_func=reg_func)

  return inputs + shortcut

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, reg_func, name):
  """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    reg_func: regularization function
    name: A string name for the tensor output of the block layer.
  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, reg_func=reg_func)

  # Only the first block per block_layer uses projection_shortcut and strides
  with tf.variable_scope("block_0"):
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides, reg_func)

  for b_idx in range(1, blocks):
    with tf.variable_scope("block_{}".format(b_idx)):
      inputs = block_fn(inputs, filters, training, None, 1, reg_func)

  return tf.identity(inputs, name)

def simple_dense(inputs, num_units, do_batch_norm, training, activation, reg_func):
  if do_batch_norm:
    inputs = tf.layers.dense(
        inputs, units=num_units, activation=None)
    inputs = batch_norm(inputs, training)
    inputs = activation(inputs)

  else:
    inputs = tf.layers.dense(
        inputs, units=num_units, activation=activation, kernel_regularizer=reg_func)

  return inputs