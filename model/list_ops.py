from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import tensorflow as tf
import numpy as np

from . import model_helper

import pdb

class cudnnlstm_bias_initializer(tf.keras.initializers.Initializer):
  """Initializer that can be used with "tf.contrib.cudnn_rnn.CudnnLSTM"
  to initialize the bias such that "forget_bias" could be initialized
  as 1. while others are initialized as 0.
  """
  def __init__(self,
               unit_forget_bias=True,
               dtype=tf.float32):  
    self.dtype = dtype
    self.unit_forget_bias = unit_forget_bias
    self.num_called = 0

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype

    if self.unit_forget_bias:
      if self.num_called % 4 == 1:
        initialized = tf.constant(0.5, dtype=dtype, shape=shape, verify_shape=False)
      else:
        initialized = tf.constant(0.0, dtype=dtype, shape=shape, verify_shape=False)
      self.num_called += 1
    else:
      initialized = tf.constant(0.0, dtype=dtype, shape=shape, verify_shape=False)
    
    return initialized
  
  def get_config(self):
    return {"num_called": self.num_called, "dtype": self.dtype.name}

class cudnnlstm_kernel_initializer(tf.keras.initializers.Initializer):
  """Initializer that can be used with "tf.contrib.cudnn_rnn.CudnnLSTM"
  to initialize the "input_kernel" by glorot_uniform initializer,
  and the "recurrent_kernel" by orthogonal initializer.

  Args:
    seed: A Python integer. Used to create random seeds.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  """
  def __init__(self,
               seed=None,
               dtype=tf.float32):
    self.seed = seed
    self.dtype = dtype
    self.num_called = 0
  
  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    
    if self.num_called < 4:
      # for input_kernel, use glorot uniform initializer.
      fan_in = shape[1]
      fan_out = shape[0]

      if fan_in < 1 or fan_out < 1:
        raise ValueError("Both fan_in and fan_out must be >= 1"
                         "fan_in: {}, fan_out: {}".format(fan_in, fan_out))

      limit = math.sqrt(6.0 / (fan_in + fan_out))
      initialized = tf.random_uniform(shape, -limit, limit, dtype, seed=self.seed)
      self.num_called += 1
    
    else:
      # for recurrent_kernel, use orthogonal initializer.
      if len(shape) < 2:
        raise ValueError("The tensor to initialize must be "
                         "at least two-dimensional")
      num_rows = 1
      for dim in shape[:-1]:
        num_rows *= dim
      num_cols = shape[-1]
      flat_shape = (num_cols, num_rows) if num_rows < num_cols else (num_rows,
                                                                     num_cols)

      # Generate a random matrix
      a = tf.random_normal(flat_shape, dtype=dtype, seed=self.seed)
      # Compute the qr factorization
      q, r = tf.linalg.qr(a, full_matrices=False)
      # Make Q uniform
      d = tf.linalg.diag_part(r)
      q *= tf.math.sign(d)
      
      if num_rows < num_cols:
        initialized = tf.linalg.matrix_transpose(q)
      else:
        initialized = q
      self.num_called += 1
    
    return initialized
  
  def get_config(self):
    return {"num_called": self.num_called, "seed": self.seed, "dtype": self.dtype.name}

def list_placeholder(len_list,
                     shape,
                     dtype=tf.float32,
                     new_scope=True):
  """ Make list of placeholders
  Args:
      len_list:  Length of the list_placeholder (i.e., number of placeholders or GPUs).
      shape:     Shape excluding the batch dimension
      name:      Name to this layer
      dtype:     Dtype of the placeholder. Defaults to tf.float32
  
  Return:
      list_output: Output placeholders.
  """
  list_output = []
  
  for gpu_idx in range(len_list):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.placeholder(dtype=dtype, shape=shape))
      
  return list_output

def list_get_dims(list_input, axis):
  assert type(list_input) == list
  list_output = []
  for inputs in list_input:
    list_output.append(tf.shape(inputs)[axis])
  
  return list_output

def list_expand_dims(list_input,
                     axis=-1,
                     new_scope=True):
  assert type(list_input) == list
  list_output = []
  
  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.expand_dims(inputs, axis=axis))
  
  return list_output

def list_squeeze(list_input,
                 axis=-1,
                 new_scope=True):
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.squeeze(inputs, axis=axis))
  
  return list_output

def list_reduce_sum(list_input,
                    axis=None,
                    new_scope=True):
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.reduce_sum(inputs, axis=axis))
  
  return list_output

def list_cast(list_input,
              dtype=tf.float32,
              new_scope=True):
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.cast(inputs, dtype=dtype))
  
  return list_output

def list_maxpool2d(list_input,
                   pool_size=(2, 2),
                   strides=None,
                   padding='valid',
                   new_scope=True):
  assert type(list_input) == list
  if strides is None:
    strides = pool_size
  
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      maxpool2d = tf.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
      list_output.append(maxpool2d(inputs))
  
  return list_output

def list_flatten(list_input,
                 new_scope=True):
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      flatten = tf.layers.Flatten()
      list_output.append(flatten(inputs))
  
  return list_output

def list_reshape(list_input, target_shape, new_scope=True):
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.reshape(inputs, target_shape))
  
  return list_output

def list_transpose(list_input, perm, new_scope=True):
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.transpose(inputs, perm))
  
  return list_output

def list_zeros_like(list_input, new_scope=True):
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.zeros_like(inputs))
  
  return list_output

def list_l2(list_y_true, list_y_pred, new_scope=True):
  assert type(list_y_true) == list
  assert type(list_y_pred) == list
  list_output = []

  for gpu_idx, (y_true, y_pred, batch_size) in enumerate(zip(list_y_true, list_y_pred, list_batch_size)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(0.5 * tf.square(y_true - y_pred))

  return list_output

def list_weighted_smooth_l1(list_y_true, list_y_pred, new_scope=True):
  assert type(list_y_true) == list
  assert type(list_y_pred) == list
  list_output = []

  for gpu_idx, (y_true, y_pred) in enumerate(zip(list_y_true, list_y_pred)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      l2_part = 0.5 * tf.square(y_true - y_pred)
      l1_part = tf.abs(y_true - y_pred) - 0.5
      loss = tf.where(l1_part < 1,
                      x=l2_part,
                      y=l1_part)
      
      list_output.append(loss)
  
  return list_output

def list_sigmoid_cross_entropy(list_labels, list_logits, new_scope=True):
  assert type(list_labels) == list
  assert type(list_logits) == list
  list_output = []

  for gpu_idx, (labels, logits) in enumerate(zip(list_labels, list_logits)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
      list_output.append(loss)
  
  return list_output

def list_boolean_mask(list_tensor, list_mask, axis=None, new_scope=True):
  assert type(list_tensor) == list
  assert type(list_mask) == list
  list_output = []

  for gpu_idx, (tensor, mask) in enumerate(zip(list_tensor, list_mask)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      outputs = tf.boolean_mask(tensor=tensor, mask=mask, axis=axis)
      list_output.append(outputs)

  return list_output

def list_less(list_x, list_y, new_scope=True):
  assert type(list_x) == list
  assert type(list_y) == list
  list_output = []

  for gpu_idx, (x, y) in enumerate(zip(list_x, list_y)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      outputs = tf.math.less(x=x, y=y)
      list_output.append(outputs)

  return list_output

def list_greater(list_x, list_y, new_scope=True):
  assert type(list_x) == list
  assert type(list_y) == list
  list_output = []

  for gpu_idx, (x, y) in enumerate(zip(list_x, list_y)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      outputs = tf.math.greater(x=x, y=y)
      list_output.append(outputs)

  return list_output

def list_where(list_condition, list_x, list_y, new_scope=True):
  assert type(list_condition) == list
  assert type(list_x) == list
  assert type(list_y) == list
  list_output = []

  for gpu_idx, (condition, x, y) in enumerate(zip(list_condition, list_x, list_y)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      outputs = tf.where(condition=condition, x=x, y=y)
      list_output.append(outputs)

  return list_output

def list_matmul(list_x, list_y, new_scope=True):
  assert type(list_x) == list
  assert type(list_y) == list
  list_output = []

  for gpu_idx, (x, y) in enumerate(zip(list_x, list_y)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      outputs = tf.linalg.matmul(x, y)
      list_output.append(outputs)

  return list_output

def list_concat2(list_tensors, axis=-1, new_scope=True):
  assert type(list_tensors) == list
  list_output = []

  for gpu_idx, tensors in enumerate(list_tensors):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.concat(list(tensors), axis=axis))
  
  return list_output

def list_add_n(list_tensors, new_scope=True):
  assert type(list_tensors) == list
  list_output = []

  for gpu_idx, tensors in enumerate(list_tensors):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.add_n(list(tensors)))
  
  return list_output

def list_tile(list_input, multiples, new_scope=True):
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.tile(inputs, multiples))
  
  return list_output

def list_concat(list_x, list_y, axis=-1, new_scope=True):
  assert type(list_x) == list
  assert type(list_y) == list
  list_output = []

  for gpu_idx, tensors in enumerate(zip(list_x, list_y)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.concat(list(tensors), axis=axis))
  
  return list_output

def list_subtract(list_x, list_y, axis=-1, new_scope=True):
  assert type(list_x) == list
  assert type(list_y) == list
  list_output = []

  for gpu_idx, (x, y) in enumerate(zip(list_x, list_y)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(x - y)
  
  return list_output

def list_add(list_x, list_y, axis=-1, new_scope=True):
  assert type(list_x) == list
  assert type(list_y) == list
  list_output = []

  for gpu_idx, (x, y) in enumerate(zip(list_x, list_y)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(x + y)
  
  return list_output

def list_divide(list_x, list_y, new_scope=True):
  assert type(list_x) == list
  assert type(list_y) == list
  list_output = []

  for gpu_idx, (x, y) in enumerate(zip(list_x, list_y)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.divide(x, y))
  
  return list_output

def list_clip_by_global_norm(list_tensors, max_norm, new_scope=True):
  assert type(list_tensors) == list
  list_clipped_tensors = []
  list_global_norm = []

  for gpu_idx, tensors in enumerate(list_tensors):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      clipped_tensors, global_norm = tf.clip_by_global_norm(tensors, max_norm)
      list_clipped_tensors.append(clipped_tensors)
      list_global_norm.append(global_norm)
  
  return list_clipped_tensors, list_global_norm

def list_global_norm(list_tensors, new_scope=True):
  assert type(list_tensors) == list
  list_global_norm = []

  for gpu_idx, tensors in enumerate(list_tensors):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      global_norm = tf.global_norm(tensors)
      list_global_norm.append(global_norm)
  
  return list_global_norm

def list_sigmoid(list_input, new_scope=True):
  """ sigmoid...
  Args:
      list_input:  Length K list of tensors, [[nD tensor], [nD tensor], ...].
  Return:
      list_output: Length K list of rectified tensors.
  """
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.math.sigmoid(inputs))
  
  return list_output

def list_relu(list_input, new_scope=True):
  """ REctified Linear Unit...
  Args:
      list_input:  Length K list of tensors, [[nD tensor], [nD tensor], ...].
  Return:
      list_output: Length K list of rectified tensors.
  """
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx) if new_scope else tf.get_default_graph().get_name_scope() + "/tower_{:d}/".format(gpu_idx)):
      list_output.append(tf.nn.relu(inputs))
  
  return list_output

def list_cudnnlstm(list_input,
                   num_layers,
                   num_units,
                   list_initial_state=None,
                   final_output=True,
                   seed=None):
  """ LSTM operation...
  Args:
      list_input:  Length K list of tensors, [[BxTxC], [BxTxC], ...] where T corresponds to sequence length.
      num_layers: the number of layers for the RNN model.
      num_units: the number of units within the RNN model.
      seed: random seed

  Return:
      list_output: Length K list of output tensors.
  """
  assert type(list_input) == list
  if seed is None:
    seed = int(time.time())

  if list_initial_state is None:
    list_initial_state = [None for _ in range(len(list_input))]

  list_output = []
  list_states = []

  for gpu_idx, (inputs, initial_states) in enumerate(zip(list_input, list_initial_state)):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope("tower_{:d}".format(gpu_idx)):
      lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, num_units, kernel_initializer=cudnnlstm_kernel_initializer(seed=seed), bias_initializer=cudnnlstm_bias_initializer(), seed=seed)
      outputs, states = lstm(inputs, initial_state=initial_states)
      if final_output:
        outputs = outputs[-1, :, :]
      tf.add_to_collection('lstm_variables_tower_{:d}'.format(gpu_idx), lstm.weights[0])
      list_output.append(outputs)
      list_states.append(states)

  return list_output, list_states

def list_dense(list_input,
               units,
               activation=None,
               kernel_regularizer=None,
               seed=None):
  """ Dense operation...
  Args:
      list_input:  Length K list of tensors, [[nD tensor], [nD tensor], ...].
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.

  Return:
      list_output: Length K list of output tensors.
  """
  assert type(list_input) == list
  if seed is None:
    seed = int(time.time())
  
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope("tower_{:d}".format(gpu_idx)):
      layer = tf.layers.Dense(units=units,
                              activation=activation,
                              kernel_initializer=tf.initializers.glorot_uniform(seed=seed),
                              bias_initializer=tf.initializers.zeros(),
                              kernel_regularizer=kernel_regularizer)

      list_output.append(layer(inputs))
      tf.add_to_collection('training_variables', layer.weights[0])
      tf.add_to_collection('training_variables', layer.weights[1])

  return list_output

def list_dense2(list_input,
                units,
                kernel_initializer,
                bias_initializer,
                activation=None,
                kernel_regularizer=None):
  """ Dense operation...
  Args:
      list_input:  Length K list of tensors, [[nD tensor], [nD tensor], ...].
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.

  Return:
      list_output: Length K list of output tensors.
  """
  assert type(list_input) == list
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope("tower_{:d}".format(gpu_idx)):
      layer = tf.layers.Dense(units=units,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer)
      list_output.append(layer(inputs))
      tf.add_to_collection('training_variables', layer.weights[0])
      tf.add_to_collection('training_variables', layer.weights[1])

  return list_output

def list_conv1d(list_input,
                filters,
                kernel_size,
                strides=1,
                padding='valid',
                dilation_rate=1,
                activation=None,
                seed=None,
                kernel_regularizer=None):
  """ Conv2D operation...
  Args:
      list_input:  Length K list of tensors, [[BTC], [BTC], ...].
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer,
          specifying the length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer.
          Specifying any stride value != 1 is incompatible with
          specifying any dilation_rate value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      dilation_rate:  An integer or tuple/list of a single integer,
          specifying the dilation rate to use for dilated convolution.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      seed: random seed to use for kernel/bias initializer.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.

  Return:
      list_output: Length K list of output tensors.
  """
  assert type(list_input) == list
  if seed is None:
    seed = int(time.time())
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope("tower_{:d}".format(gpu_idx)):
      layer = tf.layers.Conv1D(filters,
                               kernel_size,
                               strides=strides,
                               padding=padding,
                               dilation_rate=dilation_rate,
                               activation=activation,
                               kernel_initializer=tf.initializers.glorot_uniform(seed=seed),
                               bias_initializer=tf.initializers.zeros(),
                               kernel_regularizer=kernel_regularizer)
      list_output.append(layer(inputs))
      tf.add_to_collection('training_variables', layer.weights[0])
      tf.add_to_collection('training_variables', layer.weights[1])

  return list_output

def list_conv2d(list_input,
                filters,
                kernel_size,
                strides=(1, 1),
                padding='valid',
                activation=None,
                seed=None,
                kernel_regularizer=None):
  """ Conv2D operation...
  Args:
      list_input:  Length K list of tensors, [[BHWC], [BHWC], ...].
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
          width and height of the 2D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      kernel_initializer: Initializer for the `kernel` weights matrix.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.

  Return:
      list_output: Length K list of output tensors.
  """
  assert type(list_input) == list
  if seed is None:
    seed = int(time.time())
  list_output = []

  for gpu_idx, inputs in enumerate(list_input):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope("tower_{:d}".format(gpu_idx)):
      layer = tf.layers.Conv2D(filters,
                                kernel_size,
                                strides=strides,
                                padding=padding,
                                activation=activation,
                                kernel_initializer=tf.initializers.glorot_uniform(seed=seed),
                                bias_initializer=tf.initializers.zeros(),
                                kernel_regularizer=kernel_regularizer)
      list_output.append(layer(inputs))
      tf.add_to_collection('training_variables', layer.weights[0])
      tf.add_to_collection('training_variables', layer.weights[1])

  return list_output

def list_batch_norm_template(list_input, is_training, moments_dims, bn_decay):
  """ Batch normalization...
  
  Args:
      list_input:    Length K list of tensors, [[BC or BHWC], [BC or BHWC], ...].
      is_training:   boolean tf.Varialbe, true indicates training phase
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      list_output:   Length K list of batch-normalized tensors
  """
  assert type(list_input) == list
  list_output = []

  means = []
  with tf.name_scope("sub_batch_mean"):
    for gpu_idx, inputs in enumerate(list_input):
      with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}_mean".format(gpu_idx)):
        means.append([tf.reduce_mean(inputs, moments_dims)])
  
  # with tf.name_scope("all_reduce_mean"):
  all_reduced_means = model_helper.allreduce_tensors(means, average=True)
  
  variances = []
  with tf.name_scope("sub_batch_variance"):
    for gpu_idx, inputs in enumerate(list_input):
      with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}_variance".format(gpu_idx)):
        variances.append([tf.reduce_mean(tf.squared_difference(inputs, tf.stop_gradient(all_reduced_means[gpu_idx])),
                                        moments_dims)])
  
  # with tf.name_scope("all_reduce_variance"):
  all_reduced_variances = model_helper.allreduce_tensors(variances, average=True)
  
  decay = bn_decay if bn_decay is not None else 0.9

  with tf.variable_scope("normalize"):
    def mean_var_with_update(ema, mean, variance):
      ema_apply_op = tf.cond(tf.convert_to_tensor(is_training),
                             lambda: ema.apply([mean, variance]),
                             lambda: tf.no_op())
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(mean), tf.identity(variance)

    for gpu_idx, inputs in enumerate(list_input):
      with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope("tower_{:d}".format(gpu_idx)):
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        mean = all_reduced_means[gpu_idx][0]
        variance = all_reduced_variances[gpu_idx][0]

        mean, variance = tf.cond(tf.convert_to_tensor(is_training),
                                 lambda: mean_var_with_update(ema, mean, variance),
                                 lambda: (ema.average(mean), ema.average(variance)))
        
        num_channels = inputs.get_shape()[-1].value
        beta = tf.get_variable(name='beta',
                              shape=[num_channels],
                              initializer=tf.zeros_initializer(),
                              trainable=True)
        tf.add_to_collection('training_variables', beta)
        gamma = tf.get_variable(name='gamma',
                                shape=[num_channels],
                                initializer=tf.ones_initializer(),
                                trainable=True)
        tf.add_to_collection('training_variables', gamma)
        
        normed = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 1e-3)

        list_output.append(normed)

  return list_output

def list_conv1d_with_bn(list_input,
                        filters,
                        kernel_size,
                        is_training,
                        bn_decay,
                        strides=1,
                        padding='valid',
                        dilation_rate=1,
                        kernel_regularizer=None,
                        seed=None):
  assert type(list_input) == list
  with tf.variable_scope('conv'):
    net = list_conv1d(list_input, filters, kernel_size,
                      strides=strides, padding=padding, dilation_rate=dilation_rate,
                      seed=seed, kernel_regularizer=kernel_regularizer)
  
  with tf.variable_scope('bn'):
    net = list_batch_norm_for_conv1d(net, is_training, bn_decay)

  with tf.variable_scope('activation'):
    net = list_relu(net)
  
  return net

def list_conv2d_with_bn(list_input,
                        filters,
                        kernel_size,
                        is_training,
                        bn_decay,
                        kernel_regularizer=None,
                        strides=(1, 1),
                        padding='valid',
                        seed=None):
  assert type(list_input) == list
  with tf.variable_scope('conv'):
    net = list_conv2d(list_input, filters, kernel_size,
                      strides=strides, padding=padding, seed=seed,
                      kernel_regularizer=kernel_regularizer)

  with tf.variable_scope('bn'):
    net = list_batch_norm_for_conv2d(net, is_training, bn_decay)

  with tf.variable_scope('activation'):
    net = list_relu(net)
  
  return net
  
def list_dense_with_bn(list_input,
                       units,
                       is_training,
                       bn_decay,
                       kernel_regularizer=None,
                       seed=None,
                       activation='relu'):
  assert type(list_input) == list
  with tf.variable_scope('dense'):
    net = list_dense(list_input, units, kernel_regularizer=kernel_regularizer, seed=seed)
  
  with tf.variable_scope('bn'):
    net = list_batch_norm_for_fc(net, is_training, bn_decay)

  if activation == 'relu':
    with tf.variable_scope('activation'):
      net = list_relu(net)
  
  elif activation =='linear':
    pass

  else:
    raise ValueError("Unsupported actiovation {}".format(activation))

  
  return net

def list_batch_norm_for_fc(list_input, is_training, bn_decay):
  """ Batch normalization on FC layer activation...
  
  Args:
      list_input:  Length K list of tensors, [[BxC], [BxC], ...] 
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
  Return:
      list_output: Length K list of batch-normalized tensors
  """
  assert type(list_input) == list
  return list_batch_norm_template(list_input, is_training, [0,], bn_decay)

def list_batch_norm_for_conv1d(list_input, is_training, bn_decay):
  """ Batch normalization on Conv2D layer activation...
  
  Args:
      list_input:  Length K list of tensors, [[BHWC], [BHWC], ...]
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
  Return:
      list_output: Length K list of batch-normalized tensors
  """
  assert type(list_input) == list
  return list_batch_norm_template(list_input, is_training, [0,1], bn_decay)

def list_batch_norm_for_conv2d(list_input, is_training, bn_decay):
  """ Batch normalization on Conv2D layer activation...
  
  Args:
      list_input:  Length K list of tensors, [[BHWC], [BHWC], ...]
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
  Return:
      list_output: Length K list of batch-normalized tensors
  """
  assert type(list_input) == list
  return list_batch_norm_template(list_input, is_training, [0,1,2], bn_decay)