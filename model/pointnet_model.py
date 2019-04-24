from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import numpy as np

import tensorflow as tf

from .  import model_helper
from .  import layers
from .  import model
from .utils import iterator_utils
from .utils import misc_utils as utils

import pdb

__all__ = ["Model"]

class Model(model.BaseModel):
  """Simple pointnet based lidar fusion network.
  This class implements a single-layer lstm as trajectory encoder,
  PointNet and MLP ans lidar feature extraction and merginging,
  and a single-stack lstm or MLP as trajectory decoder.
  """
  def build_graph(self, hparams, scope=None):
    with tf.variable_scope("Model"):
      utils.print_out("# Creating {} graph ...".format(self.mode))
      
      is_training = (self.mode == tf.estimator.ModeKeys.TRAIN)
      
      # Encoder
      list_encoder_output, list_encoder_state = self._build_encoder(hparams, is_training)
      self.list_encoder_output = list_encoder_output

      # PointNet
      list_global_feature = self._build_pointnet(hparams, is_training)
      self.list_global_feature = list_global_feature

      # Merge Feature
      list_merged_state = self._build_merge(hparams, list_encoder_output, list_global_feature, is_training)
      self.list_merged_state = list_merged_state

      # Decoder
      list_regression, _ = self._build_decoder(hparams, list_merged_state, None, is_training)

      # Loss
      if self.mode != tf.estimator.ModeKeys.PREDICT:
        list_loss = self._compute_loss(list_regression, hparams)
      else:
        list_loss = [tf.constant(0.0) for _ in range(self.num_gpu)]

    return list_regression, list_loss

  def _build_input_projection(self, hparams, list_input, is_training):
    reg_func = tf.contrib.layers.l2_regularizer(hparams.weight_decay_factor)
    net = list_input
    if hparams.input_projector_type == "fc":
      for layer_idx, units in enumerate(hparams.fc_input_projector_units):
        with tf.variable_scope("fc{}".format(layer_idx + 1)):
          net = layers.list_dense(
              net, units=units, activation=tf.nn.relu, seed=self.random_seed, kernel_regularizer=reg_func)

    elif hparams.input_projector_type == "cnn":
      for layer_idx, (filters, kernel_size) in enumerate(zip(hparams.cnn_input_projector_filters, hparams.cnn_input_projector_kernels)):
        with tf.variable_scope("conv{}".format(layer_idx + 1)):
          net = layers.list_conv1d_with_bn(net,
                                           filters=filters,
                                           kernel_size=kernel_size,
                                           is_training=is_training,
                                           bn_decay=self.bn_decay,
                                           padding='same',
                                           seed=self.random_seed)
    else:
      raise ValueError("Unknown projector {:s}.".format(hparams.input_projector_type))
    
    return net

  def _build_rnn_encoder(self, hparams, list_input, is_training):
    if hparams.rnn_encoder_type == "cudnn_lstm":
      net, state = layers.list_cudnnlstm(list_input, num_layers=hparams.rnn_encoder_layers, num_units=hparams.rnn_encoder_units, seed=self.random_seed)
    else:
      raise ValueError("Unknown rnn {:s}.".format(hparams.rnn_encoder_type))
    
    return net, state
  
  def _build_cnn_encoder(self, hparams, list_input, is_training):
    net = list_input
    for layer_idx, (filters, kernel_size, dilation_rate) in enumerate(zip(hparams.cnn_encoder_filters, hparams.cnn_encoder_kernels, hparams.cnn_encoder_dilation_rates)):
      with tf.variable_scope("conv{}".format(layer_idx + 1)):
        net = layers.list_conv1d_with_bn(net,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         is_training=is_training,
                                         bn_decay=self.bn_decay,
                                         padding='valid',
                                         dilation_rate=dilation_rate,
                                         seed=self.random_seed)
    with tf.name_scope("flatten"):
      net = layers.list_flatten(net)
    
    return net

  def _build_encoder(self, hparams, is_training):
    """Build encoder from source."""
    with tf.variable_scope("trajectory_encoder"):
      with tf.name_scope("source_placeholder"):
        input_phs = layers.list_placeholder(self.num_gpu, (None, self.input_length, self.input_dims), tf.float32)
      for ph in input_phs:
        tf.add_to_collection('placeholder', ph)
      
      if hparams.encoder_type == "rnn":
        net = input_phs
        with tf.variable_scope("projection"):
          net = self._build_input_projection(hparams, net, is_training)

        with tf.name_scope("batch_time_transpose"):
          net = layers.list_transpose(net, perm=[1, 0, 2])

        with tf.variable_scope("rnn"):
          net, state = self._build_rnn_encoder(hparams, net, is_training)

        if hparams.relu_reconfiguration:
          with tf.variable_scope("reconfiguration"):
            net = layers.list_dense_with_bn(net,
                                            hparams.cnn_input_projector_filters[-1],
                                            is_training,
                                            self.bn_decay,
                                            seed=self.random_seed)

      elif hparams.encoder_type == "cnn":
        net = self._build_cnn_encoder(hparams, input_phs, is_training)
        state = None
      
      else:
        raise ValueError("Unknown encoder type {:s}.".format(hparams.encoder_type))

    return net, state

  def _build_fc_decoder(self, hparams, list_input, is_training):
    net = list_input
    for layer_idx, units in enumerate(hparams.fc_decoder_units):
      with tf.variable_scope("fc{}".format(layer_idx + 1)):
        if layer_idx < len(hparams.fc_decoder_units) - 1:
          net = layers.list_dense_with_bn(net,
                                          units=units,
                                          is_training=is_training,
                                          bn_decay=self.bn_decay,
                                          seed=self.random_seed)
        else:
          net = layers.list_dense(net,
                           units=units,
                           seed=self.random_seed)
    
    with tf.name_scope("expand_dims"):
      net = layers.list_expand_dims(net, axis=1)
    
    return net
  
  def _build_rnn_decoder(self, hparams, list_input, initial_states, is_training):
    if hparams.rnn_decoder_type == "cudnn_lstm":
      net, state = layers.list_cudnnlstm(list_input, num_layers=hparams.rnn_decoder_layers, num_units=hparams.rnn_decoder_units, list_initial_state=initial_states, final_output=False, seed=self.random_seed)
    else:
      raise ValueError("Unknown rnn {:s}.".format(hparams.rnn_encoder_type))
    
    return net, state
  
  def _build_output_projection(self, hparams, list_input, is_training):
    reg_func = tf.contrib.layers.l2_regularizer(hparams.weight_decay_factor)
    net = list_input
    for layer_idx, units in enumerate(hparams.output_projector_units):
      with tf.variable_scope("fc{}".format(layer_idx + 1)):
        if layer_idx < len(hparams.output_projector_units) - 1:
          net = layers.list_dense(net,
                                  units=units,
                                  activation=tf.nn.relu,
                                  seed=self.random_seed,
                                  kernel_regularizer=reg_func)
        else:
          net = layers.list_dense(net,
                                  units=units,
                                  seed=self.random_seed,
                                  kernel_regularizer=reg_func)
    
    return net
    
  def _make_initial_states(self, hparams, list_input):
    reg_func = tf.contrib.layers.l2_regularizer(hparams.weight_decay_factor)
    with tf.variable_scope("h_state"):
      with tf.variable_scope("fc"):
        h = layers.list_dense(list_input,
                              units=hparams.rnn_decoder_units * hparams.rnn_decoder_layers,
                              kernel_regularizer=reg_func,
                              seed=self.random_seed)
      
      with tf.name_scope("reshape"):
        h = layers.list_reshape(h, target_shape=[-1, hparams.rnn_decoder_layers, hparams.rnn_decoder_units])
        
      with tf.name_scope("transpose"):
        h = layers.list_transpose(h, perm=[1, 0, 2])
    
    with tf.name_scope("cell_state"):
      c = layers.list_zeros_like(h)
    
    list_lstm_tuple = [(h_, c_) for (h_, c_) in zip(h, c)]
    return list_lstm_tuple

  def _build_decoder(self, hparams, inputs, initial_state, is_training):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      inputs: The input for the decoder
      initial_state: The initial_state for the decoder (used for rnn decoder).
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        regression: size [batch_size, time, regression_dim].
    """
    ## Decoder.
    with tf.variable_scope("trajectory_decoder"):
      if hparams.decoder_type == "fc":
        regression = self._build_fc_decoder(hparams, inputs, is_training)
        final_states = None
      
      elif hparams.decoder_type == "rnn":
        list_dummy_input = []
        with tf.name_scope("dummy_input"):
          for gpu_idx in range(self.num_gpu):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx)):
              list_dummy_input.append(tf.zeros(tf.stack([self.target_length, self.batch_size[gpu_idx], 1])))
        
        with tf.variable_scope("rnn"):
          if hparams.encoder_type == "cnn":
            with tf.variable_scope("rnn_initial_state"):
              initial_state = self._make_initial_states(hparams, inputs)

          net, final_states = self._build_rnn_decoder(hparams, list_dummy_input, initial_state, is_training)

        with tf.name_scope("time_batch_transpose"):
          net = layers.list_transpose(net, perm=[1, 0, 2])
        
        with tf.variable_scope("projection"):
          regression = self._build_output_projection(hparams, net, is_training)

      else:
        raise ValueError("Unknown decoder type {:s}.".format(hparams.decoder_type))

    return regression, final_states

  def _build_input_projection(self, hparams, list_input, is_training):
    reg_func = tf.contrib.layers.l2_regularizer(hparams.weight_decay_factor)
    net = list_input
    if hparams.input_projector_type == "fc":
      for layer_idx, units in enumerate(hparams.fc_input_projector_units):
        with tf.variable_scope("fc{}".format(layer_idx + 1)):
          net = layers.list_dense(
              net, units=units, activation=tf.nn.relu, seed=self.random_seed, kernel_regularizer=reg_func)

    elif hparams.input_projector_type == "cnn":
      for layer_idx, (filters, kernel_size) in enumerate(zip(hparams.cnn_input_projector_filters, hparams.cnn_input_projector_kernels)):
        with tf.variable_scope("conv{}".format(layer_idx + 1)):
          net = layers.list_conv1d_with_bn(net,
                                           filters=filters,
                                           kernel_size=kernel_size,
                                           is_training=is_training,
                                           bn_decay=self.bn_decay,
                                           padding='same',
                                           seed=self.random_seed)
    else:
      raise ValueError("Unknown projector {:s}.".format(hparams.input_projector_type))
    
    return net
  
  def _input_transform_net(self, hparams, list_pointcloud, is_training, bn_decay=None, K=3):
    with tf.name_scope("expand_dims"):
      input_images = layers.list_expand_dims(list_pointcloud, -1)
    
    with tf.variable_scope("tconv1"):
      net = layers.list_conv2d_with_bn(input_images, 64, (1, 3), is_training, bn_decay, seed=self.random_seed)

    with tf.variable_scope("tconv2"):
      net = layers.list_conv2d_with_bn(net, 128, (1, 1), is_training, bn_decay, seed=self.random_seed)

    with tf.variable_scope("tconv3"):
      net = layers.list_conv2d_with_bn(net, 1024, (1, 1), is_training, bn_decay, seed=self.random_seed)
    
    with tf.name_scope("tmaxpool"):
      net = layers.list_maxpool2d(net, (hparams.num_point, 1))
    
    with tf.name_scope("flatten"):
      net = layers.list_flatten(net)

    with tf.variable_scope("tfc1"):
      net = layers.list_dense_with_bn(net, 512, is_training, bn_decay, seed=self.random_seed)
    
    with tf.variable_scope("tfc2"):
      net = layers.list_dense_with_bn(net, 256, is_training, bn_decay, seed=self.random_seed)

    with tf.variable_scope("transform_XYZ"):
      transform = layers.list_dense2(net, K*K,
                                     kernel_initializer=tf.initializers.zeros,
                                     bias_initializer=tf.initializers.constant(np.eye(K).flatten()))
      transform = layers.list_reshape(transform, (-1, K, K), new_scope=False)
    
    return transform
      
  def _feature_transform_net(self, hparams, list_inputs, is_training, bn_decay=None, K=64):
    with tf.variable_scope("tconv1"):
      net = layers.list_conv2d_with_bn(list_inputs, 64, (1, 1), is_training, bn_decay, seed=self.random_seed)

    with tf.variable_scope("tconv2"):
      net = layers.list_conv2d_with_bn(net, 128, (1, 1), is_training, bn_decay, seed=self.random_seed)

    with tf.variable_scope("tconv3"):
      net = layers.list_conv2d_with_bn(net, 1024, (1, 1), is_training, bn_decay, seed=self.random_seed)
    
    with tf.name_scope("tmaxpool"):
      net = layers.list_maxpool2d(net, (hparams.num_point, 1))
    
    with tf.name_scope("flatten"):
      net = layers.list_flatten(net)
    
    with tf.variable_scope("tfc1"):
      net = layers.list_dense_with_bn(net, 512, is_training, bn_decay, seed=self.random_seed)
    
    with tf.variable_scope("tfc2"):
      net = layers.list_dense_with_bn(net, 256, is_training, bn_decay, seed=self.random_seed)
      
    with tf.variable_scope("transform_XYZ"):
      transform = layers.list_dense2(net, K*K,
                                     kernel_initializer=tf.initializers.zeros,
                                     bias_initializer=tf.initializers.constant(np.eye(K).flatten()))
      transform = layers.list_reshape(transform, (-1, K, K), new_scope=False)
    
    return transform

  def _build_pointnet(self, hparams, is_training):
    """Build pointnet from raw pointcloud."""
    with tf.variable_scope("PointNet"):      
      # pointnet input
      with tf.name_scope("pointnet_placeholder"):
        input_phs = layers.list_placeholder(self.num_gpu, (None, hparams.num_point, 3), tf.int16)
      for ph in input_phs:
        tf.add_to_collection('placeholder', ph)
      
      with tf.name_scope("cast"):
        list_pointcloud = layers.list_cast(input_phs, tf.float32)

      with tf.variable_scope('transform_net'):
        list_transform = self._input_transform_net(hparams, list_pointcloud, is_training, self.bn_decay, K=3)

      with tf.name_scope('input_transform'):
        list_pointcloud_transformed = layers.list_matmul(list_pointcloud, list_transform)# ([Batch, num_point, 3]).dot([Batch, 3, 3])
        list_inputs = layers.list_expand_dims(list_pointcloud_transformed, -1, new_scope=False)
      
      with tf.variable_scope("conv1"):
        net = layers.list_conv2d_with_bn(list_inputs, 64, (1, 3), is_training, self.bn_decay, seed=self.random_seed)
      
      with tf.variable_scope("conv2"):
        net = layers.list_conv2d_with_bn(net, 64, (1, 1), is_training, self.bn_decay, seed=self.random_seed)

      with tf.variable_scope('transform_net2'):
        list_transform = self._feature_transform_net(hparams, net, is_training, self.bn_decay, K=64)
        with tf.name_scope('orthogonal_regularizer'):
          for gpu_idx, transforms in enumerate(list_transform):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{}".format(gpu_idx)):
              mat_diff = tf.matmul(transforms, tf.transpose(transforms, perm=[0, 2, 1]))
              mat_diff -= tf.constant(np.eye(64), dtype=tf.float32)
              orthogonal_loss = tf.nn.l2_loss(mat_diff)
              tf.add_to_collection("orthogonal_loss", orthogonal_loss)
          
      with tf.name_scope('feature_transform'):
        net = layers.list_squeeze(net, -2)
        point_feat = layers.list_matmul(net, list_transform, new_scope=False) # ([Batch, num_point, 64]).dot([Batch, 64, 64])
        point_feat = layers.list_expand_dims(point_feat, -2, new_scope=False)

      with tf.variable_scope("conv3"):
        net = layers.list_conv2d_with_bn(point_feat, 64, (1, 1), is_training, self.bn_decay, seed=self.random_seed)

      with tf.variable_scope("conv4"):
        net = layers.list_conv2d_with_bn(net, 128, (1, 1), is_training, self.bn_decay, seed=self.random_seed)

      with tf.variable_scope("conv5"):
        net = layers.list_conv2d_with_bn(net, 1024, (1, 1), is_training, self.bn_decay, seed=self.random_seed)

      with tf.name_scope("maxpool"):
        net = layers.list_maxpool2d(net, (hparams.num_point, 1))
        list_global_feat = layers.list_flatten(net, new_scope=False)
    
    return list_global_feat

  def _build_merge(self, hparams, final_state, global_feature, is_training):
    with tf.variable_scope('feature_merge'):
      with tf.name_scope("concat"):
        net = layers.list_concat(final_state, global_feature, axis=-1)
      
      with tf.variable_scope('fc1'):
        net = layers.list_dense_with_bn(net, 512, is_training, self.bn_decay, seed=self.random_seed)

    return net
  
  def _get_histogram_summary(self):
    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)), tf.name_scope("activation_histogram"):
      rnn_state_stack = tf.stack(self.list_encoder_output, axis=0)
      feature_stack = tf.stack(self.list_global_feature, axis=0)
      merged_state_stack = tf.stack(self.list_merged_state, axis=0)
      
      return [tf.summary.merge([tf.summary.histogram("rnn_state", rnn_state_stack),
                               tf.summary.histogram("global_feature", feature_stack),
                               tf.summary.histogram("merged_feature", merged_state_stack)])]