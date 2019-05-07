from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import numpy as np

import tensorflow as tf

from .  import model_helper
from . import list_ops
from .  import model
from .utils import iterator_utils
from .utils import misc_utils as utils

import pdb

__all__ = ["Model"]

class TrainOutputTuple(collections.namedtuple(
    "TrainOutputTuple", ("train_summary", "regression_loss", "classification_loss",
                         "global_step", "grad_norm", "learning_rate"))):
  """To allow for flexibily in returing different outputs."""
  pass

class EvalOutputTuple(collections.namedtuple(
    "EvalOutputTuple", ("regression", "stop", "regression_loss", "classification_loss"))):
  """To allow for flexibily in returing different outputs."""
  pass

class InferOutputTuple(collections.namedtuple(
    "InferOutputTuple", ("regression", "stop"))):
  """To allow for flexibily in returing different outputs."""
  pass

class Model(model.BaseModel):
  """Sequence-to-sequence stop discriminator model.
  This class implements a single-layer lstm as trajectory encoder,
  and a single-layer lstm or MLP as trajectory decoder.
  Projection dense layers are applied around lstm(s).
  Stop discriminator is also aplied.
  """
  def __init__(self,
               hparams,
               mode,
               scope=None):
    super(Model, self).__init__(hparams, mode, scope)

  def _set_other_shapes(self, hparams, scope=None):
    pass

  def train(self, sess, feed_dict):
    """Execute train graph."""
    assert self.mode == tf.estimator.ModeKeys.TRAIN
    output_tuple = TrainOutputTuple(train_summary=self.train_summary,
                                    regression_loss=self.regression_loss,
                                    classification_loss=self.classification_loss,
                                    global_step=self.global_step,
                                    grad_norm=self.grad_norm,
                                    learning_rate=self.learning_rate)
    return sess.run([self.update, output_tuple], feed_dict=feed_dict)
  
  def eval(self, sess, feed_dict):
    """Execute eval graph."""
    assert self.mode == tf.estimator.ModeKeys.EVAL
    output_tuple = EvalOutputTuple(regression=self.regression,
                                   stop=self.stop,
                                   regression_loss=self.regression_loss,
                                   classification_loss=self.classification_loss)
    return sess.run(output_tuple, feed_dict=feed_dict)

  def infer(self, sess, feed_dict):
    assert self.mode == tf.estimator.ModeKeys.PREDICT
    output_tuple = InferOutputTuple(regression=self.regression,
                                    stop=self.stop)
    return sess.run(output_tuple, feed_dict=feed_dict)

  def build_graph(self, hparams, scope=None):
    with tf.variable_scope("Model"):
      utils.print_out("# Creating {} graph ...".format(self.mode))
      
      is_training = (self.mode == tf.estimator.ModeKeys.TRAIN)
      
      # Encoder
      list_encoder_output, list_encoder_state = self._build_encoder(hparams, is_training)

      # Stop discriminator
      list_stop_score, list_classifier_result = self._build_stop_discriminator(hparams, list_encoder_output, is_training)

      # Decoder
      list_regression, _ = self._build_decoder(hparams, list_encoder_output, list_encoder_state, is_training)

      with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)), tf.name_scope("output"):
        # Concatenate final outputs from all devices
        self.regression = tf.concat(list_regression, axis=0)
        self.stop = tf.concat(list_classifier_result, axis=0)
     
      list_losses = None
      if self.mode != tf.estimator.ModeKeys.PREDICT:
        # Calculate loss in train and eval phase
        with tf.name_scope("loss"):
          list_losses = self._compute_loss(hparams, list_regression, list_stop_score)

    return (list_regression, list_classifier_result), list_losses

  def _build_input_projection(self, hparams, list_input, is_training):
    reg_func = tf.contrib.list_ops.l2_regularizer(hparams.weight_decay_factor)
    net = list_input
    if hparams.input_projector_type == "fc":
      for layer_idx, units in enumerate(hparams.fc_input_projector_units):
        with tf.variable_scope("fc{}".format(layer_idx + 1)):
          net = list_ops.list_dense(
              net, units=units, activation=tf.nn.relu, seed=self.random_seed, kernel_regularizer=reg_func)

    elif hparams.input_projector_type == "cnn":
      for layer_idx, (filters, kernel_size) in enumerate(zip(hparams.cnn_input_projector_filters, hparams.cnn_input_projector_kernels)):
        with tf.variable_scope("conv{}".format(layer_idx + 1)):
          net = list_ops.list_conv1d_with_bn(net,
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
      net, state = list_ops.list_cudnnlstm(list_input, num_layers=hparams.rnn_encoder_layers, num_units=hparams.rnn_encoder_units, seed=self.random_seed)
    else:
      raise ValueError("Unknown rnn {:s}.".format(hparams.rnn_encoder_type))
    
    return net, state
  
  def _build_cnn_encoder(self, hparams, list_input, is_training):
    net = list_input
    for layer_idx, (filters, kernel_size, dilation_rate) in enumerate(zip(hparams.cnn_encoder_filters, hparams.cnn_encoder_kernels, hparams.cnn_encoder_dilation_rates)):
      with tf.variable_scope("conv{}".format(layer_idx + 1)):
        net = list_ops.list_conv1d_with_bn(net,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         is_training=is_training,
                                         bn_decay=self.bn_decay,
                                         padding='valid',
                                         dilation_rate=dilation_rate,
                                         seed=self.random_seed)
    with tf.name_scope("flatten"):
      net = list_ops.list_flatten(net)
    
    return net

  def _build_encoder(self, hparams, is_training):
    """Build encoder from source."""
    with tf.variable_scope("trajectory_encoder"):
      with tf.name_scope("source_placeholder"):
        input_phs = list_ops.list_placeholder(self.num_gpu, (None, self.input_length, self.input_dims), tf.float32)
      for ph in input_phs:
        tf.add_to_collection('placeholder', ph)
      
      if hparams.encoder_type == "rnn":
        net = input_phs
        with tf.variable_scope("projection"):
          net = self._build_input_projection(hparams, net, is_training)

        with tf.name_scope("batch_time_transpose"):
          net = list_ops.list_transpose(net, perm=[1, 0, 2])

        with tf.variable_scope("rnn"):
          net, state = self._build_rnn_encoder(hparams, net, is_training)

        if hparams.relu_reconfiguration:
          with tf.variable_scope("reconfiguration"):
            net = list_ops.list_dense_with_bn(net,
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


  def _build_stop_discriminator(self, hparams, inputs, is_training):
    net = inputs
    with tf.variable_scope("stop_discriminator"):
      for layer_idx, units in enumerate(hparams.stop_discriminator_units):
        with tf.variable_scope("fc{}".format(layer_idx + 1)):
          if layer_idx < len(hparams.stop_discriminator_units) - 1:
            net = list_ops.list_dense_with_bn(net,
                                            units=units,
                                            is_training=is_training,
                                            bn_decay=self.bn_decay,
                                            seed=self.random_seed)
          else:
            net = list_ops.list_dense(net,
                                    units=units,
                                    seed=self.random_seed)
      with tf.name_scope("squeeze"):
        net = list_ops.list_squeeze(net)
      with tf.name_scope("inference"):
        with tf.name_scope("greater"):
          result = list_ops.list_greater(net, tf.constant(0.0, tf.float32)) # sigmoid(0) == 0.5
  
    return net, result

  def _build_fc_decoder(self, hparams, list_input, is_training):
    net = list_input
    for layer_idx, units in enumerate(hparams.fc_decoder_units):
      with tf.variable_scope("fc{}".format(layer_idx + 1)):
        if layer_idx < len(hparams.fc_decoder_units) - 1:
          net = list_ops.list_dense_with_bn(net,
                                          units=units,
                                          is_training=is_training,
                                          bn_decay=self.bn_decay,
                                          seed=self.random_seed)
        else:
          net = list_ops.list_dense(net,
                           units=units,
                           seed=self.random_seed)
    
    with tf.name_scope("expand_dims"):
      net = list_ops.list_expand_dims(net, axis=1)
    
    return net
  
  def _build_rnn_decoder(self, hparams, list_input, initial_states, is_training):
    if hparams.rnn_decoder_type == "cudnn_lstm":
      net, state = list_ops.list_cudnnlstm(list_input, num_layers=hparams.rnn_decoder_layers, num_units=hparams.rnn_decoder_units, list_initial_state=initial_states, final_output=False, seed=self.random_seed)
    else:
      raise ValueError("Unknown rnn {:s}.".format(hparams.rnn_encoder_type))
    
    return net, state
  
  def _build_output_projection(self, hparams, list_input, is_training):
    reg_func = tf.contrib.list_ops.l2_regularizer(hparams.weight_decay_factor)
    net = list_input
    for layer_idx, units in enumerate(hparams.output_projector_units):
      with tf.variable_scope("fc{}".format(layer_idx + 1)):
        if layer_idx < len(hparams.output_projector_units) - 1:
          net = list_ops.list_dense(net,
                                  units=units,
                                  activation=tf.nn.relu,
                                  seed=self.random_seed,
                                  kernel_regularizer=reg_func)
        else:
          net = list_ops.list_dense(net,
                                  units=units,
                                  seed=self.random_seed,
                                  kernel_regularizer=reg_func)
    
    return net
    
  def _make_initial_states(self, hparams, list_input):
    reg_func = tf.contrib.list_ops.l2_regularizer(hparams.weight_decay_factor)
    with tf.variable_scope("h_state"):
      with tf.variable_scope("fc"):
        h = list_ops.list_dense(list_input,
                              units=hparams.rnn_decoder_units * hparams.rnn_decoder_layers,
                              kernel_regularizer=reg_func,
                              seed=self.random_seed)
      
      with tf.name_scope("reshape"):
        h = list_ops.list_reshape(h, target_shape=[-1, hparams.rnn_decoder_layers, hparams.rnn_decoder_units])
        
      with tf.name_scope("transpose"):
        h = list_ops.list_transpose(h, perm=[1, 0, 2])
    
    with tf.name_scope("cell_state"):
      c = list_ops.list_zeros_like(h)
    
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
          net = list_ops.list_transpose(net, perm=[1, 0, 2])
        
        with tf.variable_scope("projection"):
          regression = self._build_output_projection(hparams, net, is_training)

      else:
        raise ValueError("Unknown decoder type {:s}.".format(hparams.decoder_type))

    return regression, final_states

  def _compute_loss(self, hparams, regression, stop_score):
    """Compute optimization loss."""
    # Weight Decay Loss
    with tf.name_scope("weight_decay_loss"):
      all_decay_losses = tf.losses.get_regularization_losses()
      if len(all_decay_losses):
        list_regs = [list(filter(lambda x: "tower_{:d}".format(gpu_idx) in x.name, all_decay_losses)) for gpu_idx in range(self.num_gpu)]
        with tf.name_scope("add_n"):
          list_decay_loss = list_ops.list_add_n(list_regs)

      else:
        list_decay_loss = list_ops.list_zeros_like([np.float32(0.0) for _ in range(self.num_gpu)])
      self.decay_loss = list_decay_loss[0]
    
    # Vehicle stop classification loss
    with tf.name_scope("stop_placeholder"):
      stop_phs = list_ops.list_placeholder(self.num_gpu, (None), tf.float32)
      for ph in stop_phs:
        tf.add_to_collection('placeholder', ph)
    with tf.name_scope("classifier_loss"):
      with tf.name_scope("sigmoid_cross_entropy"):
        list_classifier_loss = list_ops.list_sigmoid_cross_entropy(stop_phs, stop_score)
      with tf.name_scope("reduce_sum"):
        list_classifier_loss = list_ops.list_reduce_sum(list_classifier_loss)
      with tf.name_scope("cast"):
        batch_size_float = list_ops.list_cast(self.batch_size, tf.float32)
      with tf.name_scope("division"):
        list_classifier_loss = list_ops.list_divide(list_classifier_loss, batch_size_float)
      with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)), tf.name_scope("reduce_mean"):
        self.classifier_loss = tf.reduce_mean(list_classifier_loss)

    # Regression Loss
    with tf.name_scope("regression_loss"):
      with tf.name_scope("target_placeholder"):
        target_phs = list_ops.list_placeholder(self.num_gpu, (None, self.target_length, self.target_dims), tf.float32)
      for ph in target_phs:
        tf.add_to_collection('placeholder', ph)

      with tf.name_scope("count_non_stopped"):
        with tf.name_scope("reduce_sum"):
          list_stopped = list_ops.list_reduce_sum(stop_phs)
        with tf.name_scope("subtract"):
          list_non_stopped = list_ops.list_subtract(self.batch_size, list_stopped)

      with tf.name_scope("filter_stopped_objects"):
        list_ops.list_boolean_mask(regression_loss, classifier_result, axis=0)
        list_ops.list_boolean_mask(regression_loss, classifier_result, axis=0)

      loss_type = hparams.loss
      
      with tf.name_scope("{:s}_loss".format(loss_type)):
        if loss_type == "l2":
          loss = list_ops.list_l2(target_phs, res)
        elif loss_type == "weighted_smooth_l1":
          loss = list_ops.list_weighted_smooth_l1(target_phs, res)
        else:
          raise ValueError("Unknown loss type {:s}".format(loss_type))

      with tf.name_scope("reduce_sum"):
        loss = list_ops.list_reduce_sum(loss)
      with tf.name_scope("cast"):
        batch_size_float = list_ops.list_cast(self.batch_size, tf.float32)
      with tf.name_scope("division"):
        list_regression_loss = list_ops.list_divide(loss, batch_size_float)
      with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)), tf.name_scope("reduce_mean"):
        self.regression_loss = tf.reduce_mean(list_regression_loss)


    # Total Loss
    with tf.name_scope("total_loss"):
      list_total_loss = [*zip(list_regression_loss, list_decay_loss)]
      with tf.name_scope("add_n"):
        list_total_loss = list_ops.list_add_n(list_total_loss)
      with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)), tf.name_scope("reduce_mean"):
        self.total_loss = tf.reduce_mean(list_total_loss)

    loss_type = hparams.loss
    with tf.variable_scope(loss_type + "_loss"):
      with tf.name_scope("target_placeholder"):
        target_phs = list_ops.list_placeholder(self.num_gpu, (None, self.target_length, self.target_dims), tf.float32)
      for ph in target_phs:
        tf.add_to_collection('placeholder', ph)

      if loss_type == "l2":
        regression_loss = list_ops.list_l2(target_phs, regression, self.batch_size, normalize=False)
      elif loss_type == "weighted_smooth_l1":
        regression_loss = list_ops.list_weighted_smooth_l1(target_phs, regression, self.batch_size, normalize=False)
      else:
        assert False
      
      with tf.name_scope("filter_stopped_objects"):
        with tf.name_scope("boolean_mask"):
          filtered_regression_loss = 
        with tf.name_scope("where"):
          filtered_regression_loss = list_ops.list_where(lambda x: x > 0, regression_loss, tf.constant(0.0, tf.float32))
        

    with tf.variable_scope("loss"):
      self.regression_loss = regression_loss
      self.classifier_loss = classifier_loss
      loss_list = list_ops.list_add(classifier_loss, regression_loss)

    return loss_list