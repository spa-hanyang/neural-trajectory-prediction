from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import numpy as np

import tensorflow as tf

from . import model_helper
from . import list_ops
from .utils import misc_utils as utils

import pdb

__all__ = ["BaseModel", "Model"]

class BaseModel(object):
  """ Sequence-to-sequence base class. """
  def __init__(self,
               hparams,
               mode,
               scope=None):
    """ Create the model.
    Args:
      hparams: Hyperparameter configs.
      mode: TRAIN | EVAL | PREDICT
      scope: scope of the model.
    """
    # Set non-trainable parameters
    self._set_params_initializer(hparams, mode, scope=scope or hparams.model_name)

    # Set input/output shapes
    self._set_shapes_initializer(hparams, scope=scope or hparams.model_name)

    # Build graph
    # res[0] = list_regression, res[1:] = other informations
    # loss[0] = list_total_loss, loss[1] = list_regression loss, loss[2] = list_decay loss, loss[3:] = other losses
    (res, loss) = self.build_graph(hparams, scope=scope or hparams.model_name)

    # Define the optimizer ops, tensorboard ops
    self._set_train_or_infer(hparams, res, loss)

    # Placeholders
    self.placeholders = tf.get_collection('placeholder') + self.batch_size # + [self.is_training]

    # Saver
    global_vars = tf.global_variables()
    self.tower_vars = [list(filter(lambda x: "tower_{:d}".format(gpu_idx) in x.name, global_vars)) for gpu_idx in range(self.num_gpu)]
    self.non_tower_vars = list(filter(lambda x: "tower" not in x.name, global_vars))
    
    # Save only the first tower variables and tower-independent variables.
    self.saver = tf.train.Saver(
      self.tower_vars[0] + self.non_tower_vars, max_to_keep=hparams.num_keep_ckpts)
    
    # When restoring the variables, copy the first tower variable to other towers.
    restore_ops = []
    for gpu_idx in range(1, self.num_gpu):
      for var_idx, var in enumerate(self.tower_vars[gpu_idx]):
        restore_ops.append(tf.assign(var, self.tower_vars[0][var_idx], validate_shape=False))
    self.restore_op = tf.group(restore_ops)

  def _set_params_initializer(self,
                              hparams,
                              mode,
                              scope):
    """Initialize non-trainable parameters."""
    self.mode = mode
    self.num_gpu = hparams.num_gpu
    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
      with tf.variable_scope("non_trainable_parameters"):   
        # Batch size placeholder (Gets the dynamic batch size)
        with tf.name_scope("batch_size"):
          self.batch_size = list_ops.list_placeholder(self.num_gpu, shape=(), dtype=tf.int32)

        # Learning rate
        self.learning_rate = tf.get_variable(name="learning_rate",
                                             initializer=hparams.learning_rate,
                                             dtype=tf.float32,
                                             trainable=False)
        self.lr_decay_ratio = tf.placeholder(dtype=tf.float32, shape=(), name="lr_decay_ratio")

        # Global step
        self.global_step = tf.get_variable(name="global_step",
                                           initializer=np.int64(0),
                                           dtype=tf.int64,
                                           trainable=False)
        
        # BN decay
        with tf.variable_scope("batch_norm_decay"):
          bn_momentum = tf.train.exponential_decay(hparams.bn_init_decay,
                                                   self.global_step * hparams.batch_size,
                                                   hparams.bn_decay_step,
                                                   hparams.bn_decay_rate,
                                                   staircase=True)
          self.bn_decay = tf.minimum(hparams.bn_decay_clip, 1 - bn_momentum)

    # random_seed
    self.random_seed = hparams.random_seed

  def _set_shapes_initializer(self,
                              hparams,
                              scope):    
    # Determine the source sequence shapes.
    self.input_dims = hparams.input_dims
    self.input_length = hparams.input_length
      
    # Determine the target (sequence) shapes.
    self.target_dims = hparams.target_dims
    self.target_length = hparams.target_length
    
    # Determine other shapes (e.g., pointcloud shapes).
    self._set_other_shapes(hparams, scope)
  
  @abc.abstractmethod
  def _set_other_shapes(self, hparams, scope=None):
    """Subclass must implement this method.
    """
    pass

  @abc.abstractmethod
  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (result_tuple, loss_tuple)
    """
    pass

  def _set_train_or_infer(self, hparams, res, loss):
    """Set up training and inference."""
    # Training
    if self.mode == tf.estimator.ModeKeys.TRAIN:
      trainable_vars = tf.trainable_variables()
      total_loss = loss[0]
      # Print trainable variables
      utils.print_out("# Trainable variables")
      utils.print_out("Format: <name>, <shape>, <(soft) device placement>")
      for param in trainable_vars:
        utils.print_out("  {}, {}, {}".format(param.name,
                                              str(param.get_shape()),
                                              param.op.device))

      # [K by N]. K: num_gpu, N:num_variables per gpu
      list_vars = [list(filter(lambda x: "tower_{:d}".format(gpu_idx) in x.name, trainable_vars)) for gpu_idx in range(self.num_gpu)]
      
      with tf.variable_scope("optimization"):
        # Calculate gradient per device
        list_grads = []
        with tf.name_scope("compute_gradients"):
          for gpu_idx in range(self.num_gpu):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{}".format(gpu_idx)):
              loss = total_loss[gpu_idx]
              
              list_grads.append(
                  tf.gradients(loss,
                               list_vars[gpu_idx],
                               colocate_gradients_with_ops=hparams.colocate_gradients_with_ops))

          # Apply NCCL all reduce w/ average on the list_grads
          with tf.name_scope("all_reduce"):
            list_grads = model_helper.allreduce_tensors(list_grads, average=True)

          # Gradient clipping (Not clipped if max_gradient_norm=None)
          with tf.name_scope("clipping"):
            list_grads, list_norms = model_helper.gradient_clip(list_grads, max_gradient_norm=hparams.max_gradient_norm)
            self.grad_norm = list_norms[0]
        
        # Apply gradient per device
        opts = []
        update_ops = []
        with tf.variable_scope("optimizer"):
          for gpu_idx in range(self.num_gpu):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope("tower_{}".format(gpu_idx)):
              if hparams.optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
              elif hparams.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
              else:
                raise ValueError("Unknown optimizer type {}".format(hparams.optimizer))
            opts.append(optimizer)
            update_ops.append(optimizer.apply_gradients(zip(list_grads[gpu_idx], list_vars[gpu_idx])))
        
        add_global_step = tf.assign_add(self.global_step, 1)
        with tf.control_dependencies([add_global_step]):
          self.update = tf.group(*update_ops, name='update_op')
      
      self.train_summary = self._get_train_summary()

  def _get_train_summary(self):
    """Get train summary."""
    tf.summary.scalar("lr", self.learning_rate)
    tf.summary.scalar("bn_decay", self.bn_decay)

    return tf.summary.merge_all()

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)
    sess.run(self.restore_op)

  def learning_rate_decay(self, sess, decay_ratio):
    assert self.mode == tf.estimator.ModeKeys.TRAIN
    with tf.name_scope("learning_rate_decay"):
      return sess.run(tf.assign(self.learning_rate, self.learning_rate * self.lr_decay_ratio), feed_dict={self.lr_decay_ratio: decay_ratio})

class TrainOutputTuple(collections.namedtuple(
    "TrainOutputTuple", ("train_summary", "regression_loss", "global_step",
                         "grad_norm", "learning_rate"))):
  """To allow for flexibily in returing different outputs."""
  pass

class EvalOutputTuple(collections.namedtuple(
    "EvalOutputTuple", ("regression", "regression_loss"))):
  """To allow for flexibily in returing different outputs."""
  pass


class InferOutputTuple(collections.namedtuple(
    "InferOutputTuple", ("regression"))):
  """To allow for flexibily in returing different outputs."""
  pass

class Model(BaseModel):  
  """Sequence-to-sequence basic model.
  This class implements a single-layer lstm as trajectory encoder,
  and a single-stack lstm or MLP as trajectory decoder.
  Projection dense layers are applied around lstm(s).
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
                                    global_step=self.global_step,
                                    grad_norm=self.grad_norm,
                                    learning_rate=self.learning_rate)
    return sess.run([self.update, output_tuple], feed_dict=feed_dict)
  
  def eval(self, sess, feed_dict):
    """Execute eval graph."""
    assert self.mode == tf.estimator.ModeKeys.EVAL
    output_tuple = EvalOutputTuple(regression=self.regression,
                                   regression_loss=self.regression_loss)
    return sess.run(output_tuple, feed_dict=feed_dict)

  def infer(self, sess, feed_dict):
    assert self.mode == tf.estimator.ModeKeys.PREDICT
    output_tuple = InferOutputTuple(regression=self.regression)
    return sess.run(output_tuple, feed_dict=feed_dict)

  def build_graph(self, hparams, scope=None):
    with tf.variable_scope("Model"):
      utils.print_out("# Creating {} graph ...".format(self.mode))
      
      is_training = (self.mode == tf.estimator.ModeKeys.TRAIN)
      
      # Encoder
      list_encoder_output, list_encoder_state = self._build_encoder(hparams, is_training)

      # Decoder
      list_regression, _ = self._build_decoder(hparams, list_encoder_output, list_encoder_state, is_training)
      
      with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)), tf.name_scope("output"):
        # Concatenate final outputs from all devices
        self.regression = tf.concat(list_regression, axis=0)
    
    list_losses = None
    if self.mode != tf.estimator.ModeKeys.PREDICT:
      # Calculate loss in train and eval phase
      with tf.name_scope("loss"):
        list_losses = self._compute_loss(hparams, list_regression)

    return (list_regression, ), list_losses
  
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

      elif hparams.encoder_type == "cnn":
        net, state = self._build_cnn_encoder(hparams, input_phs, is_training)
      
      else:
        raise ValueError("Unknown encoder type {:s}.".format(hparams.encoder_type))

    return net, state

  def _build_decoder(self, hparams, list_encoder_output, list_encoder_state, is_training):
    """Build and run trajectory decoder.

    Args:
    hparams: The Hyperparameters.
      list_encoder_output: The output from the encoder
      list_encoder_state: The state from the encoder.
      is_training: Wheter training stage or not.

    Returns:
      A tuple of final logits and final decoder state:
        regression: size [batch_size, time, regression_dim].
    """
    ## Decoder.
    with tf.variable_scope("trajectory_decoder"):

      if hparams.decoder_type == "fc":
        regression = self._build_fc_decoder(hparams, list_encoder_output, is_training)
        final_states = None
      
      elif hparams.decoder_type == "rnn":
        with tf.variable_scope("rnn"):
          list_input = self._get_rnn_decoder_inputs(hparams, list_encoder_output)
          net, final_states = self._build_rnn_decoder(hparams, list_input, list_encoder_state, is_training)

        with tf.name_scope("time_batch_transpose"):
          net = list_ops.list_transpose(net, perm=[1, 0, 2])
        
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
    
    state = None
    if hparams.decoder_type == "rnn":
      # Imitate rnn state
      if hparams.rnn_decoder_type == "cudnn_lstm":
        with tf.name_scope("expand_dims"):  
          state = list_ops.list_expand_dims(net, axis=0)
        with tf.name_scope("zeros_like"):
          cell_state = list_ops.list_zeros_like(state) # zero cell state
      
      state = [*zip(state, cell_state)]

    return net, state

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

  def _get_rnn_decoder_inputs(self, hparams, encoder_outputs):
    list_inputs = []
    with tf.name_scope("decoder_inputs"):
      if hparams.rnn_decoder_inputs == "dummy":
        with tf.name_scope("zeros"):
          for gpu_idx in range(self.num_gpu):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{:d}".format(gpu_idx)):
              list_inputs.append(tf.zeros([self.target_length, self.batch_size[gpu_idx], 1]))
      
      elif hparams.rnn_decoder_inputs == "encoder_output":
        with tf.name_scope("expand_dims"):
          encoder_outputs = list_ops.list_expand_dims(encoder_outputs, axis=0)
        with tf.name_scope("tile"):
          list_inputs = list_ops.list_tile(encoder_outputs, [self.target_length, 1, 1])

      else:
        raise ValueError("Unknown decoder input type {:s}.".format(hparams.rnn_decoder_inputs))
    
    return list_inputs
  
  def _build_rnn_decoder(self, hparams, list_input, initial_states, is_training):
    if hparams.rnn_decoder_type == "cudnn_lstm":
      net, state = list_ops.list_cudnnlstm(list_input, num_layers=hparams.rnn_decoder_layers, num_units=hparams.rnn_decoder_units, list_initial_state=initial_states, final_output=False, seed=self.random_seed)
    else:
      raise ValueError("Unknown rnn {:s}.".format(hparams.rnn_encoder_type))
    
    return net, state
  
  def _build_output_projection(self, hparams, list_input, is_training):
    reg_func = tf.contrib.layers.l2_regularizer(hparams.weight_decay_factor)
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

  def _compute_loss(self, hparams, res):
    """Compute loss."""
    # Regression Loss
    with tf.name_scope("regression_loss"):
      loss_type = hparams.loss
      with tf.name_scope("target_placeholder"):
        target_phs = list_ops.list_placeholder(self.num_gpu, (None, self.target_length, self.target_dims), tf.float32)
      for ph in target_phs:
        tf.add_to_collection('placeholder', ph)
      
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

    # Total Loss
    with tf.name_scope("total_loss"):
      list_total_loss = [*zip(list_regression_loss, list_decay_loss)]
      with tf.name_scope("add_n"):
        list_total_loss = list_ops.list_add_n(list_total_loss)
      with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)), tf.name_scope("reduce_mean"):
        self.total_loss = tf.reduce_mean(list_total_loss)

    # Define TB loss summaries
    tf.summary.scalar("all_tower_mean", self.total_loss, family='total_loss')
    [tf.summary.scalar("tower_{:d}".format(gpu_idx), list_total_loss[gpu_idx], family='total_loss') for gpu_idx in range(self.num_gpu)]
    tf.summary.scalar("all_tower_mean", self.regression_loss, family='regression_loss')
    [tf.summary.scalar("tower_{:d}".format(gpu_idx), list_regression_loss[gpu_idx], family='regression_loss') for gpu_idx in range(self.num_gpu)]
    tf.summary.scalar("weight_decay_loss", self.decay_loss, family='regularization_loss')

    return list_total_loss, list_regression_loss, list_decay_loss