from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl

from . import model_helper
from . import layers
from .utils import misc_utils as utils

import pdb

LSTMStateTuple = rnn_cell_impl.LSTMStateTuple

__all__ = ["BaseModel", "Model"]


class TrainOutputTuple(collections.namedtuple(
    "TrainOutputTuple", ("train_summary", "train_loss", "global_step",
                         "grad_norm", "learning_rate"))):
  """To allow for flexibily in returing different outputs."""
  pass

class EvalOutputTuple(collections.namedtuple(
    "EvalOutputTuple", ("eval_regression", "eval_loss", "batch_size"))):
  """To allow for flexibily in returing different outputs."""
  pass


class InferOutputTuple(collections.namedtuple(
    "InferOutputTuple", ("infer_regression", "infer_summary"))):
  """To allow for flexibily in returing different outputs."""
  pass

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
    super(BaseModel, self).__init__()
    # Set params
    self._set_params_initializer(hparams, mode, scope)

    # Set shapes
    self._set_shapes_initializer(hparams, mode, scope)

    # Build graph
    # res[0] = regressions, res[1] = losses
    res = self.build_graph(hparams,
                           scope=scope or hparams.model_name)

    # Now we have all the forward calculation defined on graph.
    # Let us define the optimizer and its operations.
    self._set_train_or_infer(res, hparams)

    # Placeholders
    self.placeholders = tf.get_collection('placeholder') + self.batch_size # + [self.is_training]

    # Saver
    global_vars = tf.global_variables()
    per_tower_vars = [list(filter(lambda x: "tower_{:d}".format(gpu_idx) in x.name, global_vars)) for gpu_idx in range(self.num_gpu)]
    self.per_tower_vars = per_tower_vars
    
    self.saver = tf.train.Saver(
      per_tower_vars[0] + [self.learning_rate, self.global_step], max_to_keep=hparams.num_keep_ckpts)

    restore_ops = []
    for gpu_idx in range(1, self.num_gpu):
      for var_idx, var in enumerate(per_tower_vars[gpu_idx]):
        restore_ops.append(tf.assign(var, per_tower_vars[0][var_idx], validate_shape=False))
      
    self.restore_op = tf.group(restore_ops)

  def _set_shapes_initializer(self,
                              hparams,
                              mode,
                              scope):    
    # Determine the source sequence shapes.
    self.input_dims = hparams.input_dims
    # if hparams.orientation:
    #   self.input_dims += 1
    self.input_length = hparams.input_length

    # Determine the target (sequence) shapes.
    self.target_dims = hparams.target_dims
    target_length = hparams.target_length
    if hparams.single_target:
      self.target_length = 1
    else:
      self.target_length = target_length // hparams.target_sampling_period
    
    # Determine the pointcloud shapes.
    self.ptc_dims = 3
    
  def _set_params_initializer(self,
                              hparams,
                              mode,
                              scope):
    """Set various params for self and initialize."""
    self.mode = mode
    self.num_gpu = hparams.num_gpu
    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
      with tf.variable_scope("training_parameters"):   
        # Batch size placeholder
        self.batch_size = list(tf.placeholder(
            dtype=tf.int32, shape=[], name="tower_{}_batch_size".format(gpu_idx)) for gpu_idx in range(self.num_gpu))

        # Learning rate
        self.learning_rate = tf.get_variable(name="learning_rate",
                                             initializer=hparams.learning_rate,
                                             dtype=tf.float32,
                                             trainable=False)
        with tf.name_scope("learning_rate_decay"):
          self.decay_ratio = tf.placeholder(dtype=tf.float32, shape=[], name="lr_dacay_ratio")

        # Global step
        self.global_step = tf.get_variable(name="global_step",
                                           initializer=np.array(0, np.int64),
                                           dtype=tf.int64,
                                           trainable=False)

        with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)), tf.variable_scope("batch_norm_decay"):
          bn_momentum = tf.train.exponential_decay(hparams.bn_init_decay,
                                                   self.global_step * hparams.batch_size,
                                                   hparams.bn_decay_step,
                                                   hparams.bn_decay_rate,
                                                   staircase=True)
          self.bn_decay = tf.minimum(hparams.bn_decay_clip, 1 - bn_momentum)

    # Initializer
    self.random_seed = hparams.random_seed
    # initializer = model_helper.get_initializer(
    #     hparams.init_op, self.random_seed, hparams.init_weight)
    # tf.get_variable_scope().set_initializer(initializer)

  def _set_train_or_infer(self, res, hparams):
    """Set up training and inference."""
    trainable_vars = tf.trainable_variables()

    # Training
    if self.mode == tf.estimator.ModeKeys.TRAIN:      
      
      # reg losses
      with tf.variable_scope("regularization_loss"):
        reg_losses = self._get_regularization_losses()
      
      # orthogonal losses
      with tf.variable_scope("orthogonal_loss"):
        orthogonal_losses = self._get_orthogonal_losses()

      # [K by N]. K: num_gpu, N:num_variables per gpu
      list_vars = [list(filter(lambda x: "tower_{:d}".format(gpu_idx) in x.name, trainable_vars)) for gpu_idx in range(self.num_gpu)]
      
      with tf.name_scope("gradients"):
        grads = []
        for gpu_idx in range(self.num_gpu):
          with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.name_scope("tower_{}".format(gpu_idx)):
            train_loss = res[1][gpu_idx]
            reg_loss = reg_losses[gpu_idx]
            orthogonal_loss = orthogonal_losses[gpu_idx]
            total_loss = train_loss + reg_loss + orthogonal_loss
            grads.append(
                tf.gradients(total_loss,
                             list_vars[gpu_idx],
                             colocate_gradients_with_ops=hparams.colocate_gradients_with_ops))
          
        reduced_grad = model_helper.allreduce_tensors(grads, average=True)
        
      with tf.variable_scope("gradients_update"):
        opts = []
        update_ops = []
        grad_norms = []
        for gpu_idx in range(self.num_gpu):
          with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope("tower_{}".format(gpu_idx)):
            with tf.variable_scope("optimizer"):
              if hparams.optimizer == "sgd":
                opts.append(tf.train.GradientDescentOptimizer(self.learning_rate))
              elif hparams.optimizer == "adam":
                opts.append(tf.train.AdamOptimizer(self.learning_rate))
              else:
                raise ValueError("Unknown optimizer type {}".format(hparams.optimizer))
            
            with tf.variable_scope("gradient_clip"):
              # gradient for this gpu
              this_grad = reduced_grad[gpu_idx]

              # Gradient clipping (Not clipped if max_gradient_norm=None)
              clipped_grad, grad_norm = model_helper.gradient_clip(this_grad, max_gradient_norm=hparams.max_gradient_norm)
              grad_norms.append(grad_norm)

            update_op = opts[gpu_idx].apply_gradients(zip(clipped_grad, list_vars[gpu_idx]))
            update_ops.append(update_op)
      
      with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)), tf.variable_scope("moving_averages"):
        self.train_loss = tf.reduce_mean(res[1])
        self.reg_loss = tf.identity(reg_loss)
        self.orthogonal_loss = tf.reduce_mean(orthogonal_losses)
        self.grad_norm = tf.reduce_mean(grad_norms)

        loss_ema = tf.train.ExponentialMovingAverage(0.98)
        update_ma = [loss_ema.apply(var_list=[self.train_loss, self.reg_loss, self.orthogonal_loss] + res[1] + orthogonal_losses)]
        
        self.train_loss_ma = loss_ema.average(self.train_loss)
        self.train_loss_per_tower_ma = [loss_ema.average(res[1][gpu_idx]) for gpu_idx in range(self.num_gpu)]
        self.reg_loss_ma = loss_ema.average(self.reg_loss)
        self.orthogonal_loss_ma = loss_ema.average(self.orthogonal_loss)
        self.orthogonal_loss_per_tower_ma = [loss_ema.average(orthogonal_losses[gpu_idx]) for gpu_idx in range(self.num_gpu)]

      add_global_step = tf.assign_add(self.global_step, 1)
      with tf.control_dependencies([add_global_step] + update_ma):
        self.update = tf.group(*update_ops, name='update_op')

      # Summary
      self.train_summary = self._get_train_summary()
      
      # with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)), tf.name_scope("regression"):
      #   self.regression = tf.concat(res[0], axis=0)

    elif self.mode == tf.estimator.ModeKeys.EVAL:
      # Evaluation
      with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        self.eval_regression = tf.concat(res[0], axis=0)
        self.eval_loss = tf.reduce_mean(res[1])

    elif self.mode == tf.estimator.ModeKeys.PREDICT:
      # Inference
      with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        self.infer_regression = tf.concat(res[0], axis=0)
      self.infer_summary = self._get_infer_summary(hparams)

    # Print trainable variables
    utils.print_out("# Trainable variables")
    utils.print_out("Format: <name>, <shape>, <(soft) device placement>")
    
    for param in trainable_vars:
      utils.print_out("  {}, {}, {}".format(param.name,
                                            str(param.get_shape()),
                                            param.op.device))

  def _get_regularization_losses(self):
    all_reg_losses = tf.losses.get_regularization_losses()
    reg_losses = []
    
    for gpu_idx in range(self.num_gpu):
      with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope("tower_{:d}".format(gpu_idx)):
        if len(all_reg_losses):
          list_regs = list(filter(lambda x: "tower_{:d}".format(gpu_idx) in x.name, all_reg_losses))
          reg_losses.append(tf.add_n(list_regs))
        else:
          reg_losses.append(tf.constant(0.0, tf.float32))
    
    return reg_losses


  def _get_orthogonal_losses(self):
    orthogonal_losses = tf.get_collection('orthogonal_loss')
    if len(orthogonal_losses) == 0:
      for gpu_idx in range(self.num_gpu):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope("tower_{:d}".format(gpu_idx)):
          orthogonal_losses.append(tf.constant(0.0, tf.float32))
    
    return orthogonal_losses

  def _get_histogram_summary(self):
    return []

  def _get_train_summary(self):
    """Get train summary."""
    histogram_summary = self._get_histogram_summary()
    train_summary = tf.summary.merge(
        [tf.summary.scalar("lr", self.learning_rate),
         tf.summary.scalar("bn_decay", self.bn_decay),
         tf.summary.scalar("all_tower_mean", self.train_loss_ma, family='ma_train_loss'),
         tf.summary.scalar("ma_reg_loss", self.reg_loss_ma),
         tf.summary.scalar("all_tower_mean", self.orthogonal_loss_ma, family='ma_orthogonal_loss')] +
         [tf.summary.scalar("tower_{:d}".format(gpu_idx), self.train_loss_per_tower_ma[gpu_idx], family='ma_train_loss') for gpu_idx in range(self.num_gpu)] +
         [tf.summary.scalar("tower_{:d}".format(gpu_idx), self.orthogonal_loss_per_tower_ma[gpu_idx], family='ma_orthogonal_loss') for gpu_idx in range(self.num_gpu)] +
         histogram_summary)
    return train_summary

  def train(self, sess, feed_dict):
    """Execute train graph."""
    assert self.mode == tf.estimator.ModeKeys.TRAIN
    output_tuple = TrainOutputTuple(train_summary=self.train_summary,
                                    train_loss=self.train_loss,
                                    global_step=self.global_step,
                                    grad_norm=self.grad_norm,
                                    learning_rate=self.learning_rate)
    return sess.run([self.update, output_tuple], feed_dict=feed_dict)
  
  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)
    sess.run(self.restore_op)

  def learning_rate_decay(self, sess, decay_ratio):
    assert self.mode == tf.estimator.ModeKeys.TRAIN
    with tf.name_scope("learning_rate_decay"):
      return sess.run(tf.assign(self.learning_rate, self.learning_rate * self.decay_ratio), feed_dict={self.decay_ratio: decay_ratio})

  def eval(self, sess, feed_dict):
    """Execute eval graph."""
    assert self.mode == tf.estimator.ModeKeys.EVAL
    output_tuple = EvalOutputTuple(eval_regression=self.eval_regression,
                                   eval_loss=self.eval_loss,
                                   batch_size=self.batch_size)
    return sess.run(output_tuple, feed_dict=feed_dict)

  @abc.abstractmethod
  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (regression, loss, ...),
      where:
        regression: float32 Tensor [batch_size x prediction horizon].
        loss: loss = the total loss / batch_size.
    """
    pass

  def _compute_loss(self, regression, hparams):
    """Compute optimization loss."""
    loss_type = hparams.loss
    with tf.variable_scope(loss_type + "_loss"):
      with tf.name_scope("target_placeholder"):
        target_phs = layers.list_placeholder(self.num_gpu, (None, self.target_length, self.target_dims), tf.float32)
      for ph in target_phs:
        tf.add_to_collection('placeholder', ph)
      
      if loss_type == "l2":
        loss_list = layers.list_l2(target_phs, regression, self.batch_size)
      elif loss_type == "weighted_smooth_l1":
        loss_list = layers.list_weighted_smooth_l1(target_phs, regression, self.batch_size)
      else:
        assert False

    return loss_list

  def _get_infer_summary(self, hparams):
    del hparams
    return tf.no_op()

  def _infer(self, sess):
    assert self.mode == tf.estimator.ModeKeys.PREDICT
    output_tuple = InferOutputTuple(infer_regression=self.infer_regression,
                                    infer_summary=self.infer_summary)
    return sess.run(output_tuple)

  def decode(self, sess):
    """Decode a batch.

    Args:
      sess: tensorflow session to use.

    Returns:
      A tuple consiting of outputs, infer_summary.
        outputs: of size [batch_size, time]
    """
    output_tuple = self._infer(sess)
    infer_regression = output_tuple.infer_regression
    infer_summary = output_tuple.infer_summary

    return infer_regression, infer_summary

class Model(BaseModel):  
  """Sequence-to-sequence basic model.
  This class implements a single-layer lstm as trajectory encoder,
  and a single-stack lstm or MLP as trajectory decoder.
  Projection dense layers are applied around lstm(s).
  """
  def build_graph(self, hparams, scope=None):
    with tf.variable_scope("Model"):
      utils.print_out("# Creating {} graph ...".format(self.mode))
      
      is_training = (self.mode == tf.estimator.ModeKeys.TRAIN)
      
      # Encoder
      list_encoder_output, list_encoder_state = self._build_encoder(hparams, is_training)
      self.list_encoder_output = list_encoder_output

      # Decoder
      list_regression, _ = self._build_decoder(hparams, list_encoder_output, list_encoder_state, is_training)

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