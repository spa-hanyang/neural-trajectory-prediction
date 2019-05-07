from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
import six
from functools import reduce
import operator

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nccl_ops as nccl

from .utils import misc_utils as utils
from . import list_ops

import pdb

__all__ = [
    "create_train_model", "create_eval_model", "create_infer_model",
    "compute_loss_and_predict", "create_or_load_model", "load_model",
    "gradient_clip", "allreduce_tensors"]

class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph", "model", "placeholders"))):
  pass

def create_train_model(model_creator, hparams, scope=None):
  """Create train graph, model, and iterator."""
  # Create the train model graph
  train_graph = tf.Graph()
  
  # Build the train model on the graph.
  with train_graph.as_default(), tf.container(scope or tf.estimator.ModeKeys.TRAIN):
    train_model = model_creator(hparams, mode=tf.estimator.ModeKeys.TRAIN, scope=scope)

  return TrainModel(
      graph=train_graph,
      model=train_model,
      placeholders=train_model.placeholders)

class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model", "placeholders"))):
  pass

def create_eval_model(model_creator, hparams, scope=None):
  """Create train graph, model, src/tgt file holders, and iterator."""
  # Create the eval model graph
  eval_graph = tf.Graph()

  # Build the eval model on the graph.
  with eval_graph.as_default(), tf.container(scope or tf.estimator.ModeKeys.EVAL):
    eval_model = model_creator(hparams, mode=tf.estimator.ModeKeys.EVAL, scope=scope)
  
  return EvalModel(
      graph=eval_graph,
      model=eval_model,
      placeholders=eval_model.placeholders)

class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "placeholders"))):
  pass

def create_infer_model(model_creator, hparams, scope=None, extra_args=None):
  """Create inference model."""
  # Create the infer model graph
  infer_graph = tf.Graph()

  with infer_graph.as_default(), tf.container(scope or tf.estimator.ModeKeys.PREDICT):
    infer_model = model_creator(hparams, mode=tf.estimator.ModeKeys.PREDICT, scope=scope)

  return InferModel(
      graph=infer_graph,
      model=infer_model,
      placeholders=infer_model.placeholders)

def print_variables_in_ckpt(ckpt_path):
  """Print a list of variables in a checkpoint together with their shapes."""
  utils.print_out("# Variables in ckpt %s" % ckpt_path)
  reader = tf.train.NewCheckpointReader(ckpt_path)
  variable_map = reader.get_variable_to_shape_map()
  for key in sorted(variable_map.keys()):
    print("  %s: %s" % (key, variable_map[key]))

def load_model(model, ckpt_path, session, name):
  """Load model from a checkpoint."""
  start_time = time.time()
  try:
    model.restore(session, ckpt_path)
  except tf.errors.NotFoundError as e:
    utils.print_out("Can't load checkpoint")
    print_variables_in_ckpt(ckpt_path)
    utils.print_out("{:s}".format(str(e)))

  utils.print_out("loaded {:s} model parameters from {:s}, in {:.2f}s".format(
      name, ckpt_path, time.time() - start_time))
  return model

def create_or_load_model(model, model_dir, session, name):
  """Create translation model and initialize or load parameters in session."""
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    print("  Created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

  global_step = model.global_step.eval(session=session)
  return model, global_step

def compute_loss_and_predict(hparams, model, sess, dataflow, label):
  """Compute mean loss of the output of the model.

  Args:
    hparams: hyperparameters
    model: model for prediction.
    sess: tensorflow session to use.
    label: label of the dataset.

  Returns:
    regression: prediction output
    avg_losses: the mean losses.
  """
  regression_loss = 0
  classification_loss = 0
  total_batch = 0
  
  # Get placeholders
  placeholders = model.placeholders
  source_ph = list(filter(lambda x: "source_placeholder" in x.name, placeholders))
  target_ph = list(filter(lambda x: "target_placeholder" in x.name, placeholders))
  ph_list = source_ph + target_ph
  if hparams.stop_discriminator:
    ph_list += list(filter(lambda x: "stop_placeholder" in x.name, placeholders))
  
  batch_size_ph = list(filter(lambda x: "batch_size" in x.name, placeholders))
  ph_list += batch_size_ph

  inputs = []
  target = []
  regression = []
  stop = []
  
  utils.print_out("  Begin {} evaluation.".format(label))

  for iters, batches in enumerate(dataflow.get_data()):
    # feed dict
    batch_sizes = [batch.shape[0] for batch in batches[0]]
    feed_dict={key:value for (key, value) in zip(
        ph_list,
        reduce(operator.add, batches) + batch_sizes)}    
    dynamic_batch_size = np.sum(batch_sizes)

    # Evaluation
    output_tuple = model.eval(sess, feed_dict=feed_dict)
    
    regression_loss += output_tuple.regression_loss * dynamic_batch_size
    total_batch += dynamic_batch_size
    regression += [output_tuple.regression]
    target += batches[-1]

    if hparams.stop_discriminator:
      classification_loss += output_tuple.classification_loss * dynamic_batch_size
      inputs += batches[0]
      stop += [output_tuple.stop]
      utils.print_out("iters {:d}, regression loss {:.3f}, classification loss {:.3f}".format(iters, output_tuple.regression_loss, output_tuple.classification_loss), end='\r')
    
    else:
      utils.print_out("iters {:d}, regression loss {:.3f}".format(iters, output_tuple.regression_loss), end='\r')

  avg_regression_loss = regression_loss / total_batch
  losses = {"regression_loss": avg_regression_loss}
  utils.print_out("Done. avg regression loss {:.3f}".format(avg_regression_loss))
  
  regression = np.concatenate(regression, axis=0)
  target = np.concatenate(target, axis=0)

  if hparams.stop_discriminator:
    avg_classification_loss = classification_loss / total_batch
    utils.print_out("Done. avg regression loss {:.3f}, avg classification loss {:.3f}".format(avg_regression_loss, avg_classification_loss))
    losses["classification_loss"] = avg_classification_loss
    inputs = np.concatenate(inputs, axis=0)
    stop = np.concatenate(stop, axis=0)
    regression[stop, :, :] = np.tile(inputs[stop, -1:, :hparams.trajectory_dims], [1, hparams.target_length, 1])
  
  return regression, target, losses

def gradient_clip(list_gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  if max_gradient_norm is not None:
    list_clipped_gradients, list_gradient_norm = list_ops.list_clip_by_global_norm(list_gradients, max_gradient_norm)

  else:
    list_clipped_gradients = list_gradients
    list_gradient_norm = list_ops.list_global_norm(list_gradients)

  return list_clipped_gradients, list_gradient_norm

def allreduce_tensors(all_tensors, average=True):
    """
    REFERENCE : https://github.com/ppwwyyxx/tensorpack/blob/83e4e187af5765792408e7b7163efd4744d63628/tensorpack/graph_builder/utils.py
    All-reduce average the per-device tensors of the variables among K devices.

    Args:
        all_tensors (K x N): List of list of tensors. N is the number of (independent) variables.
        average (bool): divide the tensors by N or not.
    Returns:
        K x N: same as input, but each tensor is replaced by the all reduce over K devices.
    """
    nr_tower = len(all_tensors)
    if nr_tower == 1:
      return all_tensors # No need to apply nccl reduce
    
    new_all_tensors = []  # N x K
    for tensors in zip(*all_tensors):
      summed = nccl.all_sum(tensors)

      tensors_for_devices = []  # K
      for tensor in summed:
        with tf.device(tensor.device):
          # tensorflow/benchmarks didn't average gradients
          if average:
            tensor = tf.multiply(tensor, 1.0 / nr_tower, name='allreduce_avg')
        tensors_for_devices.append(tensor)
      new_all_tensors.append(tensors_for_devices)

    # transpose to K x N
    ret = list(zip(*new_all_tensors))
    return ret