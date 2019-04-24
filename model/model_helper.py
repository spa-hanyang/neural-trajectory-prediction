from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
import six

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nccl_ops as nccl
from functools import reduce
import operator

from .utils import misc_utils as utils

import pdb

__all__ = [
    "get_initializer", "create_train_model", "create_eval_model",
    "create_infer_model", "compute_loss_and_predict",
    "gradient_clip", "create_or_load_model", "load_model", "allreduce_grads"
]

def get_initializer(init_op, seed=None, init_weight=None):
  """Create an initializer. init_weight is only for uniform."""
  if init_op == "uniform":
    assert init_weight
    return tf.random_uniform_initializer(
        -init_weight, init_weight, seed=seed)
  elif init_op == "glorot_normal":
    return tf.keras.initializers.glorot_normal(
        seed=seed)
  elif init_op == "glorot_uniform":
    return tf.keras.initializers.glorot_uniform(
        seed=seed)
  else:
    raise ValueError("Unknown init_op %s" % init_op)

class TrainModel(
    collections.namedtuple("UnifiedModel",
                           ("graph", "model", "placeholders"))):
  pass

def create_unified_model(model_creator, hparams, scope=None):
  unified_graph = tf.Graph()

  with unified_graph.as_default(), tf.container(scope or tf.estimator.ModeKeys.TRAIN):
    unified_model = model_creator(hparams, mode=tf.estimator.ModeKeys.TRAIN, scope=scope)

  return TrainModel(
      graph=unified_graph,
      model=unified_model,
      placeholders=unified_model.placeholders)

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
    print("Can't load checkpoint")
    print_variables_in_ckpt(ckpt_path)
    print("%s" % str(e))

  print(
      "  loaded %s model parameters from %s, time %.2fs" %
      (name, ckpt_path, time.time() - start_time))
  return model

def create_or_load_model(model, model_dir, session, name):
  """Create translation model and initialize or load parameters in session."""
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    session.run(tf.global_variables_initializer())
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    print("  Created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

  global_step = model.global_step.eval(session=session)
  return model, global_step

def compute_loss_and_predict(model, sess, dataflow, label):
  """Compute mean loss of the output of the model.

  Args:
    model: model for compute perplexity.
    sess: tensorflow session to use.
    label: label of the dataset.

  Returns:
    prediction: prediction output
    avg_loss: the mean loss of the eval outputs.
  """
  total_loss = 0
  total_batch = 0
  
  prediction = []
  gt = []
  
  utils.print_out("  Begin {} evaluation.".format(label))

  for iters, batches in enumerate(dataflow.get_data()):
    # feed dict
    batch_sizes = [batch.shape[0] for batch in batches[0]]
    feed_dict={key:value for (key, value) in zip(
        model.placeholders,
        reduce(operator.add, batches) + batch_sizes)}    
    
    # Eval step
    output_tuple = model.eval(sess, feed_dict=feed_dict)
    dynamic_batch_size = np.sum(output_tuple.batch_size)
    total_loss += output_tuple.eval_loss * dynamic_batch_size
    total_batch += dynamic_batch_size
    prediction += [output_tuple.eval_regression]
    gt += [np.concatenate(batches[-1], axis=0)]
    utils.print_out("iters {:d}, loss {:.3f}".format(iters, output_tuple.eval_loss), end='\r')
  
  avg_loss = total_loss / total_batch
  prediction = np.concatenate(prediction, axis=0)
  gt = np.concatenate(gt, axis=0)
  pdb.set_trace()
  return prediction, gt, avg_loss

def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  if max_gradient_norm is not None:
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)

  else:
    clipped_gradients = gradients
    gradient_norm = tf.global_norm(gradients)

  return clipped_gradients, gradient_norm

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
        return all_tensors
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