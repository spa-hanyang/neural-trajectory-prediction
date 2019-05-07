from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import pickle as pkl
import pdb
import time

from functools import reduce
import operator

from . import model as basicmodel
from . import pointnet_model
from . import model_helper
from .utils import misc_utils as utils
from .utils import evaluation_utils
from .utils import transform_utils
from .utils import iterator_utils

__all__ = ["load_data", "inference"]

def load_data(inference_input_file, hparams=None):
  """Load inference data."""
  with open(inference_input_file, 'rb') as reader:
    inference_data = pkl.load(reader)

  return inference_data

def get_model_creator(hparams):
  """Get the right model class depending on configuration."""
  if hparams.lidar:
    model_creator = pointnet_model.Model
  else:
    model_creator = basicmodel.Model
  return model_creator

def start_sess_and_load_model(infer_model, ckpt):
  """Start session and load model."""
  sess = tf.Session(
      graph=infer_model.graph, config=utils.get_config_proto())
  
  with infer_model.graph.as_default():
    loaded_infer_model = model_helper.load_model(
        infer_model.model, ckpt, sess, "infer")
  return sess, loaded_infer_model

def inference(hparams,
              model_dir,
              ckpt,
              inference_input_file,
              inference_output_file,
              num_gpu=1,
              batch_size=16,
              scope=None):
  """Perform translation."""
  model_creator = get_model_creator(hparams)
  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

  if ckpt is None:
    ckpt = tf.train.latest_checkpoint(model_dir)
  else:
    ckpt = os.path.join(model_dir, "model.ckpt-{:d}".format(ckpt))

  sess, loaded_infer_model = start_sess_and_load_model(infer_model, ckpt)

  output_infer = inference_output_file

  # Read data
  dataset = load_data(inference_input_file, hparams)
  infer_df = iterator_utils.get_infer_iterator(hparams, dataset, num_gpu, batch_size)
  infer_df.reset_state()

  with infer_model.graph.as_default():
    # Decode
    utils.print_out("# Start decoding")
    _decode_and_evaluate(
        hparams,
        "infer",
        loaded_infer_model,
        sess,
        output_infer,
        infer_df,
        metrics=hparams.metrics)

def _decode_and_evaluate(hparams,
                         name,
                         model,
                         sess,
                         pred_file,
                         infer_dataflow,
                         metrics):
  """Decode a test set and compute a score according to the evaluation task."""
  # Decode
  utils.print_out("  Inference output to %s" % pred_file)

  # Get placeholders
  placeholders = model.placeholders
  ph_list = list(filter(lambda x: "source_placeholder" in x.name, placeholders))

  if hparams.stop_discriminator:
    ph_list += list(filter(lambda x: "stop_placeholder" in x.name, placeholders))
  
  batch_size_ph = list(filter(lambda x: "batch_size" in x.name, placeholders))
  ph_list += batch_size_ph

  inputs = []
  target = []
  regression = []
  stop = []
  for iters, batches in enumerate(infer_dataflow.get_data()):
    batch_sizes = [batch.shape[0] for batch in batches[0]]
    feed_dict={key:value for (key, value) in zip(
        ph_list,
        reduce(operator.add, batches[:-1]) + batch_sizes)}

    outputs = model.infer(sess, feed_dict)
    utils.print_out("  num_iters {}".format(iters), end='\r')
    target += batches[-1]
    regression += [outputs.regression]
    
    if hparams.stop_discriminator:
      inputs += batches[0]
      stop += [outputs.stop]

  regression = np.concatenate(regression, axis=0)
  target = np.concatenate(target, axis=0)

  with open(pred_file, 'wb') as writer:
    pkl.dump(regression, writer)

  if hparams.stop_discriminator:
    inputs = np.concatenate(inputs, axis=0)
    stop = np.concatenate(stop, axis=0)
    with open("stop_{:s}".format(pred_file), 'wb') as writer:
      pkl.dump(stop, writer)
    
    regression[stop, :, :] = np.tile(inputs[stop, -1:, :hparams.trajectory_dims], [1, hparams.target_length, 1])

  # Evaluation
  error = target - regression
  evaluation_scores = {}
  for metric in metrics:
    score = evaluation_utils.evaluate(
    hparams,
    error,
    metric)

    evaluation_scores[metric] = score
    utils.print_out("  {} {}: {:.1f}".format(metric, name, score))

  return evaluation_scores
