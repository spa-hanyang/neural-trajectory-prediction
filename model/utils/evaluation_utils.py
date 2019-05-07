from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pdb

__all__ = ["evaluate"]

def evaluate(hparams, error, metric):
  """Pick a metric and evaluate depending on task."""
  error_x = error[:, :, 0]
  error_y = error[:, :, 1]
  target_sampling_period = hparams.target_sampling_period
  metric = metric.lower()

  # MAE_X scores
  if "maex" in metric:
    try:
      horizon = int(metric[5:])
      idx = int(horizon // target_sampling_period - 1)
    except ValueError:
      if metric[5:] == "oneshot":
        idx = -1
      else:
        raise ValueError("Unknown metric %s" % metric)
    
    evaluation_score = _mae(error_x[:, idx])
  
  # MAE_Y scores
  elif "maey" in metric:
    try:
      horizon = int(metric[5:])
      idx = int(horizon // target_sampling_period - 1)
    except ValueError:
      if metric[5:] == "oneshot":
        idx = -1
      else:
        raise ValueError("Unknown metric %s" % metric)
    
    evaluation_score = _mae(error_y[:, idx])
  
  # RMSE scores
  elif "rmse" in metric:
    try:
      horizon = int(metric[5:])
      idx = int(horizon // target_sampling_period - 1)
    except ValueError:
      if metric[5:] == "oneshot":
        idx = -1
      else:
        raise ValueError("Unknown metric %s" % metric)
    
    evaluation_score = _rmse(error[:, idx, :])

  else:
    raise ValueError("Unknown metric %s" % metric)

  return evaluation_score

def _mae(error):
  return np.mean(np.abs(error))

def _rmse(error):
  return np.sqrt(np.mean(np.square(error)))