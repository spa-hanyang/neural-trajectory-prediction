from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = ["evaluate"]

def evaluate(error, metric):
  """Pick a metric and evaluate depending on task."""
  error_x = error[:, :, 0]
  error_y = error[:, :, 1]
  len_pred = error_x.shape[1]
  target_sampling_period = 20 // len_pred

  # MAE_X scores
  if metric.lower() == "maex_5":
    evaluation_score = _mae(error_x[:, int(5//target_sampling_period - 1)])
  elif metric.lower() == "maex_10":
    evaluation_score = _mae(error_x[:, int(10//target_sampling_period - 1)])
  elif metric.lower() == "maex_15":
    evaluation_score = _mae(error_x[:, int(15//target_sampling_period - 1)])
  elif metric.lower() == "maex_20":
    evaluation_score = _mae(error_x[:, int(20//target_sampling_period - 1)])
  
  # MAE_Y scores
  elif metric.lower() == "maey_5":
    evaluation_score = _mae(error_y[:, int(5//target_sampling_period - 1)])
  elif metric.lower() == "maey_10":
    evaluation_score = _mae(error_y[:, int(10//target_sampling_period - 1)])
  elif metric.lower() == "maey_15":
    evaluation_score = _mae(error_y[:, int(15//target_sampling_period - 1)])
  elif metric.lower() == "maey_20":
    evaluation_score = _mae(error_y[:, int(20//target_sampling_period - 1)])

  # RMSE scores
  elif metric.lower() == "rmse_5":
    evaluation_score = _rmse(error[:, int(5//target_sampling_period - 1), :])
  elif metric.lower() == "rmse_10":
    evaluation_score = _rmse(error[:, int(10//target_sampling_period - 1), :])
  elif metric.lower() == "rmse_15":
    evaluation_score = _rmse(error[:, int(15//target_sampling_period - 1), :])
  elif metric.lower() == "rmse_20":
    evaluation_score = _rmse(error[:, int(20//target_sampling_period - 1), :])

  else:
    raise ValueError("Unknown metric %s" % metric)

  return evaluation_score

def _mae(error):
  return np.mean(np.abs(error))

def _rmse(error):
  return np.sqrt(np.mean(np.square(error)))