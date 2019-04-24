from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = ["evaluate"]

def evaluate(error, metric):
  """Pick a metric and evaluate depending on task."""
  error_x = error[:, :, 0]
  error_y = error[:, :, 1]

  # MAE_X scores
  if metric.lower() == "maex_5":
    evaluation_score = _mae(error_x[:, 4])
  elif metric.lower() == "maex_10":
    evaluation_score = _mae(error_x[:, 9])
  elif metric.lower() == "maex_15":
    evaluation_score = _mae(error_x[:, 14])
  elif metric.lower() == "maex_20":
    evaluation_score = _mae(error_x[:, 19])
  elif metric.lower() == "maex_oneshot":
    evaluation_score = _mae(error_x[:, 0])
  
  # MAE_Y scores
  elif metric.lower() == "maey_5":
    evaluation_score = _mae(error_y[:, 4])
  elif metric.lower() == "maey_10":
    evaluation_score = _mae(error_y[:, 9])
  elif metric.lower() == "maey_15":
    evaluation_score = _mae(error_y[:, 14])
  elif metric.lower() == "maey_20":
    evaluation_score = _mae(error_y[:, 19])
  elif metric.lower() == "maey_oneshot":
    evaluation_score = _mae(error_y[:, 0])

  # RMSE scores
  elif metric.lower() == "rmse_5":
    evaluation_score = _rmse(error[:, 4, :])
  elif metric.lower() == "rmse_10":
    evaluation_score = _rmse(error[:, 9, :])
  elif metric.lower() == "rmse_15":
    evaluation_score = _rmse(error[:, 14, :])
  elif metric.lower() == "rmse_20":
    evaluation_score = _rmse(error[:, 19, :])
  elif metric.lower() == "rmse_oneshot":
    evaluation_score = _rmse(error[:, 0, :])

  else:
    raise ValueError("Unknown metric %s" % metric)

  return evaluation_score

def _mae(error):
  return np.mean(np.abs(error))

def _rmse(error):
  return np.sqrt(np.mean(np.square(error)))