from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import signal
import tty
import time
from datetime import datetime
import pickle as pkl
from functools import reduce
import operator

import termios
import tensorflow as tf
import numpy as np

from .  import model as basicmodel
from .  import pointnet_model
from .  import stop_discriminator_model
from .  import model_helper
from .  import inference
from .utils import misc_utils as utils
from .utils import evaluation_utils
from .utils import iterator_utils

import pdb

SKIP_INIT_EVAL = True

__all__ = [
    "run_eval", "init_stats", "update_stats",
    "print_step_info", "process_stats", "train",
    "get_model_creator","get_best_results"
]

def run_eval(model_dir,
             eval_model,
             eval_sess,
             hparams,
             summary_writer,
             use_test_set=True):
  
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval")
  
  data_path = os.path.abspath(hparams.data_path)
  pkl_path = os.path.join(data_path, 'processed', 'pickle', hparams.trajectory_code)

  dev_file = os.path.join(pkl_path, "{}.pkl".format(hparams.dev_prefix))
  dev_dataset = inference.load_data(dev_file, hparams)
  dev_df = iterator_utils.get_eval_iterator(hparams, dev_dataset, os.path.join(model_dir, 'dev.lmdb'))
  dev_loss, dev_scores = _eval(loaded_eval_model, global_step, eval_sess,
                               hparams, dev_df, "dev", summary_writer)
  
  test_loss = None
  test_scores = None
  if use_test_set and hparams.test_prefix:
    test_file = os.path.join(pkl_path, "{}.pkl".format(hparams.test_prefix))
    test_dataset = inference.load_data(test_file, hparams)
    test_df = iterator_utils.get_eval_iterator(hparams, test_dataset, os.path.join(model_dir, 'test.lmdb'))
    test_loss, test_scores = _eval(loaded_eval_model, global_step, eval_sess,
                                   hparams, test_df, "test", summary_writer)

  metrics = {
      "dev_loss": dev_loss,
      "test_loss": test_loss,
      "dev_scores": dev_scores,
      "test_scores": test_scores}
  result_summary = _format_results("dev", dev_loss, dev_scores, hparams.metrics)
  
  if use_test_set and hparams.test_prefix:
    result_summary += _format_results("test", test_loss, test_scores,
                                             hparams.metrics)

  utils.print_out(result_summary)

  return result_summary, global_step, metrics

def init_stats():
  """Initialize statistics that we want to accumulate."""
  return {"step_time": 0.0,
          "regression_loss": 0.0,
          "grad_norm": 0.0}

def update_stats(stats, start_time, step_result, hparams):
  """Update stats: write summary and accumulate statistics."""
  _, output_tuple = step_result

  # Update statistics
  stats["step_time"] += time.time() - start_time
  stats["regression_loss"] += output_tuple.regression_loss
  stats["grad_norm"] += output_tuple.grad_norm

  return (output_tuple.global_step, output_tuple.learning_rate,
          output_tuple.train_summary)

def print_step_info(prefix, global_step, info, log_f):
  """Print all info at the current global step."""
  utils.print_out(
      "{:s} step {:d}, lr {:g}, step-time {:.2f}s, gN {:.2f}, regression_loss {:.2f}, {:s}".format(
       prefix, global_step, info["learning_rate"], info["avg_step_time"],
       info["avg_grad_norm"], info["avg_regression_loss"], time.ctime()), log_f)

def process_stats(stats, info, global_step, steps_per_stats, log_f):
  """Update info and check for overflow."""
  # Per-step info
  info["avg_step_time"] = stats["step_time"] / steps_per_stats
  info["avg_regression_loss"] = stats["regression_loss"] / steps_per_stats
  info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats

def before_train(loaded_train_model, train_model, train_sess, global_step,
                 hparams, log_f):
  """Misc tasks to do before training."""
  stats = init_stats()
  info = {"avg_step_time": 0.0,
          "avg_regression_loss": 0.0,
          "avg_grad_norm": 0.0,
          "learning_rate": loaded_train_model.learning_rate.eval(
              session=train_sess)}
  start_train_time = time.time()
  utils.print_out("# Start step {:d}, lr {:g}, {:s}".format(global_step, info["learning_rate"], time.ctime()), log_f)

  return stats, info, start_train_time

def get_model_creator(hparams):
  """Get the right model class depending on configuration."""
  if hparams.lidar:
    model_creator = pointnet_model.Model
  elif hparams.stop_discriminator:
    model_creator = stop_discriminator_model.Model
  else:
    model_creator = basicmodel.Model
  return model_creator

def train(hparams, scope=None):
  log_device_placement = hparams.log_device_placement
  model_dir = hparams.model_dir
  num_train_epochs = hparams.num_train_epochs
  steps_per_stats = hparams.steps_per_stats
  evals_per_epoch = hparams.evals_per_epoch

  model_creator = get_model_creator(hparams)
  train_model = model_helper.create_train_model(model_creator, hparams, scope)
  eval_model = model_helper.create_eval_model(model_creator, hparams, scope)

  # Log and output files
  datecode = datetime.now().strftime("%Y%m%d.%H%M")
  log_file = os.path.join(model_dir, "log_{}".format(datecode))
  log_f = tf.gfile.GFile(log_file, mode="a")
  utils.print_out("# log_file={}".format(log_file))

  # TensorFlow model
  config_proto = utils.get_config_proto(
      log_device_placement=log_device_placement)

  train_sess = tf.Session(config=config_proto, graph=train_model.graph)
  eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
  
  pkl_path = os.path.join(hparams.data_path, 'processed', 'pickle', hparams.trajectory_code)

  train_file = os.path.join(pkl_path, "{}.pkl".format(hparams.train_prefix))

  train_dataset = inference.load_data(train_file, hparams)
  train_df = iterator_utils.get_iterator(hparams, train_dataset, os.path.join(model_dir, 'train.lmdb'), shuffle=True, drop_remainder=True, nr_proc=4)

  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, model_dir, train_sess, "train")
  # Summary writer
  summary_writer = tf.summary.FileWriter(
      model_dir, train_model.graph)

  if not SKIP_INIT_EVAL:
    utils.print_out("# Beginning the test evaluation.")
    run_eval(model_dir, eval_model, eval_sess, hparams, summary_writer)

  last_stats_step = global_step

  stats, info, start_train_time = before_train(
      loaded_train_model, train_model, train_sess, global_step, hparams, log_f)

  utils.print_out("# Initialize train iterator")
  train_df.reset_state()
  utils.print_out("# {} batches are ready!".format(len(train_df)))
  steps_per_eval = len(train_df) // evals_per_epoch + 1
  last_eval_step = global_step

  # Get placeholders
  placeholders = loaded_train_model.placeholders
  source_ph = list(filter(lambda x: "source_placeholder" in x.name, placeholders))
  target_ph = list(filter(lambda x: "target_placeholder" in x.name, placeholders))
  ph_list = source_ph + target_ph
  if hparams.stop_discriminator:
    ph_list += list(filter(lambda x: "stop_placeholder" in x.name, placeholders))
  
  batch_size_ph = list(filter(lambda x: "batch_size" in x.name, placeholders))
  ph_list += batch_size_ph

  while hparams.epoch < num_train_epochs:

    for batches in train_df.get_data():
      start_time = time.time()

      # feed dict
      batch_sizes = [batch.shape[0] for batch in batches[0]]
      feed_dict={key:value for (key, value) in zip(
          ph_list,
          reduce(operator.add, batches) + batch_sizes)}

      # Train step
      step_result = loaded_train_model.train(train_sess, feed_dict=feed_dict)
      global_step, info["learning_rate"], step_summary = update_stats(stats, start_time, step_result, hparams)
  
      # Once in a while, write summaries out statistics.
      if global_step - last_stats_step >= steps_per_stats:
        last_stats_step = global_step
        summary_writer.add_summary(step_summary, global_step)
        process_stats(stats, info, global_step, steps_per_stats, log_f)
        print_step_info("# Train, ", global_step, info, log_f)
        # Reset statistics
        stats = init_stats()
      
      if global_step - last_eval_step >= steps_per_eval:
        loaded_train_model.saver.save(
            train_sess,
            os.path.join(model_dir, "model.ckpt"),
            global_step=global_step)

        run_eval(model_dir, eval_model, eval_sess, hparams, summary_writer)
        last_eval_step = global_step
    
    # Finished going through the training dataset.  Go to next epoch.
    hparams.epoch += 1
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(model_dir, "model.ckpt"),
        global_step=global_step)
    
    utils.print_out("# Reached {:d} epoch in step {:d}.".format(hparams.epoch, global_step), end=" ")
    run_eval(model_dir, eval_model, eval_sess, hparams, summary_writer)
    last_eval_step = global_step

    # learning rate decay
    if hparams.epoch in hparams.learning_rate_decay_epochs:
      utils.print_out("# Decaying learning rate {} >> {}".format(info["learning_rate"], info["learning_rate"] * hparams.learning_rate_decay_ratio))
      loaded_train_model.learning_rate_decay(train_sess, hparams.learning_rate_decay_ratio)
    
  # Done training
  loaded_train_model.saver.save(
      train_sess,
      os.path.join(model_dir, "model.ckpt"),
      global_step=global_step)

  _, _, final_eval_metrics = run_eval(model_dir, eval_model, eval_sess, hparams, summary_writer)

  print_step_info("# Final, ", global_step, info, log_f)
  utils.print_out("# Done training!")

  summary_writer.close()

  return final_eval_metrics, global_step

def _format_results(name, losses, scores, metrics):
  """Format results."""
  result_str = ""
  for losses_key in losses:
    result_str += "{} {}: {:.3f}\n".format(name, losses_key, losses[losses_key])
  for metric in metrics:
    result_str += "{} {} {:.3f}\n".format(name, metric, scores[metric])
  return result_str

def get_best_results(hparams):
  """Summary of the current best results."""
  tokens = []
  for metric in hparams.metrics:
    tokens.append("{} {:.2f}".format(metric, getattr(hparams, "best_" + metric)))
  return ", ".join(tokens)

def _eval(model, global_step, sess, hparams, dataflow,
          label, summary_writer, save_on_best=True):
  """Compute loss and external metrics"""
  model_dir = hparams.model_dir
  output_file = os.path.join(model_dir, "output_{}.ckpt-{}.pkl".format(label, global_step))
  metrics = hparams.metrics
  
  trained = global_step > 0

  utils.print_out("# Initialize {} iterator".format(label))
  dataflow.reset_state()
  utils.print_out("# {} batches are ready!".format(len(dataflow)))

  prediction, gt, losses = model_helper.compute_loss_and_predict(
      hparams, model, sess, dataflow, label)
  
  for key in losses.keys():
    utils.add_summary(summary_writer, global_step, label+'_'+key, losses[key])

  scores = {}
  error = gt - prediction
  for metric in metrics:
    score = evaluation_utils.evaluate(hparams, error, metric)
    scores[metric] = score

  if trained:
    with open(output_file, 'wb') as writer:
      pkl.dump(prediction, writer)

    for metric in metrics:
      best_metric_label = "best_{}_".format(label) + metric
      utils.add_summary(summary_writer, global_step, label+"_{}".format(metric),
                        scores[metric])

      # metric: smaller is better
      if save_on_best and scores[metric] < float(getattr(hparams, best_metric_label)):
        setattr(hparams, best_metric_label, "{:.4f}".format(scores[metric]))
        with open(os.path.join(getattr(hparams, best_metric_label + "_dir"), "ckpt.txt"), 'a') as writer:
          writer.write(str(model.global_step) + '\n')

    utils.save_hparams(model_dir, hparams)
  return losses, scores