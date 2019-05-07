from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import os
import random
import sys
import shutil

import numpy as np
import tensorflow as tf

from .  import inference
from .  import train
from .utils import evaluation_utils
from .utils import misc_utils as utils

import pdb

FLAGS = None

def _bool(s):
  v = (s.lower() == "true") or (s != "0")
  return v

def _list(s):
  v = ast.literal_eval(s)
  if type(v) is not list:
    raise argparse.ArgumentTypeError("Argument \"{}\" is not a list".format(s))
  return v

def add_arguments(parser):
  """ Build argument parser"""
  parser.register("type", "bool", _bool)
  parser.register("type", "list", _list)

  ## 1. Model and data locations
  parser.add_argument("--model_name", type=str, default="model",
                      help="Model name")
  parser.add_argument("--data_path", type=str, default="./dataset",
                      help="dataset path")
  parser.add_argument("--trajectory_code", type=str, default="filled/absolute_50",
                      help="preprocessed trajecotory code")
  parser.add_argument("--train_prefix", type=str, default="train",
                      help="Train dataset prefix")
  parser.add_argument("--dev_prefix", type=str, default="dev",
                      help="Dev dataset prefix")
  parser.add_argument("--test_prefix", type=str, default="test",
                      help="Test dataset prefix")
  parser.add_argument("--out_dir", type=str, default="log",
                      help="Directory to store log/model/evaluation files.")
  parser.add_argument("--ramdisk_dir", type=str,
                      help="Ramdisk path (optional).")

  ## 2. hardware parameters
  parser.add_argument("--gpu_id", type=str, default="0",
                      help="gpu_ids to use for training (comma-separated).")
  parser.add_argument("--num_cpu", type=int, default=20,
                      help="Number of CPUs.")
  parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?", default=True,
                      help=("Try to colocate gradients with corresponding op"))

  ## 3. Sequence pre-processing parameters
  parser.add_argument("--input_dims", type=int, default=2,
                      help="left for the compatibility (will soon be removed)")
  parser.add_argument("--target_dims", type=int, default=2,
                      help="left for the compatibility (will soon be removed)")

  parser.add_argument("--trajectory_dims", type=int, default=2,
                      help="trajectory diemension 2: (x,y) 3: (x,y,z)..")
  parser.add_argument("--input_length", type=int, default=30,
                      help="input (observation) length.")
  parser.add_argument("--target_length", type=int, default=20,
                      help="target length.")
  parser.add_argument("--target_sampling_period", type=int, default=1,
                      help="""Sampling period to apply on the raw target.
                            \"target length\" must be divisible by this.""")
  # parser.add_argument("--relative", action='store_true', help="Use relative trajectory. In this case, input_dims is force set to 4:(x,y,ego_vel,ego_steer).")
  
  parser.add_argument("--single_target", action='store_true', help="One shot prediction to the latest target.")
  parser.add_argument("--single_target_horizon", type=int, default=20,
                      help="One shot prediction to a target frame.")

  
  # parser.add_argument("--orientation", action='store_true',
  #                     help="Concatenate orientation to input. The input dimension is increased by one if true.")
  parser.add_argument("--zero_centered_trajectory", action='store_true',
                      help=("Whether to zero center the trajectory"
                            "by subtracting the final observation location."))
  parser.add_argument("--polar_representation", action='store_true',
                      help="new")

  ## 4. lidar inputs
  parser.add_argument("--lidar", action='store_true', help="Use lidar pointcloud fusion.")
  parser.add_argument("--lidar_type", type=str, default="raw",
                      help="raw | bev")
  
  #### 4.1. Raw point cloud preprocessing parameters 
  parser.add_argument("--raw_center", type=str, default="target",
                      help="target | ego.")
  parser.add_argument("--raw_lon_range", type="list", nargs="?", default=[100, 100],
                      help="""Longitudinal (forward and back of the vehicle) ranges around\
                      target vehicle to generate bev. [back, forward] (m).""")
  parser.add_argument("--raw_lat_range", type="list", nargs="?", default=[100, 100],
                      help="""Lateral (left and right of the vehicle) ranges around\
                      target vehicle to generate bev. [left, right] (m).""")
  parser.add_argument("--raw_height_range", type="list", nargs="?", default=[10, 10],
                      help="""Height (up and down of the vehicle) ranges around\
                      target vehicle to generate bev. [up, down] (m).""")
  parser.add_argument("--num_point", type=int, default=20000,
                      help="""number of points per sample.\
                      Zero-padding or sampling is applied to match the num_point.""")

  #### 4.2. Bird's Eye View preprocessing parameters
  parser.add_argument("--bev_center", type=str, default="target",
                      help="target | ego. Must be target when zero_centered_trajectory is True.")
  parser.add_argument("--bev_lon_range", type="list", nargs="?", default=[50, 50],
                      help="""Longitudinal (forward and back of the vehicle) ranges around\
                      target vehicle to generate bev. [back, forward] (m).""")
  parser.add_argument("--bev_lat_range", type="list", nargs="?", default=[50, 50],
                      help="""Lateral (left and right of the vehicle) ranges around\
                      target vehicle to generate bev. [left, right] (m).""")
  parser.add_argument("--bev_height_range", type="list", nargs="?", default=[5, 5],
                      help="""Height (up and down of the vehicle) ranges around\
                      target vehicle to generate bev. [up, down] (m).""")
  parser.add_argument("--bev_res", type="list", nargs="?", default=[0.1, 0.1, 0.1],
                      help="Bev Resolution [x, y, z] (m / voxel)")
  
  ## 5. trajectory encoder parameters
  parser.add_argument("--encoder_type", type=str, default="rnn",
                      help="rnn | cnn")

  #### 5.1. rnn encoder parameters
  parser.add_argument("--rnn_encoder_layers", type=int, default=1,
                      help="number or rnn layers.")
  parser.add_argument("--rnn_encoder_units", type=int, default=48,
                      help="rnn unit size.")
  parser.add_argument("--rnn_encoder_type", type=str, default="cudnn_lstm",
                      help="cudnn_lstm | lstm (not implemented) | cudnn_gru (not implemented) | gru (not implemented)")
  parser.add_argument("--input_projector_type", type=str, default="cnn",
                      help="fc | cnn")
  parser.add_argument("--relu_reconfiguration", action='store_true', help="use relu reconfiguration to the rnn output")
  
  ###### 5.1.1. fc projector parameters
  parser.add_argument("--fc_input_projector_units", type="list", default=[8, 16],
                      help="List of the input projector unit sizes in order.")
  ###### 5.1.2. cnn projector parameters
  parser.add_argument("--cnn_input_projector_filters", type="list", default=[16],
                      help="List of the cnn input projector filter numbers in order.")
  parser.add_argument("--cnn_input_projector_kernels", type="list", default=[3],
                      help="List of the input projector kernel sizes in order.")

  #### 5.2. cnn encoder parameters
  parser.add_argument("--cnn_encoder_filters", type="list", default=[16,24,32,48],
                      help="List of the cnn filter numbers in order.")
  parser.add_argument("--cnn_encoder_kernels", type="list", default=[5, 5, 5, 3],
                      help="List of the cnn kernel sizes in order.")
  parser.add_argument("--cnn_encoder_dilation_rates", type="list", default=[2, 2, 2, 2],
                      help="List of the cnn dilation rates in order.")

  ## Stop discriminator
  parser.add_argument("--stop_discriminator", action='store_true',
                      help="new")
  parser.add_argument("--stop_discriminator_units", type="list", default=[32, 16, 1],
                      help="List of the stop discriminator unit sizes in order.")

  ## 6. trajectory decoder parameters
  parser.add_argument("--decoder_type", type=str, default="rnn",
                      help="fc | rnn. fc decoder is applicable only if \"single_target\" is True")

  #### 6.1. rnn decoder parameters
  parser.add_argument("--rnn_decoder_inputs", type=str, default="dummy",
                      help="dummy | encoder_output")
  parser.add_argument("--rnn_decoder_layers", type=int, default=1,
                      help="number or rnn layers.")
  parser.add_argument("--rnn_decoder_units", type=int, default=48,
                      help="rnn unit size.")
  parser.add_argument("--rnn_decoder_type", type=str, default="cudnn_lstm",
                      help="cudnn_lstm | lstm (not implemented) | cudnn_gru (not implemented) | gru (not implemented)")
  parser.add_argument("--output_projector_units", type="list", default=[16, 2],
                      help="List of the output projector unit sizes in order.")
  
  #### 6.2. fc decoder
  parser.add_argument("--fc_decoder_units", type="list", default=[64, 2],
                      help="List of the output projector unit sizes in order.")

  ## 7. Training parameters
  parser.add_argument("--optimizer", type=str, default="adam", help="sgd | adam")
  parser.add_argument("--max_gradient_norm", type=float, default=None,
                      help="""clip gradients to this norm\
                      (recommended if rnn encoder-decoder is used).""")
  parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Initial learning rate. Adam: 0.001 | 0.0001")
  parser.add_argument("--weight_decay_factor", type=float, default=0.003,
                    help="weight_decay_factor.")
  parser.add_argument("--rnn_weight_decay_factor", type=float, default=0.003,
                    help="rnn_weight_decay_factor.")
  parser.add_argument("--loss", type=str, default="weighted_smooth_l1",
                      help="l2 | weighted_smooth_l1")
  parser.add_argument("--num_train_epochs", type=int, default=50,
                      help="Num epochs to train.")
  parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
  parser.add_argument("--bn_init_decay", type=float, default=0.5,
                      help="bn_init_decay.")
  parser.add_argument("--bn_decay_step", type=int, default=250000,
                      help="bn_decay_step.")
  parser.add_argument("--bn_decay_rate", type=float, default=0.5,
                      help="bn_decay_rate.")
  parser.add_argument("--bn_decay_clip", type=float, default=0.99,
                      help="bn_decay_clip.")
  parser.add_argument("--learning_rate_decay_epochs", type="list", default=[15,30,45])
  parser.add_argument("--learning_rate_decay_ratio", type=float, default=0.5)

  ## 8. Misc
  parser.add_argument("--log_device_placement", type="bool", nargs="?",
                      default=False, help="Debug GPU allocation")
  parser.add_argument("--steps_per_stats", type=int, default=100,
                      help="How many training steps to do per stats logging.")
  parser.add_argument("--evals_per_epoch", type=int, default=1,
                      help="How many evals to do per epoch.")
  parser.add_argument("--metrics", type=str,
                      default=("maex_5,maex_10,maex_15,maex_20,"
                               "maey_5,maey_10,maey_15,maey_20,"
                               "rmse_5,rmse_10,rmse_15,rmse_20"))
  parser.add_argument("--random_seed", type=int, default=None,
                      help="Random seed")
  parser.add_argument("--num_keep_ckpts", type=int, default=1000,
                      help="Max number of checkpoints to save.")

  # Inference
  parser.add_argument("--ckpt", type=int, default=None,
                      help="Checkpoint number to load a model for inference.")
  parser.add_argument("--inference_input_file", type=str, default=None,
                      help="Sequence to predict")
  parser.add_argument("--inference_output_file", type=str, default=None,
                      help="Output file to store decoding results.")
  parser.add_argument("--infer_batch_size", type=int, default=128,
                      help="Batch size for inference mode.")
  parser.add_argument("--infer_gpu_id", type=str, default="0",
                      help="gpu_ids to use for inference (comma-separated).")


def create_hparams(flags):
  """ Create training hyper parameters """
  hparams = tf.contrib.training.HParams(
      # 1. Model and data locations
      model_name=flags.model_name,
      data_path=flags.data_path,
      trajectory_code=flags.trajectory_code,
      train_prefix=flags.train_prefix,
      dev_prefix=flags.dev_prefix,
      test_prefix=flags.test_prefix,
      ramdisk_dir=flags.ramdisk_dir,
      # 2. hardware parameters
      gpu_id=flags.gpu_id,
      num_cpu=flags.num_cpu,
      colocate_gradients_with_ops=flags.colocate_gradients_with_ops,
      # 3. Sequence pre-processing parameters
      input_length=flags.input_length,
      target_length=flags.target_length,
      trajectory_dims=flags.trajectory_dims,
      single_target=flags.single_target,
      # orientation=flags.orientation,
      zero_centered_trajectory=flags.zero_centered_trajectory,
      polar_representation=flags.polar_representation,
      # 4. lidar inputs
      lidar=flags.lidar,
      # 5. trajectory encoder parameters
      encoder_type=flags.encoder_type,

      stop_discriminator=flags.stop_discriminator,
  
      # 6. trajectory decoder parameters
      decoder_type=flags.decoder_type,
      # 7. Training parameters
      optimizer=flags.optimizer,
      max_gradient_norm=flags.max_gradient_norm,
      learning_rate=flags.learning_rate,
      weight_decay_factor=flags.weight_decay_factor,
      rnn_weight_decay_factor=flags.rnn_weight_decay_factor,
      loss=flags.loss,
      num_train_epochs=flags.num_train_epochs,
      batch_size=flags.batch_size,
      bn_init_decay=flags.bn_init_decay,
      bn_decay_step=flags.bn_decay_step,
      bn_decay_rate=flags.bn_decay_rate,
      bn_decay_clip=flags.bn_decay_clip,
      learning_rate_decay_epochs=flags.learning_rate_decay_epochs,
      learning_rate_decay_ratio=flags.learning_rate_decay_ratio,
    
      # 8. Misc
      log_device_placement=flags.log_device_placement,
      steps_per_stats=flags.steps_per_stats,
      evals_per_epoch=flags.evals_per_epoch,
      metrics=flags.metrics.split(","),
      random_seed=flags.random_seed,
      num_keep_ckpts=flags.num_keep_ckpts)
  
  ## Set derived parmeters
  # num_gpu
  _add_argument(hparams, "num_gpu", len(flags.gpu_id.split(",")))
  if hparams.batch_size % hparams.num_gpu != 0:
    raise ValueError("""batch_size {:d} is not evenly divisible\
                     by num_gpu {:d}.""".format(hparams.batch_size, hparams.num_gpu))
  
  # if hparams.relative:
  #   if hparams.zero_centered_trajectory:
  #     raise ValueError("zero_centered_trajectory cannot be set for relative trajectory.")
  #   _add_argument(hparams, "input_dims", 4)
  if hparams.trajectory_dims == 2:
    _add_argument(hparams, "input_dims", 2)
    _add_argument(hparams, "target_dims", 2)
  elif hparams.trajectory_dims == 3:
    _add_argument(hparams, "input_dims", 3)
    _add_argument(hparams, "target_dims", 3)
  else:
    raise ValueError("trajectory_dims must be \"2\" or \"3\".")

  if hparams.polar_representation:
    _add_argument(hparams, "input_dims", hparams.input_dims + 3) # l2 distance, cos, sin
    _add_argument(hparams, "target_dims", hparams.target_dims + 3)
    _add_argument(hparams, "input_length", hparams.input_length - 1)
    
  # encoder settings
  if hparams.encoder_type == "rnn":
    _add_argument(hparams, "rnn_encoder_layers", flags.rnn_encoder_layers)
    _add_argument(hparams, "rnn_encoder_units", flags.rnn_encoder_units)
    _add_argument(hparams, "rnn_encoder_type", flags.rnn_encoder_type)
    _add_argument(hparams, "input_projector_type", flags.input_projector_type)
    _add_argument(hparams, "relu_reconfiguration", flags.relu_reconfiguration)
    if hparams.input_projector_type == "fc":
      _add_argument(hparams, "fc_input_projector_units", flags.fc_input_projector_units)
    elif hparams.input_projector_type == "cnn":
      _add_argument(hparams, "cnn_input_projector_filters", flags.cnn_input_projector_filters)
      _add_argument(hparams, "cnn_input_projector_kernels", flags.cnn_input_projector_kernels)  
    else:
      raise ValueError("Unknown input_projector_type {s}".format(hparams.input_projector_type))
  elif hparams.encoder_type == "cnn":
    _add_argument(hparams, "cnn_encoder_filters", flags.cnn_encoder_filters)
    _add_argument(hparams, "cnn_encoder_kernels", flags.cnn_encoder_kernels)
    _add_argument(hparams, "cnn_encoder_dilation_rates", flags.cnn_encoder_dilation_rates)
  else:
    raise ValueError("Unknown encoder_type {s}".format(hparams.encoder_type))
  
  # Stop discriminator
  if hparams.stop_discriminator:
    _add_argument(hparams, "stop_discriminator_units", flags.stop_discriminator_units)

  # decoder settings
  if hparams.decoder_type == "rnn":
    _add_argument(hparams, "rnn_decoder_inputs", flags.rnn_decoder_inputs)
    _add_argument(hparams, "rnn_decoder_layers", flags.rnn_decoder_layers)
    _add_argument(hparams, "rnn_decoder_units", flags.rnn_decoder_units)
    _add_argument(hparams, "rnn_decoder_type", flags.rnn_decoder_type)
    _add_argument(hparams, "output_projector_units", flags.output_projector_units)
  elif hparams.decoder_type == "fc":
    _add_argument(hparams, "fc_decoder_units", flags.fc_decoder_units)
  else:
    raise ValueError("Unknown decoder_type {s}".format(hparams.decoder_type))

  # target trajectory
  if hparams.single_target:
    _add_argument(hparams, "single_target_horizon", flags.single_target_horizon)
    if hparams.single_target_horizon > hparams.target_length:
      raise ValueError("target horizon cannot be longer then target length.")
    _add_argument(hparams, "target_length", 1)
    
  else:
    # No fc decoder
    if hparams.decoder_type == "fc":
      raise ValueError("Cannot use fc decoder if single_target==False")

    # Target sampling period
    _add_argument(hparams, "target_sampling_period", flags.target_sampling_period)
    if hparams.target_length % hparams.target_sampling_period != 0:
      raise ValueError("""target_length {:d} is not evenly divisible\
                       by target_sampling_period {:d}.""".format(hparams.target_length, hparams.target_sampling_period))
    _add_argument(hparams, "target_length", int(hparams.target_length // hparams.target_sampling_period))

  # lidar settings
  if hparams.lidar:
    _add_argument(hparams, "lidar_type", flags.lidar_type)
    if hparams.lidar_type == "raw":
      _add_argument(hparams, "raw_center", flags.raw_center)
      _add_argument(hparams, "raw_lon_range", flags.raw_lon_range)
      _add_argument(hparams, "raw_lat_range", flags.raw_lat_range)
      _add_argument(hparams, "raw_height_range", flags.raw_height_range)
      _add_argument(hparams, "num_point", flags.num_point)
    elif hparams.lidar_type == "bev":
      _add_argument(hparams, "bev_center", flags.bev_center)
      _add_argument(hparams, "bev_lon_range", flags.bev_lon_range)
      _add_argument(hparams, "bev_lat_range", flags.bev_lat_range)
      _add_argument(hparams, "bev_height_range", flags.bev_height_range)
      _add_argument(hparams, "bev_res", flags.bev_res)
    else:
      assert ValueError("Unknown lidar_type {s}".format(hparams.lidar_type))
  
  return hparams

def _add_argument(hparams, key, value, update=True):
  """Add an argument to hparams; if exists, change the value if update==True."""
  if hasattr(hparams, key):
    if update:
      setattr(hparams, key, value)
  else:
    hparams.add_hparam(key, value)

def extend_hparams(hparams, model_dir):
  """Add new arguments to hparams."""
  # model_dir
  _add_argument(hparams, "model_dir", model_dir, update=False)
  # epoch
  _add_argument(hparams, "epoch", 0, update=False)

  # Evaluation
  for metric in hparams.metrics:
    best_dev_dir = os.path.join(model_dir, "best_dev_" + metric)
    best_test_dir = os.path.join(model_dir, "best_test_" + metric)
    if not tf.gfile.Exists(best_dev_dir): tf.gfile.MakeDirs(best_dev_dir)
    if not tf.gfile.Exists(best_test_dir): tf.gfile.MakeDirs(best_test_dir)
    _add_argument(hparams, "best_dev_" + metric, "999", update=False)
    _add_argument(hparams, "best_dev_" + metric + "_dir", best_dev_dir)
    _add_argument(hparams, "best_test_" + metric, "999", update=False)
    _add_argument(hparams, "best_test_" + metric + "_dir", best_test_dir)

  return hparams

def create_or_load_hparams(
    model_dir, default_hparams, save_hparams=True):
  """Create hparams or load hparams from model_dir."""
  hparams = utils.load_hparams(model_dir)
  if not hparams: # case that hparams file not exist of not valid.
    hparams = default_hparams

  hparams = extend_hparams(hparams, model_dir)

  if save_hparams:
    utils.save_hparams(model_dir, hparams)

  # Print HParams
  utils.print_hparams(hparams)
  return hparams

def run_main(flags, default_hparams, train_fn, inference_fn):
  """Run main."""

  # Random seed
  random_seed = flags.random_seed
  if random_seed is not None:
    print("# Set random seed to %d" % random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

  # Model Directories
  out_dir = os.path.abspath(flags.out_dir)
  model_dir = os.path.join(out_dir, flags.model_name)
  
  if not tf.gfile.Exists(model_dir): tf.gfile.MakeDirs(model_dir)

  # Load hparams
  hparams = create_or_load_hparams(
      model_dir, default_hparams)
  
  # Inference
  if flags.inference_input_file:
    # Inference GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.infer_gpu_id
    num_gpu = len(flags.infer_gpu_id.split(","))
    # Inference batch size
    batch_size = flags.infer_batch_size
    # CheckPoint code
    ckpt = flags.ckpt
    # Inference input file
    input_file = flags.inference_input_file
    # Inference output file
    output_file = flags.inference_output_file
    output_dir = os.path.dirname(output_file)
    if not tf.gfile.Exists(output_dir):
      tf.gfile.MakeDirs(output_dir)

    # Do inference
    inference_fn(hparams, model_dir, ckpt, input_file, output_file, num_gpu, batch_size)

  # Train
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpu_id
    # Backup model codes
    code_dir = os.path.join(model_dir, 'model_code')

    if tf.gfile.Exists(code_dir):
      copy_idx = 0
      while(True):
        copy_dir = os.path.join(code_dir, '_{:d}'.format(copy_idx))
        if tf.gfile.Exists(copy_dir):
          copy_idx += 1
        else:
          shutil.copytree(os.path.abspath('./model'), copy_dir)
          break
    else:
      shutil.copytree(os.path.abspath('./model'), code_dir)
    
    # Do train
    train_fn(hparams)

def main(unused_argv):
  default_hparams = create_hparams(FLAGS)
  train_fn = train.train
  inference_fn = inference.inference
  run_main(FLAGS, default_hparams, train_fn, inference_fn)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main)
