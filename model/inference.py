from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import pickle as pkl
import pdb

from . import model as basicmodel
from . import pointnet_model
from . import model_helper
from .utils import misc_utils as utils
from .utils import evaluation_utils
from .utils import transform_utils



__all__ = ["load_data", "inference"]

def load_data(inference_input_file, hparams=None):
  """Load inference data."""
  with open(inference_input_file, 'rb') as reader:
    inference_data = pkl.load(reader)

  #if hparams.inference_indices is not None:
  #  inference_indices = hparams.inference_indices
  #  inference_data = inference_data[inference_indices]

  data_path = os.path.abspath(hparams.data_path)

  return inference_data

  # for sample in inference_data: #np.random.choice(inference_data, 128, replace=False):
  #   version = sample['dataset_version']
  #   code = sample['sample_code']
  #   frame_idx = sample['reference_time']
  #   seq_id = sample['sequence_id']
    
  #   source.append(sample['source'][:, :3])
  #   source_orientation.append([sample['source'][:, 3]])
  #   target.append(sample['target'][:, :3])
  #   sample_info.append('v{}.{}.{}_id{}'.format(version,
  #                                         code,
  #                                         frame_idx,
  #                                         seq_id))

  #   if hparams.lidar:
  #     if hparams.lidar_type == "preprocessed":
  #       ptc_path.append(os.path.join(data_path, 'processed', 'bev',
  #                       'v{}'.format(version), code,
  #                       '{:010d}.tfrecord'.format(frame_idx)))
  #     else:
  #       ptc_path.append(os.path.join(data_path, 'raw_data', 'v{}'.format(version), code,
  #                                  'Lidar', 'HDL32', 'data',
  #                                  '{:010d}.bin'.format(frame_idx)))
  #       ins_path = os.path.join(data_path, 'raw_data', 'v{}'.format(version), code,
  #                               'INS', 'data', '{:010d}.txt'.format(frame_idx))
  #       ins = np.loadtxt(ins_path)
  #       roll = ins[3]
  #       pitch = ins[4]
  #       vf_to_hf = TR(0, 0, 0, roll, pitch, 0)
  #       transform.append(vf_to_hf.dot(hdl_to_vf_list[version]))

  # src = np.array(source, dtype=np.float32)
  # src_orientation = np.array(source_orientation, dtype=np.float32)
  # tgt = np.array(target, dtype=np.float32)
  # ptc_path = np.array(ptc_path, dtype=np.string_)
  # transform = np.array(transform, dtype=np.float32)
  # reference_idx = hparams.input_length - 1
  
  # if hparams.lidar and hparams.bev_center == "target":
  #   transform[:, :3, 3] -= src[:, reference_idx, :]

  # input_dims = hparams.input_dims
  # target_dims = hparams.target_dims

  # if hparams.single_target:
  #   tgt = np.take(tgt[:, :, :target_dims], [-1], axis=1)
  # else:
  #   target_sampling_period = hparams.target_sampling_period
  #   tgt = tgt[:, target_sampling_period-1::target_sampling_period, :target_dims]
  
  # src = src[:, :, :input_dims]
  

  # if hparams.zero_centered_trajectory:
  #   tgt -= np.take(src[:, :, :target_dims], [reference_idx], axis=1)
  #   src -= np.take(src, [reference_idx], axis=1)
    
  # if hparams.orientation:
  #   src = np.concatenate((src, src_orientation), axis=-1)
  
  # sample_info = np.array(sample_info, dtype=np.string_)
  # return (src, tgt, ptc_path, transform, sample_info)

def get_model_creator(hparams):
  """Get the right model class depending on configuration."""
  if hparams.lidar:
    model_creator = pointnet_model.Model
  else:
    model_creator = basicmodel.Model
  return model_creator

def start_sess_and_load_model(infer_model, ckpt_path):
  """Start session and load model."""
  sess = tf.Session(
      graph=infer_model.graph, config=utils.get_config_proto())
  with infer_model.graph.as_default():
    loaded_infer_model = model_helper.load_model(
        infer_model.model, ckpt_path, sess, "infer")
  return sess, loaded_infer_model

def inference(ckpt_path,
              inference_input_file,
              inference_output_file,
              hparams,
              scope=None):
  """Perform translation."""
  model_creator = get_model_creator(hparams)
  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)
  sess, loaded_infer_model = start_sess_and_load_model(infer_model, ckpt_path)

  output_infer = inference_output_file

  # Read data
  src, tgt, ptc_filenames, transform, sample_info = load_data(inference_input_file, hparams)
  
  use_lidar = hparams.lidar

  iterator_feed_dict = {
      infer_model.src_placeholder: src,
      infer_model.batch_size_placeholder: hparams.infer_batch_size
  }

  if use_lidar:
    iterator_feed_dict[infer_model.ptc_placeholder] = ptc_filenames
    if hparams.lidar_type == "raw":
      iterator_feed_dict[infer_model.transform_placeholder] = transform

  with infer_model.graph.as_default():
    sess.run(
        infer_model.iterator.initializer,
        feed_dict=iterator_feed_dict)
    # Decode
    utils.print_out("# Start decoding")
    if hparams.inference_indices:
      _decode_inference_indices(
          loaded_infer_model,
          sess,
          output_infer=output_infer,
          inference_indices=hparams.inference_indices)
    else:
      _decode_and_evaluate(
          "infer",
          loaded_infer_model,
          sess,
          output_infer,
          tgt,
          metrics=hparams.metrics)

def _decode_inference_indices(model, sess, output_infer, inference_indices):
  """Decoding only a specific set of sentences."""
  utils.print_out("  regression {} , num sequences {}.".format(
      output_infer, len(inference_indices)))

  for idx in inference_indices:
    outputs, _ = model.decode(sess)
    filename = output_infer + str(idx) + ".npy"
    np.save(filename, outputs)

  utils.print_out("  done")

def _decode_and_evaluate(name,
                         model,
                         sess,
                         pred_file,
                         target,
                         metrics):
  """Decode a test set and compute a score according to the evaluation task."""
  # Decode
  utils.print_out("  decoding to output %s" % pred_file)

  decode_list = []
  num_outputs = 0
  while True:
    try:
      outputs, _ = model.decode(sess)
      batch_size = outputs.shape[0]
      num_outputs += batch_size
      decode_list.append(outputs)
      utils.print_out("  num_outputs {}".format(num_outputs), end='\r')
    except tf.errors.OutOfRangeError:
      utils.print_out("  done, num outputs {}".format(num_outputs))
      decoded = np.concatenate(decode_list, axis=0)
      break

  np.save(pred_file, decoded)

  # Evaluation
  error = target - decoded
  evaluation_scores = {}
  for metric in metrics:
    score = evaluation_utils.evaluate(
    error,
    metric)
    evaluation_scores[metric] = score
    utils.print_out("  {} {}: {:.1f}".format(metric, name, score))

  return evaluation_scores
