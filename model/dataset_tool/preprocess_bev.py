import tensorflow as tf
import numpy as np

import random
import os
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

import pdb

import transform_utils
import visualize_utils


"""
Read the raw HDL32 pointcloud, then make BEV(Bird's Eye View == Top-down view) feature map in vehicle horizontal frame.
Then save it into GZIP compressed TFRecord.
"""

data_path = os.path.abspath('../../dataset')
raw_path = os.path.join(data_path, 'raw_data')
bev_path = os.path.join(data_path, 'processed', 'bev')

FORWARD = 60
LEFT = 60
HEIGHT = 4
RESOLUTION = 0.25

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

hdl_to_vf_list = {'v1': transform_utils.TR(193.42, 0, -175, 0.0286, 180.50, -1.34, scale_factor=0.01, degrees=True),
                  'v2': transform_utils.TR(203, -2.5, -150, 179.4, 0.15, -91.3, scale_factor=0.01, degrees=True)}

versions = ['v1', 'v2']
for v in versions:
  scenario_list = os.listdir(os.path.join(raw_path, v))

  for scenario in scenario_list:

    print("Working on {}_{}...".format(v, scenario))

    ins_path = os.path.join(raw_path, v, scenario, 'INS', 'data')
    hdl_path = os.path.join(raw_path, v, scenario, 'Lidar', 'HDL32', 'data')

    num_frames = len(os.listdir(ins_path))

    for frame_idx in tqdm(range(num_frames)):
      ins_frame = np.loadtxt(
          os.path.join(ins_path, '{:010d}.txt'.format(frame_idx)))
      roll, pitch = ins_frame[3:5]
  
      hdl_to_vf = hdl_to_vf_list[v].T
      vf_to_hf = transform_utils.TR(0, 0, 0, roll, pitch, 0).T

      hdl_to_hf = hdl_to_vf.dot(vf_to_hf)

      ptc = np.fromfile(
          os.path.join(hdl_path, '{:010d}.bin'.format(frame_idx)),
          dtype=np.int16)
      ptc = ptc.reshape(-1, 4).astype(np.float32) # Data type conversion
      ptc[:, :3] /= 100.0 # Convert to meter
      ptc[:, 3] = 1.0 # Convenience for rigid body transforms
      
      ptc_hf = ptc.dot(hdl_to_hf)

      bev = visualize_utils.bird_eye_view(ptc_hf[:, :3], Forward=FORWARD, Left=LEFT, Height=HEIGHT, res=RESOLUTION)

      save_path = Path(bev_path, v, scenario)
      save_path.mkdir(parents=True, exist_ok=True)
      
      writer_option = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.GZIP)
      writer = tf.python_io.TFRecordWriter(
          str(save_path.joinpath('{:010d}.tfrecord'.format(frame_idx))),
          options=writer_option)

      record_dict = {"bev_data": _bytes_feature(bev.tobytes())}
      tf_example = tf.train.Example(features=tf.train.Features(feature=record_dict))
      writer.write(tf_example.SerializeToString())
      writer.close()