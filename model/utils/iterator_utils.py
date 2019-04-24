from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections

import numpy as np
from tensorpack.dataflow.parallel_map import MultiThreadMapData
from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow.parallel import PrefetchData, PrefetchDataZMQ
from tensorpack.dataflow.raw import DataFromList
from tensorpack.dataflow.serialize import LMDBSerializer

import pdb

transform_tuple = collections.namedtuple("transform_tuple", "v1 v2")

def TR(tx, ty, tz, rx, ry, rz, scale_factor=1.0, degrees=False):
  """ Build affine transform matrix.
  Args
    tx: x-axis translation
    ty: y-axis translation
    tz: z-axis translation
    rx: roll angle
    ry: pitch angle
    rz: yaw angle
    scale_factor: scale factor to multiply to translation.
    degrees: Whether the angles are in degree 
  Return
    4 by 4 affine transform matrix.
  """
  def _rotation_mat(rx, ry, rz, degrees):
    if degrees:
      [rx, ry, rz] = np.radians([rx, ry, rz])

    c, s = np.cos(rx), np.sin(rx)                                                                                                            
    rx_mat = np.array([1, 0, 0, 0, c, -s, 0, s, c]).reshape(3, 3)                                                                            
    c, s = np.cos(ry), np.sin(ry)                                                                                                            
    ry_mat = np.array([c, 0, s, 0, 1, 0, -s, 0, c]).reshape(3, 3)                                                                            
    c, s = np.cos(rz), np.sin(rz)                                                                                                            
    rz_mat = np.array([c, -s, 0, s, c, 0, 0, 0, 1]).reshape(3, 3)                                                                            
    return rz_mat.dot(ry_mat.dot(rx_mat))

  def _translation_mat(tx, ty, tz, scale_factor):
    return np.array([[tx], [ty], [tz]]) * scale_factor

  R = _rotation_mat(rx, ry, rz, degrees=degrees)
  T = _translation_mat(tx, ty, tz, scale_factor=scale_factor)

  return np.block([[R, T], [0, 0, 0, 1]])

hdl_to_vfs = transform_tuple(TR(193.42, 0, -175, 0.0286, 180.50, -1.34, scale_factor=1.0, degrees=True),
                             TR(203, -2.5, -150, 179.4, 0.15, -91.3, scale_factor=1.0, degrees=True))

def map_func(dataset, hparams):
  source = dataset['source']
  target = dataset['target']
  version = dataset['dataset_version']
  code = dataset['sample_code']
  frame = dataset['reference_time']
  sequence_id = dataset['sequence_id']
  
  data_path = os.path.abspath(hparams.data_path)

  reference_idx = hparams.input_length - 1
  src_ref = source[reference_idx, :3].copy()  # deep copy source at ref idx (to be used in lidar processing).
  
  if hparams.relative:
    src = source[:, [0,1,3,4]]
    
  else:
    src = source[:, :3] # source in (m) metric

  input_dims = hparams.input_dims
  target_dims = hparams.target_dims

  if hparams.single_target:
    tgt = np.take(target[:, :target_dims], [hparams.single_target_horizon - 1], axis=0)
  else:
    target_sampling_period = hparams.target_sampling_period
    tgt = target[target_sampling_period-1::target_sampling_period, :target_dims]
  
  src = src[:, :input_dims]

  if hparams.zero_centered_trajectory:
    tgt -= np.take(src[:, :target_dims], [reference_idx], axis=0)
    src -= np.take(src, [reference_idx], axis=0)

  lidar = None
  # if raw lidar point cloud
  if hparams.lidar:
    # File paths
    ptc_path = os.path.join(data_path, 'raw_data', 'v{}'.format(version), code, 'Lidar', 'HDL32', 'data', '{:010d}.bin'.format(frame))
    ptc = np.fromfile(ptc_path, np.int16).reshape([-1, 4]) # ptc in (cm) metric.
    ptc = ptc[:, :3]
    
    # Remove zero points
    nz_mask = np.all((ptc != np.array([[0, 0, 0]], dtype=np.int16)), axis=1)
    ptc = ptc[nz_mask].astype(np.float32)
    
    # Calculate affine transform
    ins_path = os.path.join(data_path, 'raw_data', 'v{}'.format(version), code, 'INS', 'data', '{:010d}.txt'.format(frame))
    ins = np.loadtxt(ins_path)
    roll, pitch = ins[3:5]
    vf_to_hf = TR(0, 0, 0, roll, pitch, 0)
    
    if hparams.lidar_type == "raw":
      center = hparams.raw_center
      lon_range = hparams.raw_lon_range
      lon_length = lon_range[1] + lon_range[0]
      lon_diff = lon_range[1] - lon_range[0]

      lat_range = hparams.raw_lat_range
      lat_length = lat_range[1] + lat_range[0]
      lat_diff = lat_range[1] - lat_range[0]

      height_range = hparams.raw_height_range
      height_length = height_range[1] + height_range[0]
      height_diff = height_range[1] - height_range[0]
      
      num_point = hparams.num_point
      
      if version == 1:
        transform = vf_to_hf.dot(hdl_to_vfs.v1)
      elif version == 2:
        transform = vf_to_hf.dot(hdl_to_vfs.v2)
      else:
        raise ValueError("Unknown version. Version: {}".format(version))

      rotation = transform[:3, :3].T
      translation = transform[:3, 3]
      if center == "target": translation -= src_ref * 100.0 # conversion to cm

      # convert ptc
      ptc_hf = ptc.dot(rotation) + translation

      # Filter points out of boundary
      boundary = np.array([[lon_length/2, lat_length/2, height_length/2]], dtype=np.float32) * 100.0 # conversion to cm
      ptc_translated = ptc_hf - np.array([[lon_diff/2, lat_diff/2, height_diff/2]], dtype=np.float32) * 100.0 # conversion to cm
      mask = np.all(np.less(np.abs(ptc_translated), boundary), axis=1)
      ptc_filtered = ptc_hf[mask].astype(np.int16)
      # random shuffle points
      # np.random.shuffle(ptc_filtered)

      # Adjust sample to the regular shape [num_point, 3]
      points = ptc_filtered.shape[0]
      if points < num_point:
        # zero padding
        adjusted_ptc = np.zeros([num_point, 3], np.int16)
        adjusted_ptc[:points, :] = ptc_filtered
      else:
        # sampling
        random_idx = np.random.choice(points, size=num_point, replace=False)
        adjusted_ptc = ptc_filtered[random_idx]
        #adjusted_ptc = ptc_filtered[:num_point, :]
      
      lidar = adjusted_ptc
  
    # output as bev
    elif hparams.lidar_type == "bev":
      center = hparams.bev_center

      lon_range = hparams.bev_lon_range
      lon_length = lon_range[1] + lon_range[0]
      lon_diff = lon_range[1] - lon_range[0]

      lat_range = hparams.bev_lat_range
      lat_length = lat_range[1] + lat_range[0]
      lat_diff = lat_range[1] - lat_range[0]

      height_range = hparams.bev_height_range
      height_length = height_range[1] + height_range[0]
      height_diff = height_range[1] - height_range[0]

      bev_res = hparams.bev_res

      if version == 1:
        transform = vf_to_hf.dot(hdl_to_vfs.v1)
      elif version == 2:
        transform = vf_to_hf.dot(hdl_to_vfs.v2)
      else:
        raise ValueError("Unknown version. Version: {}".format(version))

      rotation = transform[:3, :3].T
      translation = transform[:3, 3]
      if center == "target": translation -= src_ref * 100.0 # conversion to cm

      # convert ptc
      ptc_hf = ptc.dot(rotation) + translation

      # Filter points out of boundary
      boundary = np.array([[lon_length/2, lat_length/2, height_length/2]], dtype=np.float32) * 100.0 # conversion to cm
      ptc_translated = ptc_hf - np.array([[lon_diff/2, lat_diff/2, height_diff/2]], dtype=np.float32) * 100.0 # conversion to cm
      mask = np.all(np.less(np.abs(ptc_translated), boundary), axis=1)
      ptc_filtered = ptc_hf[mask]

      # Make bird's eye view map
      ptc_translated = ptc_filtered + np.array([[lon_range[0], lat_range[0], height_range[0]]], dtype=np.float32) * 100.0 # conversion to cm
      XYZ_coord = (ptc_translated / bev_res * 100.0).astype(np.int32)
      map_size = np.asarray([lon_length / bev_res, lat_length / bev_res, height_length / bev_res], dtype=np.int32)

      bev_map = -np.ones(map_size, dtype=np.int8)
      bev_map[map_size[0] - 1 - XYZ_coord[0], XYZ_coord[1], XYZ_coord[2]] = 1

      lidar = bev_map
    else:
      raise ValueError("Unknown lidar type {}".format(hparams.lidar_type))

  if lidar is None:
    output_tuple = (src, tgt)
  else:
    output_tuple = (src, lidar, tgt)

  return output_tuple

def serialize_to_lmdb(dataset,
                      hparams,
                      lmdb_path):
  if os.path.isfile(lmdb_path):
    print("lmdb file ({}) exists!".format(lmdb_path))
  else:
    df = DataFromList(dataset, shuffle=False)
    df = MapData(df, lambda data: map_func(data, hparams))
    print("Creating lmdb cache...")
    LMDBSerializer.save(df, lmdb_path)


def get_iterator(dataset,
                 hparams,
                 lmdb_path,
                 shuffle=True,
                 drop_remainder=True,
                 nr_proc=4):
  
  serialize_to_lmdb(dataset, hparams, lmdb_path)

  batch_size = hparams.batch_size
  num_gpu = hparams.num_gpu
  df = LMDBSerializer.load(lmdb_path, shuffle=shuffle)

  batched_df = BatchData(df, batch_size=batch_size, remainder=not drop_remainder)
  splitted_df = MapData(batched_df, lambda x: [np.array_split(x[idx], num_gpu) for idx in range(len(x))])
  prefetched_df = PrefetchDataZMQ(splitted_df, nr_proc=nr_proc, hwm=batch_size*10)

  return prefetched_df

def get_infer_iterator(dataset,
                       hparams,
                       lmdb_path):
  
  serialize_to_lmdb(dataset, hparams, lmdb_path)

  batch_size = hparams.infer_batch_size
  num_gpu = hparams.num_gpu

  df = LMDBSerializer.load(lmdb_path, shuffle=False)

  batched_df = BatchData(df, batch_size=batch_size, remainder=False)
  splitted_df = MapData(batched_df, lambda x: [np.array_split(x[idx], num_gpu) for idx in range(len(x))])
  prefetched_df = PrefetchDataZMQ(splitted_df, nr_proc=1, hwm=batch_size*10)
  
  return prefetched_df
