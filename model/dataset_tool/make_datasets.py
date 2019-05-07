import tensorflow as tf
import numpy as np

import random
import os
import _pickle as pkl
from tqdm import tqdm

import copy

import pdb

import transform_utils
import visualize_utils

def read_trajectory(trajectory_files, shuffle=False):
  dataset = []
  for traj_file in trajectory_files:
    file_path = os.path.join(traj_path, traj_file)
    with open(file_path, 'rb') as reader:
      samples = pkl.load(reader)
    dataset += samples
  
  if shuffle:
    np.random.shuffle(dataset)

  return dataset

def extract_dataset(dataset):
  keys = ['source', 'target', 'dataset_version',
          'sample_code', 'reference_time', 'sequence_id']
  output_tuple = ([], [], [], [], [], [])
  for sample in dataset:  
    for idx, key in enumerate(keys):
      output_tuple[idx].append(sample[key])
  
  return output_tuple

random_seed = 1211
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_random_seed(random_seed)

data_path = os.path.abspath('../../dataset')
traj_path = os.path.join(data_path, 'trajectory', 'non_filled', 'absolute_50')
output_path = os.path.join(data_path, 'processed', 'pickle', 'non_filled', 'absolute_50')

traj_files = os.listdir(traj_path)
np.random.shuffle(traj_files)

v1_traj = list(filter(lambda x: "v1" in x, traj_files))
v2_traj = list(filter(lambda x: "v2" in x, traj_files))

num_v1 = len(v1_traj)
num_v2 = len(v2_traj)

test_files = v1_traj[:num_v1//5] + v2_traj[:num_v2//5]
devtrain_files = v1_traj[num_v1//5:] + v2_traj[num_v2//5:]
np.random.shuffle(devtrain_files)

num_devtrain = len(devtrain_files)
dev_files = devtrain_files[:num_devtrain//5]
train_files = devtrain_files[num_devtrain//5:]

traj_files.sort()
total_set = read_trajectory(traj_files)
test_set = read_trajectory(test_files)
dev_set = read_trajectory(dev_files)
train_set = read_trajectory(train_files, shuffle=True)
devtrain_set = read_trajectory(devtrain_files, shuffle=True)

total_output = os.path.join(output_path, 'total.pkl')
with open(total_output, 'wb') as writer:
  pkl.dump(total_set, writer)

trainset_output = os.path.join(output_path, 'train.pkl')
with open(trainset_output, 'wb') as writer:
  pkl.dump(train_set, writer)

devset_output = os.path.join(output_path, 'dev.pkl')
with open(devset_output, 'wb') as writer:
  pkl.dump(dev_set, writer)

testset_output = os.path.join(output_path, 'test.pkl')
with open(testset_output, 'wb') as writer:
  pkl.dump(test_set, writer)

devtrain_output = os.path.join(output_path, 'devtrain.pkl')
with open(devtrain_output, 'wb') as writer:
  pkl.dump(devtrain_set, writer)

# def _int64_feature(value):
#     """Wrapper for inserting int64 features into Example proto."""
#     if not isinstance(value, list):
#         value = [value]
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# def _float_feature(value):
#     """Wrapper for inserting float features into Example proto."""
#     if not isinstance(value, list):
#         value = [value]
#     return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# def _bytes_feature(value):
#     """Wrapper for inserting bytes features into Example proto."""
#     if not isinstance(value, list):
#         value = [value]
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

# TR = transform_utils.TR
# bird_eye_view = visualize_utils.bird_eye_view

# hdl_to_vf_list = {1: TR(193.42, 0, -175, 0.0286, 180.50, -1.34, scale_factor=0.01, degrees=True),
#                   2: TR(203, -2.5, -150, 179.4, 0.15, -91.3, scale_factor=0.01, degrees=True)}

# record_name = os.path.join(output_path, 'tfrecord', 'dev.tfrecord')
# options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
# writer = tf.python_io.TFRecordWriter(path=record_name, options=options)

# for sample in tqdm(dev_set):
#   v = 'v{}'.format(sample['dataset_version'])
#   sample_code = sample['sample_code']
#   frame = sample['reference_time']

#   source = sample['source']
#   target = sample['target']

#   sample_path = os.path.join(raw_path, v, sample_code)
#   hdl_frame_path = os.path.join(sample_path,
#                                 'Lidar',
#                                 'HDL32',
#                                 'data',
#                                 '{:010d}.bin'.format(frame))
#   ins_frame_path = os.path.join(sample_path,
#                                 'INS',
#                                 'data',
#                                 '{:010d}.txt'.format(frame))
  
#   ins_frame = np.loadtxt(ins_frame_path)
#   roll, pitch = ins_frame[3:5]
  
#   hdl_to_vf = hdl_to_vf_list[sample['dataset_version']]
#   vf_to_hf = TR(0, 0, 0, roll, pitch, 0)

#   ptc = np.fromfile(hdl_frame_path, dtype=np.int16).reshape(-1, 4)
#   ptc = ptc.astype(np.float32) # Data type conversion
#   ptc[:, :3] /= 100.0 # Convert to meter
#   ptc[:, 3] = 1.0 # Convenience for rigid body transforms
#   ptc = ptc.T # Transpose

#   ptc_hf = vf_to_hf.dot(hdl_to_vf.dot(ptc))
#   ptc_hf = ptc_hf.T
#   bev = bird_eye_view(ptc_hf, Forward=60, Left=60, Height=4, res=0.25)
  
#   record_dict = {
#       "bev_data": _bytes_feature(bev.tobytes()),
#       "source": _bytes_feature(source.tobytes()),
#       "target": _bytes_feature(target.tobytes()),
#       "dataset_version": _int64_feature(sample['dataset_version']),
#       "sample_code": _bytes_feature(sample_code.encode("utf-8")),
#       "reference_frame": _int64_feature(frame),
#       "sequence_id": _int64_feature(sample['sequence_id'])}
#   tf_example = tf.train.Example(features=tf.train.Features(feature=record_dict))
#   writer.write(tf_example.SerializeToString())

# writer.close()

# record_name = os.path.join(output_path, 'tfrecord', 'test.tfrecord')
# options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
# writer = tf.python_io.TFRecordWriter(path=record_name, options=options)

# for sample in tqdm(test_set):
#   v = 'v{}'.format(sample['dataset_version'])
#   sample_code = sample['sample_code']
#   frame = sample['reference_time']

#   source = sample['source']
#   target = sample['target']

#   sample_path = os.path.join(raw_path, v, sample_code)
#   hdl_frame_path = os.path.join(sample_path,
#                                 'Lidar',
#                                 'HDL32',
#                                 'data',
#                                 '{:010d}.bin'.format(frame))
#   ins_frame_path = os.path.join(sample_path,
#                                 'INS',
#                                 'data',
#                                 '{:010d}.txt'.format(frame))
  
#   ins_frame = np.loadtxt(ins_frame_path)
#   roll, pitch = ins_frame[3:5]
  
#   hdl_to_vf = hdl_to_vf_list[sample['dataset_version']]
#   vf_to_hf = TR(0, 0, 0, roll, pitch, 0)

#   ptc = np.fromfile(hdl_frame_path, dtype=np.int16).reshape(-1, 4)
#   ptc = ptc.astype(np.float32) # Data type conversion
#   ptc[:, :3] /= 100.0 # Convert to meter
#   ptc[:, 3] = 1.0 # Convenience for rigid body transforms
#   ptc = ptc.T # Transpose

#   ptc_hf = vf_to_hf.dot(hdl_to_vf.dot(ptc))
#   ptc_hf = ptc_hf.T
#   bev = bird_eye_view(ptc_hf, Forward=60, Left=60, Height=4, res=0.25)
  
#   record_dict = {
#       "bev_data": _bytes_feature(bev.tobytes()),
#       "source": _bytes_feature(source.tobytes()),
#       "target": _bytes_feature(target.tobytes()),
#       "dataset_version": _int64_feature(sample['dataset_version']),
#       "sample_code": _bytes_feature(sample_code.encode("utf-8")),
#       "reference_frame": _int64_feature(frame),
#       "sequence_id": _int64_feature(sample['sequence_id'])}
#   tf_example = tf.train.Example(features=tf.train.Features(feature=record_dict))
#   writer.write(tf_example.SerializeToString())

# writer.close()

# record_name = os.path.join(data_path, 'tfrecord', 'train.tfrecord')
# options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
# writer = tf.python_io.TFRecordWriter(path=record_name, options=options)

# for sample in tqdm(train_set):
#   v = 'v{}'.format(sample['dataset_version'])
#   sample_code = sample['sample_code']
#   frame = sample['reference_time']

#   source = sample['source']
#   target = sample['target']

#   sample_path = os.path.join(raw_path, v, sample_code)
#   hdl_frame_path = os.path.join(sample_path,
#                                 'Lidar',
#                                 'HDL32',
#                                 'data',
#                                 '{:010d}.bin'.format(frame))
#   ins_frame_path = os.path.join(sample_path,
#                                 'INS',
#                                 'data',
#                                 '{:010d}.txt'.format(frame))
  
#   ins_frame = np.loadtxt(ins_frame_path)
#   roll, pitch = ins_frame[3:5]
  
#   hdl_to_vf = hdl_to_vf_list[sample['dataset_version']]
#   vf_to_hf = TR(0, 0, 0, roll, pitch, 0)

#   ptc = np.fromfile(hdl_frame_path, dtype=np.int16).reshape(-1, 4)
#   ptc = ptc.astype(np.float32) # Data type conversion
#   ptc[:, :3] /= 100.0 # Convert to meter
#   ptc[:, 3] = 1.0 # Convenience for rigid body transforms
#   ptc = ptc.T # Transpose

#   ptc_hf = vf_to_hf.dot(hdl_to_vf.dot(ptc))
#   ptc_hf = ptc_hf.T
#   bev = bird_eye_view(ptc_hf, Forward=60, Left=60, Height=4, res=0.25)
  
#   record_dict = {
#       "bev_data": _bytes_feature(bev.tobytes()),
#       "source": _bytes_feature(source.tobytes()),
#       "target": _bytes_feature(target.tobytes()),
#       "dataset_version": _int64_feature(sample['dataset_version']),
#       "sample_code": _bytes_feature(sample_code.encode("utf-8")),
#       "reference_frame": _int64_feature(frame),
#       "sequence_id": _int64_feature(sample['sequence_id'])}
#   tf_example = tf.train.Example(features=tf.train.Features(feature=record_dict))
#   writer.write(tf_example.SerializeToString())

# writer.close()

