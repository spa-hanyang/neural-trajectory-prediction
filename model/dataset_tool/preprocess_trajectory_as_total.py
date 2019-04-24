from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import collections
import pickle as pkl

from transform_utils import *
from visualize_utils import *

import pdb

ABSOLUTE = False

# Set data processing parameters
OBSVLEN = 30 # Observation Length
PREDLEN = 20 # Prediction label length
MINLEN = OBSVLEN + PREDLEN 

# Set datapath constants
DATA_PATH = os.path.abspath("../../dataset")
RAW_PATH = os.path.join(DATA_PATH, 'raw_data')
PROCESSED_PATH = os.path.join(DATA_PATH, 'trajectory')

# Set random seed
RANDOM_SEED = 114
np.random.seed(RANDOM_SEED)

def extract_sequence(tracklet_list, seq_id):
  len_tracklet = len(tracklet_list)
  
  # Vehicle XYZ coordinate and orientation.
  XYZO = []

  for t_idx in range(len_tracklet):
    temp_tracklet = tracklet_list[t_idx]
    if len(temp_tracklet.shape) == 1:
      # Expand dimension if 1D array (only one vehicle in the tracklet).
      temp_tracklet = np.expand_dims(temp_tracklet, axis=0)
    num_vehicles = temp_tracklet.shape[0]

    for v_idx in range(num_vehicles):
      if seq_id == temp_tracklet[v_idx, 0]:
        XYZO.append(temp_tracklet[v_idx, [2, 3, 4, 8]])
        break
      else:
        continue

  # Stack the list as 4 by N array.
  XYZO = np.stack(XYZO, axis=1)
  return XYZO


def valid_sequence(tracklet_list):
  """
  Filter the track_list to have only the valid tracklet sequences.
  Currently the list is filtered such that only car class objects are contained
  and the object ID and sequence number are continuous for whole MINLEN frame.
  """
  first_frame = tracklet_list[0]
  if len(first_frame.shape) == 1:
    # Expand dimension if 1D array is given (signle vehicle in the list).
    first_frame = np.expand_dims(first_frame, axis=0)

  num_vehicles = first_frame.shape[0]
  seq_numbers = first_frame[:, 0]
  obj_ids = first_frame[:, 1]
  obj_classes = first_frame[:, 9]
  
  # Indices of the valid vehicles.
  valid_vehicles_idx = list(range(num_vehicles))

  # Filter out non-vehicles
  for v_idx in valid_vehicles_idx.copy():
    if obj_classes[v_idx] < 20:
      valid_vehicles_idx.remove(v_idx)
  
  for t_idx in range(1, len(tracklet_list)):
    temp_tracklet = tracklet_list[t_idx]
    if len(temp_tracklet.shape) == 1:
      # Expand dimension if 1D array (only one vehicle in the tracklet).
      temp_tracklet = np.expand_dims(temp_tracklet, axis=0)
    temp_num_vehicles = temp_tracklet.shape[0]
    
    for v_idx in valid_vehicles_idx.copy():
      matched = False

      seq_number = seq_numbers[v_idx]
      obj_id = obj_ids[v_idx]

      for temp_v_idx in range(temp_num_vehicles):
        matched = False
        seq_number_temp = temp_tracklet[temp_v_idx, 0]
        seq_number_matched = (seq_number == seq_number_temp)
        obj_id_temp = temp_tracklet[temp_v_idx, 1]
        obj_id_matched = (obj_id == obj_id_temp)

        matched = seq_number_matched and obj_id_matched
        if matched:
          break

      if not matched:
        valid_vehicles_idx.remove(v_idx)

  # After the iterations, indices of the valid vehicles last in valid_vehicles_idx.
  # Gather corresponding sequence ids of the indices and return.
  return [seq_numbers[idx] for idx in valid_vehicles_idx]

# Samples path list
sample_paths = []
for v in ['v1', 'v2']:
  version_path = os.path.join(RAW_PATH, v)
  sample_paths += [os.path.join(version_path, s)
      for s in os.listdir(version_path)]
np.random.shuffle(sample_paths)

len_sample_code = len('0000Y11M22D33H44m55s') # sample code length
hdl_to_vf_mats = {
    'v1': TR(193.42, 0, -175, 0.0286, 180.50, -1.34, scale_factor=0.01, degrees=True),
    'v2': TR(203, -2.5, -150, 179.4, 0.15, -91.3, scale_factor=0.01, degrees=True)}

for s_idx, s_path in enumerate(sample_paths):
  if "2018Y02M20D09H59m22s" in s_path
    continue
     
  # Set parameters depend on the sample version.
  if 'v1' in s_path:
    sample_version = 'v1'
  else:
    sample_version = 'v2'
    
  hdl_to_vf = hdl_to_vf_mats[sample_version]
  sample_code = s_path[-len_sample_code:]
  
  print("Processing {}_{}".format(sample_version, sample_code))
  track_path = os.path.join(s_path, 'tracklets', 'data')
  track_frames = [os.path.join(track_path, frame) for frame in os.listdir(track_path)]
  track_frames.sort()

  ins_path = os.path.join(s_path, 'INS', 'data')
  ins_frames = [os.path.join(ins_path, frame) for frame in os.listdir(ins_path)]
  ins_frames.sort()

  num_frames = len(track_frames)
  
  pickle_list = []
  for base in tqdm(range(num_frames - MINLEN + 1)):
    reference_frame = base + OBSVLEN - 1

    tracklets = []
    ins = []
    
    empty_flag = False
    for offset in range(MINLEN):
      """
      Quickly detect frame with empty tracklet
      within the processing range [base, base+MINLEN).
      """
      track_file = track_frames[base + offset]
      with open(track_file, 'r') as reader:
        if len(reader.readline()) == 0:
          empty_flag = True
          break
    if empty_flag:
      continue # continue to next iteration if there is any empty tracklet.

    for offset in range(MINLEN):
      """
      Read tracklets and INS in processing range [base, base+MINLEN)
      and save them into the lists in order.
      """
      tmp_track = track_frames[base + offset]
      tracklets.append(np.loadtxt(tmp_track))
      
      tmp_ins = ins_frames[base + offset]
      ins.append(np.loadtxt(tmp_ins))
    
    # Check if the lengths are legit.
    assert len(tracklets) == MINLEN
    assert len(ins) == MINLEN
    
    # Filter the valid tracklets.
    valid_seq_id = valid_sequence(tracklets)
    if len(valid_seq_id) == 0:
      continue
    
    vf_to_tm = []
    steer = []
    wheelspd = []

    for offset in range(MINLEN):
      ins_temp = ins[offset]
      # Calculate rigid body transforms
      lat, lon, alt, roll, pitch, yaw = ins_temp[:6]
      N, E = WGS84toTM(lat, lon)
      vf_to_tm.append(TR(N, E, -alt, roll, pitch, yaw))
      
      # Transforms at the end of OBSVLEN
      if offset == OBSVLEN - 1:
        ref_vf_to_tm = vf_to_tm[offset]
        ref_vf_to_hf = TR(0, 0, 0, roll, pitch, 0)

      # Steer angle and wheel spped
      steer.append(ins_temp[9])
      wheelspd.append(np.mean(ins_temp[12:]))
    
    ego_info = np.array([steer, wheelspd], np.float32)

    sequences = []

    if not ABSOLUTE:
      for seq_id in valid_seq_id:
        XYZO = extract_sequence(tracklets, seq_id)
        XYZ = XYZO[:3, :]
        XYZ /= 100.0
        XYZ = np.concatenate((XYZ, ego_info), axis=0)
        sequences.append(XYZ.T)

    else:
      for seq_id in valid_seq_id:
        XYZO = extract_sequence(tracklets, seq_id)
        XYZ = XYZO.copy()
        XYZ[:3, :] /= 100.0 # Convert to meters
        XYZ[3, :] = 1.0 # Convenience for rigid body transforms
        
        # Transform each frame's coordinates into the horizontal frame of the reference time.
        for offset in range(MINLEN):
          temp_XYZ = XYZ[:, offset]
          temp_XYZ_tm = vf_to_tm[offset].dot(hdl_to_vf.dot(temp_XYZ))
          temp_XYZ_ref_vf = np.linalg.solve(ref_vf_to_tm, temp_XYZ_tm)
          XYZ[:, offset] = ref_vf_to_hf.dot(temp_XYZ_ref_vf)
        
        XYZ = XYZ[:3, :]
        # Replace the coordinates
        sequences.append(XYZ.T)

    for sequence, seq_id in zip(sequences, valid_seq_id):
      source_sequence = sequence[:OBSVLEN].astype(np.float32)
      target_sequence = sequence[OBSVLEN:].astype(np.float32)
      pickle_dict = {"source": source_sequence,
                     "target": target_sequence,
                     "dataset_version": int(sample_version[-1]),
                     "sample_code": sample_code,
                     "reference_time": int(reference_frame),
                     "sequence_id": int(seq_id)}
      pickle_list.append(pickle_dict)
    
  with open(os.path.join(PROCESSED_PATH, ('absolute' if ABSOLUTE else 'relative'), '{}.{}'.format(sample_version, sample_code)), 'wb') as file:
    pkl.dump(pickle_list, file)