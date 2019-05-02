from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import scipy.io as sio
import pdb

from load_datasets import load_datasets
from save_datasets import *
from transform_utils import *

def classify_tracklets(datasets):
  """
  divide the datasets into odjectID to make dictionary 
  key = object ID
  value = data of the object 
  """
  full_datasets = {}
  for time, dataset in enumerate(datasets):
    data_arr = dataset
    
    if data_arr.shape == (0,):
      continue
    data_arr = append_time(data_arr, time)

    num_vehicles = data_arr.shape[0]
    for i in range(num_vehicles):
      # filtering not vehicle
      if data_arr[i, 9] < 20:
        continue       
      elif data_arr[i, 1] in full_datasets:
        full_datasets[int(data_arr[i, 1])].append(data_arr[i])
      else:
        full_datasets[int(data_arr[i, 1])] = []
        full_datasets[int(data_arr[i, 1])].append(data_arr[i])
    
  for key in list(full_datasets.keys()):
    full_datasets[key] = np.vstack(full_datasets[key])

  return full_datasets     

def append_time(data, time):
    """append tracklets time to last element of data."""
    # the number of tracklets
    if len(data.shape) == 1:
      # Expand dimension if 1D array is given (signle vehicle in the list)
      data = np.expand_dims(data, axis=0)
    
    num_vehicles = data.shape[0]
    number_arr = time * np.ones(num_vehicles)
    data = np.c_[data, number_arr]
    
    return data


def len_filter(datasets, M=60):
  # M is minimum length of data
  
  for obj_ID in list(datasets.keys()):
    if datasets[obj_ID][-1, 10] - datasets[obj_ID][0, 10] < M:
      del datasets[obj_ID]
      # print("delete {:d} object.".format(obj_ID))  
  
  return datasets


def detect_emptynum(datasets):
  '''detect the index numbers that are not in tracklet data '''
  # save as directory
  emptynum = dict()
  for obj_ID in list(datasets.keys()):
    First_check = True
    checknum = 0
    for i in range(len(datasets[obj_ID]) - 1):
      interval = int(datasets[obj_ID][i+1,10] - datasets[obj_ID][i,10])
      if interval == 0 and not i == 0 :
        # velocity of before sequence
        velocity_1 = np.sqrt(np.sum(np.square(datasets[obj_ID][i,[2,3]] 
                              - datasets[obj_ID][i - 1,[2,3]])))
        velocity_2 = np.sqrt(np.sum(np.square(datasets[obj_ID][i + 1,[2,3]] 
                              - datasets[obj_ID][i - 1,[2,3]])))
        if velocity_1 > velocity_2 :
          np.delete(datasets[obj_ID], i, axis=0)
        else :
          np.delete(datasets[obj_ID], i + 1, axis=0)
      elif interval == 1 :
        continue
      elif interval == 2 :
        if First_check :
          emptynum[obj_ID] = []
          emptynum[obj_ID].append(datasets[obj_ID][i,10] + 1)
          # print("{:d} object sequence has empty number {:.1f}.".format\
          #       (obj_ID, datasets[obj_ID][i,10] + 1)) 
        else :           
          new_obj_ID = obj_ID + checknum 
          emptynum[new_obj_ID] = []
          emptynum[new_obj_ID].append(datasets[obj_ID][i,10] + 1)
          # print("{:.2f} object sequence has empty number {:.1f}.".format\
          #       (new_obj_ID, datasets[obj_ID][i,10] + 1)) 
        First_check = False
        checknum += 0.01
      else :
        if First_check :
          emptynum[obj_ID] = []                    
          for j in range(1, interval) :
            emptynum[obj_ID].append(datasets[obj_ID][i,10] + j)
          # print("{:d} object sequence has empty number {:.1f} to {:.1f}.".format\
          #      (obj_ID, datasets[obj_ID][i,10] + 1, datasets[obj_ID][i+1,10] - 1))   
        else : 
          new_obj_ID = obj_ID + checknum           
          emptynum[new_obj_ID] = []            
          for j in range(1, interval) :
            emptynum[new_obj_ID].append(datasets[new_obj_ID][i,10] + j)
          # print("{:.2f} object sequence has empty number {:.1f} to {:.1f}.".format\
          #        (new_obj_ID, datasets[obj_ID][i,10] + 1, datasets[obj_ID][i+1,10] - 1))                                     
        First_check = False
        checknum += 0.01

  return emptynum

def TRtoTM(tr_datasets, INS_datasets, version):
  """x,y,z of tracklet datas transform to TM datas."""
  # select x,y,z in tracklet datas
  XYZ = {}
  for obj_ID in list(tr_datasets.keys()):
    XYZ[obj_ID] = tr_datasets[obj_ID][:, [2, 3, 4]]
    XYZ[obj_ID] = XYZ[obj_ID] / 100.0 # Convert to meters
    XYZ[obj_ID] = np.c_[XYZ[obj_ID], np.ones(len(XYZ[obj_ID]))] # for rigid transform (X,Y,Z) -> (X,Y,Z,1)
  
  # transform matrix of hdl_to_vf in version
  hdl_to_vf_mats = {'v1': TR(193.42, 0, -175, 0.0286, 180.50, -1.34, scale_factor=0.01, degrees=True),
                    'v2': TR(203, -2.5, -150, 179.4, 0.15, -91.3, scale_factor=0.01, degrees=True)}

  if version == 'v1':
    hdl_to_vf = hdl_to_vf_mats['v1']
  elif version == 'v2':
    hdl_to_vf = hdl_to_vf_mats['v2']
  
  # transform matrix of vf_to_tm
  vf_to_tm = []  
  for seq in range(len(INS_datasets)):
    vf_to_tm.append(get_vf_to_tm(INS_datasets[seq]))
    # altitude has to be negative
  
  XYZ_tm = {}
  for obj_ID in list(tr_datasets.keys()):
    
    temp_XYZ_tm = [] 
    for offset in range(len(XYZ[obj_ID])):
      # matrix multiply of vf_to_tm and hdl_to_vf
      hdl_to_tm = vf_to_tm[int(tr_datasets[obj_ID][offset,10])].dot(hdl_to_vf)
      # multiply of matrix_hdl_to_tm and vector_(XYZ,1)
      temp_XYZ_tm.append(XYZ[obj_ID][offset].dot(hdl_to_tm.T))
  
    XYZ_tm[obj_ID] = np.vstack(temp_XYZ_tm)
    XYZ_tm[obj_ID] = XYZ_tm[obj_ID][:,:3]
  
  # data restruction
  tf_dataset = copy.deepcopy(tr_datasets)  
  for obj_ID in list(tf_dataset.keys()):
    tf_dataset[obj_ID][:,(2,3,4)] = XYZ_tm[obj_ID]
  
  return tf_dataset, vf_to_tm

def fill_emptynum(datasets, emptynum) :
  '''fill empty space with reconstructed data'''
  for obj_ID in list(emptynum.keys()) :
    if int(obj_ID) == obj_ID :
      checknum = 0.01
    
    # previous number of the first empty index
    index_num = np.where(datasets[int(obj_ID)][:,10] == emptynum[obj_ID][0] - 1)[0]
    
    # if length empty index is one, fill empty index using average value 
    if len(emptynum[obj_ID]) == 1 :
      stuff_arr = datasets[int(obj_ID)][index_num]
      np.put(stuff_arr, 10, emptynum[obj_ID][0])

      make_coordinate = (datasets[int(obj_ID)][index_num, 2:5] + datasets[int(obj_ID)][index_num + 1, 2:5]) / 2.0
      np.put(stuff_arr, [2, 3, 4], make_coordinate)
      
      make_yaw = (datasets[int(obj_ID)][index_num, 8] + datasets[int(obj_ID)][index_num + 1, 8]) / 2.0
      np.put(stuff_arr, 8, make_yaw)
      
      datasets[int(obj_ID)] = np.insert(datasets[int(obj_ID)], index_num + 1, stuff_arr, axis=0)
      # print("stuff {} object's empty number {}.".format(int(obj_ID), emptynum[obj_ID][0]))

    # if index number is zero or index number + 2 exceed datasets length, split array 
    elif index_num == 0 or index_num + 2 > len(datasets[int(obj_ID)]) - 1:
      datasets_split = np.vsplit(datasets[int(obj_ID)], [int(index_num + 1), ])
      
      if len(datasets_split[0]) < 50 :
        datasets[int(obj_ID)] = datasets_split[1]

      elif int(obj_ID) == obj_ID :
        obj_ID = int(obj_ID) + checknum
        datasets[obj_ID] = datasets_split[0]
        datasets[int(obj_ID)] = datasets_split[1]
        checknum += 0.01 
      else:
        obj_ID = int(obj_ID) + checknum
        datasets[obj_ID] = datasets_split[0]
        datasets[int(obj_ID)] = datasets_split[1]
        checknum += 0.01 
    else:
      # velocity of before sequence
      velocity_1 = np.sqrt(np.sum(np.square(datasets[int(obj_ID)][index_num,[2,3]] 
                            - datasets[int(obj_ID)][index_num - 1,[2,3]])))
      # velocity of later sequence
      velocity_2 = np.sqrt(np.sum(np.square(datasets[int(obj_ID)][index_num + 2,[2,3]] 
                            - datasets[int(obj_ID)][index_num + 1,[2,3]])))
      
      # if length of empty space is longer than 20, split array
      if len(emptynum[obj_ID]) > 20:
        datasets_split = np.vsplit(datasets[int(obj_ID)], [int(index_num + 1), ])
        
        if len(datasets_split[0]) < 50:
          datasets[int(obj_ID)] = datasets_split[1]
        
        elif int(obj_ID) == obj_ID:
          obj_ID = int(obj_ID) + checknum
          datasets[obj_ID] = datasets_split[0]
          datasets[int(obj_ID)] = datasets_split[1]
          checknum += 0.01 
        else:
          obj_ID = int(obj_ID) + checknum
          datasets[obj_ID] = datasets_split[0]
          datasets[int(obj_ID)] = datasets_split[1]
          checknum += 0.01

      # if substraction of velocities is smaller than 1m/s, fill empty space using a linear equation
      elif abs(velocity_2 - velocity_1) < 0.1:
        stuff_arr_x = np.linspace(datasets[int(obj_ID)][index_num, 2], datasets[int(obj_ID)][index_num + 1, 2], endpoint=False)
        stuff_arr_x = stuff_arr_x[1:]
        
        stuff_arr_y = np.linspace(datasets[int(obj_ID)][index_num, 2], datasets[int(obj_ID)][index_num + 1, 2], endpoint=False)
        stuff_arr_y = stuff_arr_y[1:]
        
        stuff_arr = []
        for offset in range(len(emptynum[obj_ID])):
          offset_arr = datasets[int(obj_ID)][index_num]
          np.put(offset_arr, [2, 3, 10], [stuff_arr_x[offset], stuff_arr_y[offset], emptynum[obj_ID][offset]] )
          stuff_arr.append(offset_arr)
        
        for offset in range(len(emptynum[obj_ID])):
          datasets[int(obj_ID)] = np.insert(datasets[int(obj_ID)], index_num + 1 + offset, stuff_arr[offset], axis=0)
        # print("stuff {} object's empty number {} to {}.".format(int(obj_ID), emptynum[obj_ID][0], emptynum[obj_ID][-1]))   
       
      else:
        datasets_split = np.vsplit(datasets[int(obj_ID)], [int(index_num + 1), ])
        
        if len(datasets_split[0]) < 50:
          datasets[int(obj_ID)] = datasets_split[1]

        elif int(obj_ID) == obj_ID:
          obj_ID = int(obj_ID) + checknum
          datasets[obj_ID] = datasets_split[0]
          datasets[int(obj_ID)] = datasets_split[1]
          checknum += 0.01 
        else:
          obj_ID = int(obj_ID) + checknum
          datasets[obj_ID] = datasets_split[0]
          datasets[int(obj_ID)] = datasets_split[1]
          checknum += 0.01
  
  # sort keys
  sorted_datasets = {}
  for key in sorted(datasets.keys()):
    sorted_datasets[key] = datasets[key]
  
  return sorted_datasets

def get_vf_to_tm(INS_dataset, degree=0.) :
  R_matrix = _rotation_mat(0, 0, degree, degrees=True) 
  lat, lon, alt, roll, pitch, yaw = INS_dataset[:6]
  N, E = WGS84toTM(lat, lon)
  seq_arr = np.array([N, E, alt])
  N, E, alt = seq_arr.dot(R_matrix)    
  return TR(N, E, -alt, roll, pitch, yaw)

def data_rotation(tr_datasets, INS_datasets, degree):
  R_matrix = _rotation_mat(0, 0, degrees, degrees=True) 
 
  transform_tr = copy.deepcopy(tr_datasets)  
  for obj_ID in list(tr_datasets.keys()):
    for seq in range(len(tr_datasets[obj_ID])):  
      XYZ = tr_datasets[obj_ID][seq,[2, 3, 4]]
      XYZ = XYZ.dot(R_matrix)
      
      O = tr_datasets[obj_ID][seq, 8] - degrees
      XYZO = np.append(XYZ,O)
      
      np.put(transform_tr[obj_ID][seq], [2, 3, 4, 8], XYZO)

  vf_to_tm = {}
  for seq in range(len(INS_datasets)):
    vf_to_tm[seq] = get_vf_to_tm(INS_datasets[seq], degree)

  return transform_tr, vf_to_tm
