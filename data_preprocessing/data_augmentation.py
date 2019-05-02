from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import argparse

import numpy as np
import scipy.io as sio
import pdb

from load_datasets import load_datasets
from save_datasets import *
from transform_utils import *
from augment_utils import *

def _list(s):
  v = ast.literal_eval(s)
  if type(v) is not list:
    raise argparse.ArgumentTypeError("Argument \"{}\" is not a list".format(s))
  return v

def add_arguments(parser):
  """ Build argument parser"""
  parser.register("type", "list", _list)

  parser.add_argument("--type", type=str, nargs='*', default="filling",
                      help="Type of data augmentation (filling,rotation)")
  parser.add_argument("--format", type=str, default="pkl",
                      choices=['pkl','mat'],
                      help="Save file format of augmented data")
  parser.add_argument("--length", type=int, default=60,
                      help="Minimum length of data")
  parser.add_argument("--length_margin", type=int, default=40,
                      help="Marginal length of data for filling")
  parser.add_argument("--degree", type=float, default=0.,
                      help="degree of rotation (0.0 : random)")
  parser.add_argument("--path", type=str, default='./',
                      help='Path of dataset')

def augmentation(config) :
  version = ['v1', 'v2']

  path = config.path
  path = os.path.join(path,'dataset')
  org_data_path = os.path.join(path, 'raw_data')
  make_dir(path, 'dataset_augument')
  data_path = os.path.join(path, 'dataset_augument')
  save_path = dict()
  for t in config.type :
    save_path[t] = os.path.join(data_path, t)
    make_dir(data_path, t)

  for ver in version:
    make_dir(data_path , ver)
    file_path = os.path.join(raw_data_path, ver)
    for file in os.listdir(file_path):
      if file == '2018Y02M20D09H59m22s':
        continue

      loads = load_datasets(ver, file)
      data_tracklet = loads.load_tracklets()
      data_INS = loads.load_INS()

      ## preprocessing
      fd = classify_tracklets(data_tracklet)
      xyz, vf_to_tm = TRtoTM(fd, data_INS, ver)
      fd = len_filter(xyz,config.length_margin)

      if 'filling' in config.type :
        emp = detect_emptynum(fd)
        ffd = fill_emptynum(fd, emp)
        fd = len_filter(ffd, config.length) 
        save = save_datasets(save_path['filling'], ver, file)
        save.save_as_pkl([fd,vf_to_tm])
        del save

      if 'rotation' in config.type :
        fd, vf_to_tm = data_rotation(fd, data_INS, degree)
        save = save_datasets(save_path['rotation'], ver, file)
        save.save_as_pkl([fd,vf_to_tm])
        del save

      save = save_datasets(data_path, ver, file)
      save.save_as_pkl(fd)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  config = parse_args(parser)
  augmentation(config)
