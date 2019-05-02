from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle as pkl
import scipy.io as sio

import pdb

path = os.path.abspath('./')
dataset_path = 'dataset_agument'
dir_path = os.path.join(path, dataset_path)


class save_datasets(object):

  def __init__(self, save_path, version, filename):

    # self.PATH = os.path.abspath('./')  
    self.PATH = save_path
    self.VER = version
    self.FILE = filename
    self.SAVE_VER = os.path.join(self.PATH, self.VER)


  def save_as_pkl(self, dataset):
    
    with open(os.path.join(self.SAVE_VER, '{}.{}'.format(self.VER, self.FILE)), 'wb') as file:
      pkl.dump(dataset, file)
  
  
  def save_as_mat(self, dataset):  
     
    dataset = rebuild_key(dataset)
    sio.savemat(os.path.join(self.SAVE_VER, '{}_{}'.format(self.VER, self.FILE)), mdict=dataset)


def make_dir(making_path, dirname):
  try:
    if os.path.isdir(os.path.join(making_path, dirname)):
      pass
    else:
      os.makedirs(os.path.join(making_path, dirname))
  except OSError:
    print("Creation of the directory {} failed".format(os.path.join(making_path, dirname)))
  else:
    print("Successfully created the directory {}".format(os.path.join(making_path, dirname)))


# rebuild dectionary keys for mat file
def rebuild_key(dictionary):
  '''rebuild the dictionary's keys to make matlab file   '''
  save_key = 0
  for key in list(dictionary.keys()):
    if save_key == 0:    
      dictionary['key_' + str(int(key))] = dictionary.pop(key)
      save_key = int(key)
      numcount = 1
    elif int(key) - save_key >= 1:
      dictionary['key_' + str(int(key))] = dictionary.pop(key)
      save_key = int(key)
      numcount = 1
    else:
      dictionary['key_' + str(int(key)) + '_' + str(numcount)] = dictionary.pop(key)
      numcount += 1
    
  return dictionary
