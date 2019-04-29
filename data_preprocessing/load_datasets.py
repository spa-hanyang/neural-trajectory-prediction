from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

path = os.path.abspath('./')
raw_data_path = os.path.join(path, 'dataset', 'raw_data')


class load_datasets(object):
  """Implementation of making datasets."""
  
  def __init__(self, version, dirname):
    """
    Args:
        version: version of raw data
          ex) 'v1', 'v2'
        dirname: directory name of tracklets  
          ex) '2017Y03M08D11H11m07s'
    """
 
    self.VER = version
    self.DIR = dirname
    self.TR_PATH = os.path.join(raw_data_path, self.VER, self.DIR, 
                                 'tracklets', 'data')
    self.INS_PATH = os.path.join(raw_data_path, self.VER, self.DIR, 
                                 'INS', 'data')
  

  def load_tracklets(self):
    """load tracklets data."""

    load_path = [os.path.join(self.TR_PATH, dataset) for dataset in os.listdir(self.TR_PATH)]
    load_path.sort()
    
    datasets = []
    for num in range(len(load_path)):
      dataset = np.loadtxt(load_path[num], dtype=np.float64)
      datasets.append(dataset)
    print("start load {} file".format(self.DIR))
    
    return datasets

  
  def load_INS(self):
    """load INS data."""

    load_path = [os.path.join(self.INS_PATH, dataset) for dataset in os.listdir(self.INS_PATH)]
    load_path.sort()
    
    datasets = []
    for num in range(len(load_path)):
      dataset = np.loadtxt(load_path[num], dtype=np.float64)
      datasets.append(dataset)

    return datasets

  
  
