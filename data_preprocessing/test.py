from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import scipy.io as sio
import pdb
import pickle as pkl

from load_datasets import load_datasets
from save_datasets import *
from transform_utils import *

path = os.path.abspath('./')
dataset_path = 'dataset_agument'
dir_path = os.path.join(path, dataset_path)


with open(os.path.join(dir_path, 'pickle_data', 'v1', 'v1_2017Y03M17D15H51m25s'), 'rb') as file:
    test = pkl.load(file)

pdb.set_trace()
