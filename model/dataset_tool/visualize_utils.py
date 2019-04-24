from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pdb

__all__ = ["bird_eye_view", "uint8_3ch"]

def bird_eye_view(pcloud, Forward=60, Left=60, Height=4, res=0.25, dtype=np.uint8):
  ''' Generate bird's eye view image from a single lidar frame.'''
  FLH_array = np.array([[Forward, Left, Height]])
  mask = np.all(np.less(np.abs(pcloud), FLH_array), axis=1)

  pcloud = pcloud[mask]
  
  YXZ = ((pcloud + FLH_array) / res).astype(np.int32)

  map_size = np.array([2 * Left / res, 2 * Forward / res, 2 * Height / res], dtype=np.int32)

  bev_map = np.zeros(map_size, dtype=dtype)
  bev_map[YXZ[:, 1], YXZ[:, 0], YXZ[:, 2]] = 1

  return bev_map

def uint8_3ch(bev_img):
  img_shape = list(bev_img.shape[:2])
  img_ch = bev_img.shape[2]
  
  output_shape = img_shape + [3]
  output_ch, R = np.divmod(img_ch, 3)
  output = np.zeros(output_shape)
  
  ch_ptr = 0
  for i in range(3):
    prev_ptr = ch_ptr
    ch_ptr += output_ch
    if i < R:
      ch_ptr += 1

    output[:, :, i] = np.sum(bev_img[:, :, prev_ptr:ch_ptr], axis=2) * 255 / (ch_ptr - prev_ptr)

  return output.astype(np.uint8)