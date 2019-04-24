from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = ["bird_eye_view", "uint8_3ch"]

def bird_eye_view(pcloud, Forward=80, Left=80, Height=3, res=0.2, dtype=np.uint8):
  ''' Generate bird's eye view image from a single lidar frame.'''

  X_mask = np.abs(pcloud[:, 0]) < Forward
  Y_mask = np.abs(pcloud[:, 1]) < Left
  Z_mask = np.abs(pcloud[:, 2]) < Height

  mask = np.all([X_mask, Y_mask, Z_mask], axis=0)

  pcloud = pcloud[mask, :3]
  
  X_coord = ((pcloud[:,0] + Forward) / res).astype(np.int32)
  Y_coord = ((pcloud[:,1] + Left) / res).astype(np.int32)
  Z_coord = ((pcloud[:,2] + Height) / res).astype(np.int32)
  map_size = np.asarray([2 * Left / res, 2 * Forward / res, 2 * Height / res], dtype=np.int32)

  bev_map = np.zeros(map_size, dtype=dtype)
  bev_map[map_size[0] - 1 - X_coord, Y_coord, Z_coord] = 1

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