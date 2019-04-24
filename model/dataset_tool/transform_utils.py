from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = ["WGS84toTM", "TR"]

def WGS84toTM(latitude, longitude):
  # WGS84 to TM parameters
  m_dA0 = 6378137.0

  m_dEp = 0.00673949674228 # 0.00673949674226
  m_dE2 = 0.00669437999014 # 0.00669437999013
  m_dE4 = m_dE2 ** 2
  m_dE6 = m_dE2 * m_dE4

  m_dM0 = 1 - m_dE2/4 - 3*m_dE4/64 - 5*m_dE6/256
  m_dM1 = 3*m_dE2/8 + 3*m_dE4/32 + 45*m_dE6/1024
  m_dM2 = 15*m_dE4/256 + 45*m_dE6/1024
  m_dM3 = 35*m_dE6/3072

  m_dX0 = 600000
  m_dY0 = 200000

  B3 = np.radians(latitude) #
  B3_0 = np.radians(38) #

  dTanT = np.tan(B3) #
  dCosT = np.cos(B3) #
  dSinT = np.sin(B3) #

  T = dTanT ** 2 #
  C = m_dE2 / (1-m_dE2) * dCosT ** 2 #
  A = np.radians(longitude - 127.0) * dCosT #
  N = m_dA0 / np.sqrt(1 - m_dE2 * dSinT ** 2) #

  M = m_dA0 * (m_dM0 * B3 - m_dM1 * np.sin(2*B3) + m_dM2 * np.sin(4*B3) - m_dM3 * np.sin(6*B3))
  M0 = m_dA0 * (m_dM0 * B3_0 - m_dM1 * np.sin(2*B3_0) + m_dM2 * np.sin(4*B3_0) - m_dM3 * np.sin(6*B3_0))
  
  dY1 = A ** 3 / 6 * (1 - T + C) #
  dY2 = A ** 5 / 120 * (5 - 18 * T + T ** 2 + 72 * C - 58 * m_dEp)

  dX1 = A ** 2 / 2
  dX2 = A ** 4 / 24 * (5 - T + 9 * C + 4 * C ** 2)
  dX3 = A ** 6 / 720 * (61 - 58 * T + T ** 2 + 600 * C - 330 * m_dEp)

  Y = m_dY0 + N * ( A + dY1 + dY2)
  X = m_dX0 + (M - M0 + N * dTanT *(dX1 + dX2 + dX3))

  return X, Y

def TR(tx, ty, tz, rx, ry, rz, scale_factor=1.0, degrees=False):
  R = _rotation_mat(rx, ry, rz, degrees=degrees)
  T = _translation_mat(tx, ty, tz, scale_factor=scale_factor)

  return np.block([[R, T], [0, 0, 0, 1]])

def _rotation_mat(rx, ry, rz, degrees=False):
  if degrees:
    [rx, ry, rz] = np.radians([rx, ry, rz])

  c, s = np.cos(rx), np.sin(rx)                                                                                                            
  rx_mat = np.array([1, 0, 0, 0, c, -s, 0, s, c]).reshape(3, 3)                                                                            
  c, s = np.cos(ry), np.sin(ry)                                                                                                            
  ry_mat = np.array([c, 0, s, 0, 1, 0, -s, 0, c]).reshape(3, 3)                                                                            
  c, s = np.cos(rz), np.sin(rz)                                                                                                            
  rz_mat = np.array([c, -s, 0, s, c, 0, 0, 0, 1]).reshape(3, 3)                                                                            
  return rz_mat.dot(ry_mat.dot(rx_mat))

def _translation_mat(tx, ty, tz, scale_factor=1.0):
  return np.array([[tx], [ty], [tz]]) * scale_factor
