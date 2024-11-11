"""
File: sattellite_antenna_model.py

Purpose:
TODO: This file defines...


Author: Ernesto Fontes Pupo / Claudia Carballo González
        University of Cagliari
Date: 2024-10-30
Version: 1.0.0
SPDX-License-Identifier: Apache-2.0

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
import math as ma

import channel_models.ff_model_jakes as ff

# 6.4.1 HAPS/Satellite antenna, tr.38.811


elevation_angle = np.linspace(-90, 90, 1000)  # An array of values from 0 to 20
angle_att_all = np.zeros(len(elevation_angle))

for i in range(len(elevation_angle)):
    c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
    f = 2
    k = 2 * ma.pi * f / c
    a = 10 * c / f  # The normalized gain pattern for a = 10 c/f (aperture radius of 10 wavelengths) is shown in Figure 6.4.1-1.

    angle = ma.radians(elevation_angle[i])
    x = k * a * ma.sin(angle)
    J1_x = jn(1, x)
    angle_att = ff.linear_to_db(4 * abs(J1_x / (k * a * ma.sin(angle))) ** 2)
    angle_att_all[i] = angle_att

# Plot the Bessel function
plt.ylim(-90,90)
plt.figure(figsize=(10, 6))
plt.plot(elevation_angle, angle_att_all, label="Antenna Model s.t. Bessel function ")
plt.title('Satellite antenna gain pattern for aperture radius 10 wavelengths, a=10 c/f ')
plt.xlabel('Elevation Angle (degree)')
plt.ylabel('Relative antenna gain (dB)')
plt.grid(True)
plt.legend()
plt.show()