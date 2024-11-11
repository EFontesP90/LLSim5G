"""
File: terrestrial_antenna_model.py

Purpose:
This file is used to validate the implementation of the TN antenna model.


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


h_angle = np.linspace(-180, 180, 1000)  # An array of values from 0 to 20
angle_att_all = np.zeros(len(h_angle))

for i in range(len(h_angle)):
    max_h_angle_att = 30  # is the 3 dB beamwidth (corresponding to h_angle_3dB= 70º)
    h_angle_att = -min(12 * pow(h_angle[i] / 65, 2), max_h_angle_att)
    angle_att_all[i] = h_angle_att

# Plot the Bessel function
plt.ylim(-180,180)
plt.figure(figsize=(10, 6))
plt.plot(h_angle, angle_att_all, label="Antenna Model s.t. [ITU-R M.2135]")
plt.title('Simplified antenna gain pattern of a single antenna element (for TN), s.t. tr.38.901, Table 7.3-1')
plt.xlabel('Horizontal Angle (degree)')
plt.ylabel('Relative antenna gain (dB)')
plt.grid(True)
plt.legend()
plt.show()