"""
File: atmospheric_absorption_tn.py

Purpose:
This file defines...


Author: Ernesto Fontes Pupo / Claudia Carballo González
        University of Cagliari
Date: 2024-10-30
Version: 1.0.0
SPDX-License-Identifier: Apache-2.0

"""

import numpy as np
import math as ma
import matplotlib.pyplot as plt


fc_array = np.linspace(1, 100, 1000)  # An array of values from 0 to 20
att_los = np.zeros(len(fc_array))
att_nlos = np.zeros(len(fc_array))

dds = 100*1e-9  # Nominal delay spread
# Speed of light
c = 3e8  # m/s
# Calculate OL_n(fc)

for f in range(len(fc_array)):
    fc = fc_array[f]
    if fc <= 52:
        alpha_fc = 0
    elif fc <= 53:
        alpha_fc = 1
    elif fc <= 54:
        alpha_fc = 2.2
    elif fc <= 55:
        alpha_fc = 4
    elif fc <= 56:
        alpha_fc = 6.6
    elif fc <= 57:
        alpha_fc = 9.7
    elif fc <= 58:
        alpha_fc = 12.6
    elif fc <= 59:
        alpha_fc = 14.6
    elif fc <= 60:
        alpha_fc = 15
    elif fc <= 61:
        alpha_fc = 14.6
    elif fc <= 62:
        alpha_fc = 14.3
    elif fc <= 63:
        alpha_fc = 10.5
    elif fc <= 64:
        alpha_fc = 6.8
    elif fc <= 65:
        alpha_fc = 3.9
    elif fc <= 66:
        alpha_fc = 1.9
    elif fc <= 67:
        alpha_fc = 1
    elif fc > 67:
        alpha_fc = 0


    ds = 363e-9
    d_3d = 1000
    # if los:
    r_tau = 3.0
    tau_delta = 0
    # c_delay_normalized_los = 0
    c_delay_normalized_los = 0.5596  # "D_ntn" tap 2, Table 6.9.2-4. NTN-TDL-D at elevation
    print((alpha_fc / 1000) * (d_3d + c * (dds * c_delay_normalized_los + tau_delta)))

    att_los[f] = (alpha_fc / 1000) * (d_3d + c * (dds * c_delay_normalized_los + tau_delta))


    # else:
    r_tau = 2.1
    # c_delay_normalized_nlos = 0
    c_delay_normalized_nlos = 1.0811  # "A_ntn" tap 2, Table 6.9.2-1. NTN-TDL-A at elevation
    x_n = np.random.uniform(0, 1)
    tau_prime_n = -r_tau * ds * np.log(x_n)
    tau_delta = tau_prime_n
    att_nlos[f] = (alpha_fc / 1000) * (d_3d + c * (dds * c_delay_normalized_nlos + tau_delta))


plt.ylim(-5,300)
plt.yscale('log')
plt.xscale('symlog')
plt.figure(figsize=(10, 6))
plt.plot(fc_array, att_los, label="att_los", color="red")
plt.plot(fc_array, att_nlos, label="att_nlos", color="blue")

plt.title('7.6.1 Oxygen absorption  ')
plt.xlabel('Fc (GHz)')
plt.ylabel('Oxygen absorption (dB)')
# plt.yscale('log')
# plt.xscale('symlog')
plt.xlim(1,100)
plt.grid(True)
plt.legend()
plt.show()