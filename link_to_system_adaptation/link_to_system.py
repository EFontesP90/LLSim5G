"""
File: link_to_system.py

Purpose:
This file defines the functions for executing the link to system adaptation and getting the CQI from the SINR and
the target BLER.

Author: Ernesto Fontes Pupo / Claudia Carballo González
Date: 2024-10-30
Version: 1.0.0
SPDX-License-Identifier: Apache-2.0

"""

import numpy as np


class L2S(object):
    """

    Required attributes: (sinr_array, bler_array, target_bler, sinr)

    Returns (over the functions: get_cqi_bler, get_bler, get_sinr_min_target_bler, get_sinr_max_target_bler ):
        cqi: Channel Quality Indicator corresponding to a specific SINR and BLER
        bler: block error rate for a specific CQI and SINR
        sinr_min: Minimum SINR value for a specific CQI according to a defined BLER target.
        sinr_max: Maximum SINR value for a specific CQI according to a defined BLER target.

    """

    def __init__(self, sinr_array, bler_array, target_bler, sinr):

        self.sinr_array = sinr_array  # Numpy array (15x...) with the SINR values corresponding to the respective BLER value for the possible 15 CQI values.
        self.bler_array = bler_array  # Numpy array (15x...) with the BLER values corresponding to the respective SINR value for the possible 15 CQI values.
        self.target_bler = target_bler  # Defined target bler for an adequate signal demodulation
        self.sinr = sinr  # Actual sinr values experienced by the rx



    def get_bler(self, cqi_):


        bler = 0.0
        if self.sinr <= self.sinr_array[cqi_ - 1][0]:
            bler = 1.0
            return bler
        elif (self.sinr >= self.sinr_array[cqi_ - 1][-1]):
            bler = 0.000001
            return bler
        else:
            for i in range(self.sinr_array[cqi_ - 1].size):
                if (self.sinr >= self.sinr_array[cqi_ - 1][i]) and (self.sinr <= self.sinr_array[cqi_ - 1][i + 1]):
                    index = i
                    r = (self.sinr - self.sinr_array[cqi_ - 1][index]) / (self.sinr_array[cqi_ - 1][index + 1] - self.sinr_array[cqi_ - 1][index])
                    bler = self.bler_array[cqi_ - 1][index] + r * (self.bler_array[cqi_ - 1][index + 1] - self.bler_array[cqi_ - 1][index])
        if bler == 0: bler = 0.000001
        return round(bler, 4)


    def get_cqi(self):

        cqi_array = np.array(range(15))
        cqi_array_list = list(cqi_array)
        cqi_array_list.sort(reverse=True)
        cqi_rx = 0
        for i in cqi_array_list:
            bler = self.get_bler(i)
            if bler <= self.target_bler:
                cqi_rx = i + 1
                break
        return cqi_rx, round(bler, 4)

    def sinr_min_target_bler(self):
        sinr_min_sl = np.zeros(15)

        for i in range(len(sinr_min_sl)):
            for ii in range(len(self.bler_array[i])):
                if self.bler_array[i][ii] <= self.target_bler:
                    sinr_min_sl[i] = self.sinr_array[i][ii]
                    break
                else:
                    sinr_min_sl[i] = self.sinr_array[i][-1]
        return sinr_min_sl

    def sinr_max_target_bler(self):
        sinr_max = np.zeros(15)
        sinr_min_sl = self.sinr_min_target_bler()
        for i in range(len(sinr_max)):
             if i < 14: sinr_max[i] = sinr_min_sl[i + 1] - 0.001
             else: sinr_max[i] = self.sinr_array[i][-1]
        return sinr_max


def get_cqi_bler(sinr_array, bler_array, target_bler, sinr):
    l2s = L2S(sinr_array, bler_array, target_bler, sinr)
    cqi, bler = l2s.get_cqi()

    return cqi, bler

def get_bler(sinr_array, bler_array, target_bler, sinr):
    l2s = L2S(sinr_array, bler_array, target_bler, sinr)
    cqi = l2s.get_cqi()
    bler = l2s.get_bler(cqi)
    return bler


def get_sinr_min_target_bler(sinr_array, bler_array, target_bler, sinr):
    l2s = L2S(sinr_array, bler_array, target_bler, sinr)
    sinr_min = l2s.sinr_min_target_bler()
    return sinr_min

def get_sinr_max_target_bler(sinr_array, bler_array, target_bler, sinr):
    l2s = L2S(sinr_array, bler_array, target_bler, sinr)
    sinr_max = l2s.sinr_max_target_bler()
    return sinr_max


