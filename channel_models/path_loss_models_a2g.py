"""
File: path_loss_models_a2g.py

Purpose:
This file comprises the Path loss computation for A2G link model used between abs (UAV acting as a BSs) and EDs, s.t.
Khawaja, W., Guvenc, I., Matolak, D. W., Fiebig, U. C., & Schneckenburger, N. (2019). A survey of air-to-ground
propagation channel modeling for unmanned aerial vehicles. IEEE Communications Surveys & Tutorials, 21(3), 2361-2391.

Author: Ernesto Fontes Pupo / Claudia Carballo González
Date: 2024-10-30
Version: 1.0.0
SPDX-License-Identifier: Apache-2.0

"""

# Third-party imports
import math as ma

class A2G_path_loss(object):
    """
    07/05/2024

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los)

    Output(a2g_path_loss)
    path_loss
    """

    def __init__(self, d_2d, d_3d, h_rx, h_tx, fc, los):
        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height

        self.los = los  # True or False variable for enabling or not if the user is in line-of-sight (los).

        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        # TODO: REVIEW, check this implementation.
        c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
        n_los = 1.6
        n_nlos = 23
        path_loss_los = 20 * ma.log(((self.d_3d * self.fc * 1000000000 * 4 * 3.1416) / c), 10) + n_los
        path_loss = path_loss_los
        if not self.los:
            path_loss_nlos = 20 * ma.log(((self.d_3d * self.fc * 1000000000 * 4 * 3.1416) / c), 10) + n_nlos
            path_loss = max(path_loss_los, path_loss_nlos)

        return path_loss


def a2g_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los):
    a2g_pl = A2G_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los)

    path_loss = a2g_pl.compute_path_loss()
    return path_loss