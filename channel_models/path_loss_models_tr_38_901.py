import math as ma
import warnings

import numpy as np
from numpy import random


class UMa_path_loss(object):
    """
    07/05/2024
    Path loss computation for UMa link model s.t. Table 7.4.1-1: Pathloss models,
    3GPP TR 38.901 version 16.1.0 Release 16.

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los):
    """

    def __init__(self, d_2d, d_3d, h_rx, h_tx, fc, los):

        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height

        self.los = los  # True or False variable for enabling or not if the user is in line-of-sight (los).

        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
        if self.d_2d < 10:
            self.d_2d = 10
            self.d_3d = 25.54

        if not ((1.5 <= self.h_rx) and (self.h_rx <= 22.5)):
            #raise "UE height outside the path loss formula's applicability range"
            warnings.warn("Receiver height is out of the path loss formula's applicability range (1.5m <= h_rx <= 22.5) s.t. tr.38.901 for UMa", UserWarning)
        if not (self.h_tx == 25):
            #raise "BS is not the default value"  # Todo: need to check for correction formulas
            warnings.warn("Transmitter height is out of the path loss formula's default value (25m) s.t. tr.38.901 for UMa", UserWarning)

        # compute break-point distance, Note 1: Table 7.4.1-1: Pathloss models
        cc = 0
        if self.h_rx < 13:
            cc = 0
        elif (13 <= self.h_rx) and (self.h_rx <= 23):
            if self.d_2d <= 18:
                g = 0
            else:
                g = (5 / 4) * ((self.d_2d / 100) ** 3) * np.exp(-self.d_2d / 150)
            cc = (((self.h_rx - 13) / 10) ** 1.5) * g
        if random.uniform(0.0, 1.0) < 1 / (1 + cc):
            h_e = 1
        else:
            h_e = random.uniform(0.0,
                                 12)  # TODO: REVIEW, I am not totaly clear about this, check in Table 7.4.1-1: Pathloss models

        h_rx_ = self.h_rx - h_e
        h_tx_ = self.h_tx - h_e
        d_bp = 4 * h_tx_ * h_rx_ * (self.fc * 1000000000) / c  # Break point distance (Table 7.4.1.1, Note 1)
        # Cumpute Path loss

        if 10 <= self.d_2d <= d_bp:
            path_loss_los = 28 + 22 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
        elif d_bp <= self.d_2d <= 5000:
            path_loss_los = 28 + 40 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10) - 9 * ma.log(
                (d_bp ** 2 + (self.h_tx - self.h_rx) ** 2), 10)
        path_loss = path_loss_los
        if not self.los:
            path_loss_nlos = 13.54 + 39.08 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10) - 0.6 * (self.h_rx - 1.5)
            path_loss = max(path_loss_los, path_loss_nlos)

        return path_loss


class UMi_path_loss(object):
    """
    07/05/2024
    Path loss computation for UMi link model s.t. Table 7.4.1-1: Pathloss models,
    3GPP TR 38.901 version 16.1.0 Release 16.

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los):
    """

    def __init__(self, d_2d, d_3d, h_rx, h_tx, fc, los):

        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height

        self.los = los  # True or False variable for enabling or not if the user is in line-of-sight (los).

        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
        if self.d_2d < 10:
            self.d_2d = 10
            self.d_3d = 25.54

        if not ((1.5 <= self.h_rx) and (self.h_rx <= 22.5)):
            #raise "UE height outside the path loss formula's applicability range"
            warnings.warn("Receiver height is out of the path loss formula's applicability range (1.5m <= h_rx <= 22.5) s.t. tr.38.901 for UMi", UserWarning)
        if not (self.h_tx == 10):
            #raise "BS is not the default value"  # Todo: need to check for correction formulas
            warnings.warn("Transmitter height is out of the path loss formula's default value (10m) s.t. tr.38.901 for UMi", UserWarning)

        # compute break-point distance, Note 1: Table 7.4.1-1: Pathloss models
        cc = 0
        if self.h_rx < 13:
            cc = 0
        elif (13 <= self.h_rx) and (self.h_rx <= 23):
            if self.d_2d <= 18:
                g = 0
            else:
                g = (5 / 4) * ((self.d_2d / 100) ** 3) * np.exp(-self.d_2d / 150)
            cc = (((self.h_rx - 13) / 10) ** 1.5) * g
        if random.uniform(0.0, 1.0) < 1 / (1 + cc):
            h_e = 1
        else:
            h_e = random.uniform(0.0,
                                 12)  # TODO: REVIEW, I am not totaly clear about this, check in Table 7.4.1-1: Pathloss models

        h_rx_ = self.h_rx - h_e
        h_tx_ = self.h_tx - h_e
        d_bp = 4 * h_tx_ * h_rx_ * (self.fc * 1000000000) / c  # Break point distance (Table 7.4.1.1, Note 1)
        # Cumpute Path loss

        if 10 <= self.d_2d <= d_bp:
            path_loss_los = 32.4 + 21 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
        elif d_bp <= self.d_2d <= 5000:
            path_loss_los = 32.4 + 40 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10) - 9.5 * ma.log(
                (d_bp ** 2 + (self.h_tx - self.h_rx) ** 2), 10)
        path_loss = path_loss_los
        if not self.los:
            path_loss_nlos = 35.3 * ma.log(self.d_3d, 10) + 22.4 + 21.3 * ma.log(self.fc, 10) - 0.3 * (self.h_rx - 1.5)
            path_loss = max(path_loss_los, path_loss_nlos)

        return path_loss


class RMa_path_loss(object):
    """
    07/05/2024
    Path loss computation for Rma link model s.t. Table 7.4.1-1: Pathloss models,
    3GPP TR 38.901 version 16.1.0 Release 16.

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los):
    """

    def __init__(self, d_2d, d_3d, h_rx, h_tx, fc, los):

        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height

        self.los = los  # True or False variable for enabling or not if the user is in line-of-sight (los).

        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
        if self.d_2d < 10:
            self.d_2d = 10
            self.d_3d = 25.54

        if not ((1 <= self.h_rx) and (self.h_rx <= 10)):
            #raise "UE height outside the path loss formula's applicability range"
            warnings.warn("Receiver height is out of the path loss formula's applicability range (1m <= h_rx <= 10) s.t. tr.38.901 for RMa", UserWarning)
        if not ((10 <= self.h_tx) and (self.h_tx <= 150)):
            #raise "BS is not the default value"  # Todo: need to check for correction formulas
            warnings.warn("Transmitter height is out of the path loss formula's default value (10m <= h_rx <= 150) s.t. tr.38.901 for RMa", UserWarning)

        # compute break-point distance, Note 1: Table 7.4.1-1: Pathloss models
        d_bp = 2 * ma.pi * self.h_tx * self.h_tx * (self.fc * 1000000000) / c  # Break point distance (Table 7.4.1.1, Note 5)
        # Cumpute Path loss

        # TODO: REVIEW, Evaluar if it is valuable to make h and w inputs.
        h = 5  # h = avg. building height, The applicability ranges: 5m ≤ h ≤50m
        w = 20  # W = avg. street width, The applicability ranges: 5m ≤ h ≤50m

        if 10 <= self.d_2d <= d_bp:
            path_loss_los = 20 * ma.log(40 * ma.pi * self.d_3d * self.fc / 3, 10) \
                                   + min(0.03 * (h ** 1.72), 10) * ma.log(self.d_3d, 10) \
                                   - min(0.044 * (h ** 1.72), 14.77) \
                                   + 0.002 * ma.log(h, 10) * self.d_3d
        elif d_bp <= self.d_2d <= 10000:
            path_loss_los = 20 * ma.log(40 * ma.pi * self.d_3d * self.fc / 3, 10) \
                                   + min(0.03 * (h ** 1.72), 10) * ma.log(self.d_3d, 10) \
                                   - min(0.044 * (h ** 1.72), 14.77) \
                                   + 0.002 * ma.log(h, 10) * self.d_3d \
                                   + 40 * ma.log(self.d_3d/d_bp, 10)
        path_loss = path_loss_los
        if not self.los:
            if d_bp <= self.d_2d <= 5000:
                path_loss_nlos = 161.04 - 7.1 * ma.log(w, 10) \
                                  + 7.5 * ma.log(h, 10) \
                                  - (24.37 - 3.7 * ((h / self.h_tx) ** 2)) * ma.log(self.h_tx, 10) \
                                  + (43.42 - 3.1 * ma.log(self.h_tx, 10)) * (ma.log(self.d_3d, 10) - 3) \
                                  + 20 * ma.log(self.fc, 10) \
                                  - (3.2 * (ma.log(11.75 * self.h_rx, 10) ** 2) - 4.97)

                path_loss = max(path_loss_los, path_loss_nlos)

        return path_loss


class InH_path_loss(object):
    """
    07/05/2024
    Path loss computation for InH link model s.t. Table 7.4.1-1: Pathloss models,
    3GPP TR 38.901 version 16.1.0 Release 16.

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los):
    """

    def __init__(self, d_2d, d_3d, h_rx, h_tx, fc, los):

        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height

        self.los = los  # True or False variable for enabling or not if the user is in line-of-sight (los).

        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
        if self.d_2d < 1:
            self.d_2d = 1
            self.d_3d = 3.16


        # Cumpute Path loss

        if 1 <= self.d_3d <= 150:
            path_loss_los = 32.4 + 17.3 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
        elif self.d_3d > 150:
            raise "The d_3D is outside the path loss formula's applicability range (1 <= d_3D <= 150)"
        path_loss = path_loss_los
        if not self.los:
            if 1 <= self.d_3d <= 150:
                path_loss_nlos = 38.3 * ma.log(self.d_3d, 10) + 17.30 + 24.9 * ma.log(self.fc, 10)
                path_loss = max(path_loss_los, path_loss_nlos)
            elif self.d_3d > 150:
                raise "The d_3D is outside the path loss formula's applicability range (1 <= d_3D <= 150)"

        return path_loss


class InF_SL_path_loss(object):
    """
    07/05/2024
    Path loss computation for InF-SL link model s.t. Table 7.4.1-1: Pathloss models,
    3GPP TR 38.901 version 16.1.0 Release 16.

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los):
    """

    def __init__(self, d_2d, d_3d, h_rx, h_tx, fc, los):

        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height

        self.los = los  # True or False variable for enabling or not if the user is in line-of-sight (los).

        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
        if self.d_2d < 1:
            self.d_2d = 1
            self.d_3d = 3.16


        # Cumpute Path loss

        if 1 <= self.d_3d <= 600:
            path_loss_los = 31.84 + 21.5 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
        elif self.d_3d > 150:
            raise "The d_3D is outside the path loss formula's applicability range (1 <= d_3D <= 600)"
        path_loss = path_loss_los
        if not self.los:
            if 1 <= self.d_3d <= 600:
                path_loss_nlos = 33 + 25.5 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
                path_loss = max(path_loss_los, path_loss_nlos)
            elif self.d_3d > 600:
                raise "The d_3D is outside the path loss formula's applicability range (1 <= d_3D <= 600)"

        return path_loss


class InF_DL_path_loss(object):
    """
    07/05/2024
    Path loss computation for InF-DL link model s.t. Table 7.4.1-1: Pathloss models,
    3GPP TR 38.901 version 16.1.0 Release 16.

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los):
    """

    def __init__(self, d_2d, d_3d, h_rx, h_tx, fc, los):

        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height

        self.los = los  # True or False variable for enabling or not if the user is in line-of-sight (los).

        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
        if self.d_2d < 1:
            self.d_2d = 1
            self.d_3d = 3.16


        # Cumpute Path loss

        if 1 <= self.d_3d <= 600:
            path_loss_los = 31.84 + 21.5 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
        elif self.d_3d > 150:
            raise "The d_3D is outside the path loss formula's applicability range (1 <= d_3D <= 600)"
        path_loss = path_loss_los
        if not self.los:
            if 1 <= self.d_3d <= 600:
                path_loss_nlos = 18.6 + 35.7 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
                path_loss = max(path_loss_los, path_loss_nlos)
            elif self.d_3d > 600:
                raise "The d_3D is outside the path loss formula's applicability range (1 <= d_3D <= 600)"

        return path_loss


class InF_SH_path_loss(object):
    """
    07/05/2024
    Path loss computation for InF-SH link model s.t. Table 7.4.1-1: Pathloss models,
    3GPP TR 38.901 version 16.1.0 Release 16.

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los):
    """

    def __init__(self, d_2d, d_3d, h_rx, h_tx, fc, los):

        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height

        self.los = los  # True or False variable for enabling or not if the user is in line-of-sight (los).

        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
        if self.d_2d < 1:
            self.d_2d = 1
            self.d_3d = 3.16


        # Cumpute Path loss

        if 1 <= self.d_3d <= 600:
            path_loss_los = 31.84 + 21.5 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
        elif self.d_3d > 150:
            raise "The d_3D is outside the path loss formula's applicability range (1 <= d_3D <= 600)"
        path_loss = path_loss_los
        if not self.los:
            if 1 <= self.d_3d <= 600:
                path_loss_nlos = 32.4 + 23.0 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
                path_loss = max(path_loss_los, path_loss_nlos)
            elif self.d_3d > 600:
                raise "The d_3D is outside the path loss formula's applicability range (1 <= d_3D <= 600)"

        return path_loss


class InF_DH_path_loss(object):
    """
    07/05/2024
    Path loss computation for InF-DH link model s.t. Table 7.4.1-1: Pathloss models,
    3GPP TR 38.901 version 16.1.0 Release 16.

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los, outdoor_to_indoor):
    """

    def __init__(self, d_2d, d_3d, h_rx, h_tx, fc, los):

        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height

        self.los = los  # True or False variable for enabling or not if the user is in line-of-sight (los).

        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
        if self.d_2d < 1:
            self.d_2d = 1
            self.d_3d = 3.16


        # Cumpute Path loss

        if 1 <= self.d_3d <= 600:
            path_loss_los = 31.84 + 21.5 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
        elif self.d_3d > 150:
            raise "The d_3D is outside the path loss formula's applicability range (1 <= d_3D <= 600)"
        path_loss = path_loss_los
        if not self.los:
            if 1 <= self.d_3d <= 600:
                path_loss_nlos = 33.6 + 21.9 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
                path_loss = max(path_loss_los, path_loss_nlos)
            elif self.d_3d > 600:
                raise "The d_3D is outside the path loss formula's applicability range (1 <= d_3D <= 600)"

        return path_loss


class D2D_path_loss(object):
    """
    07/05/2024
    Path loss computation for D2D link model s.t. Table 7.4.1-1: Pathloss models,
    3GPP TR 38.901 version 16.1.0 Release 16, for Umi without the restriction of bs height = 10.

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los):
    """

    def __init__(self, d_2d, d_3d, h_rx, h_tx, fc, los):

        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height

        self.los = los  # True or False variable for enabling or not if the user is in line-of-sight (los).

        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
        if self.d_2d < 10:
            self.d_2d = 10
            self.d_3d = 25.54

        if not ((1.5 <= self.h_rx) and (self.h_rx <= 22.5)):
            #raise "UE height outside the path loss formula's applicability range"
            warnings.warn("Receiver height is out of the path loss formula's applicability range (1.5m <= h_rx <= 22.5) s.t. tr.38.901 for UMi",
            UserWarning)
        if not ((1.5 <= self.h_tx) and (self.h_tx <= 22.5)):
            #raise "Forwarding devise (fd) height outside the path loss formula's applicability range"
            warnings.warn("Forwarding devise (fd) height is out of the path loss formula's applicability range (1.5m <= h_rx <= 22.5) s.t. tr.38.901 for UMi",
            UserWarning)

        # compute break-point distance, Note 1: Table 7.4.1-1: Pathloss models
        cc = 0
        if self.h_rx < 13:
            cc = 0
        elif (13 <= self.h_rx) and (self.h_rx <= 23):
            if self.d_2d <= 18:
                g = 0
            else:
                g = (5 / 4) * ((self.d_2d / 100) ** 3) * np.exp(-self.d_2d / 150)
            cc = (((self.h_rx - 13) / 10) ** 1.5) * g
        if random.uniform(0.0, 1.0) < 1 / (1 + cc):
            h_e = 1
        else:
            h_e = random.uniform(0.0,
                                 12)  # TODO: REVIEW, I am not totaly clear about this, check in Table 7.4.1-1: Pathloss models

        h_rx_ = self.h_rx - h_e
        h_tx_ = self.h_tx - h_e
        d_bp = 4 * h_tx_ * h_rx_ * (self.fc * 1000000000) / c  # Break point distance (Table 7.4.1.1, Note 1)
        # Cumpute Path loss

        if 10 <= self.d_2d <= d_bp:
            path_loss_los = 32.4 + 21 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10)
        elif d_bp <= self.d_2d <= 5000:
            path_loss_los = 32.4 + 40 * ma.log(self.d_3d, 10) + 20 * ma.log(self.fc, 10) - 9.5 * ma.log(
                (d_bp ** 2 + (self.h_tx - self.h_rx) ** 2), 10)
        path_loss = path_loss_los
        if not self.los:
            path_loss_nlos = 35.3 * ma.log(self.d_3d, 10) + 22.4 + 21.3 * ma.log(self.fc, 10) - 0.3 * (self.h_rx - 1.5)
            path_loss = max(path_loss_los, path_loss_nlos)

        return path_loss


def uma_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los):
    uma_pl = UMa_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los)
    path_loss = uma_pl.compute_path_loss()
    return path_loss


def umi_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los):
    umi_pl = UMi_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los)
    path_loss = umi_pl.compute_path_loss()
    return path_loss


def rma_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los):
    rma_pl = RMa_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los)

    path_loss = rma_pl.compute_path_loss()
    return path_loss


def inh_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los):
    inh_pl = InH_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los)

    path_loss = inh_pl.compute_path_loss()
    return path_loss


def inf_sl_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los):
    inf_sl_pl = InF_SL_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los)

    path_loss = inf_sl_pl.compute_path_loss()
    return path_loss


def inf_dl_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los):
    inf_dl_pl = InF_DL_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los)

    path_loss = inf_dl_pl.compute_path_loss()
    return path_loss


def inf_sh_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los):
    inf_sh_pl = InF_SH_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los)

    path_loss = inf_sh_pl.compute_path_loss()
    return path_loss


def inf_dh_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los):
    inf_dh_pl = InF_DH_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los)

    path_loss = inf_dh_pl.compute_path_loss()
    return path_loss

def d2d_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los):
    d2d_pl = D2D_path_loss(d_2d, d_3d, h_rx, h_tx, fc, los)

    path_loss = d2d_pl.compute_path_loss()
    return path_loss