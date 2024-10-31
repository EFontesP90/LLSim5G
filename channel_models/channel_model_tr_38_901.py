"""
File: channel_model_tr_38_901.py

Purpose:
This file comprises a terrestrial network (TN) Channel model fully implementation according to 3gpp tr-38-901. For a pair tx (e.g., tbs, abs, or
d2d possible forwarding user) and rx (e.g., user equipment), our implementation computes the o2i probability,
o2i losses, los probability, hb probability and losses, shadowing fading, the angular attenuation, the path loss,
the fast fading attenuation and finally the resulting SINR.

Author: Ernesto Fontes Pupo / Claudia Carballo González
Date: 2024-10-30
Version: 1.0.0
SPDX-License-Identifier: Apache-2.0

"""

# Third-party imports
import math as ma
import numpy as np
from numpy import random
from numpy.random import normal
from scipy.stats import lognorm

# Local application/library-specific imports
import channel_models.ff_model_jakes as jakes
from channel_models import path_loss_models_a2g as pl_a2g
from channel_models import path_loss_models_tr_38_901 as pl_tn
from channel_models.ff_models_tr_38_901and811 import tdl_mdels as tdl
from channel_models.ff_models_tr_38_901and811 import cdl_models as cdl
from channel_models.geometry import geometry as gm
from link_to_system_adaptation import bler_curves as bl
from link_to_system_adaptation import link_to_system as l2s


class Ch_tr_138_901(object):
    """
    14/05/2024
    Channel model fully implementation according to 3gpp tr-38-901. For a pair tx (e.g., tbs, abs, or d2d possible
    forwarding user) and rx (e.g., user equipment), our implementation computes the o2i probability,
    o2i losses, los probability, hb probability and losses, shadowing fading, the angular attenuation, the path loss,
    the fast fading attenuation and finally the resulting SINR.

    Required attributes:
    (channel_model, tx_antenna_mode, shadowing, dynamic_los, dynamic_hb, outdoor_to_indoor, inside_what_o2i, penetration_loss_model,
                 d_2d, d_3d, h_rx, h_tx, h_ceiling, block_density, fc, d_correlation_map_rx, t_now, t_old,
                 speed_rx, speed_tx, rx_coord, tx_coord, h_angle, v_angle, ds_angle, v_tilt, n_rb, jakes_map, fast_fading_model, hb_map_rx,
                 cable_loss_tx, thermal_noise, bw_rb, rx_noise_figure, fast_fading, tx_power, antenna_gain_tx, antenna_gain_rx,
                 atmospheric_absorption, desired_delay_spread, fast_fading_los_type, fast_fading_nlos_type, num_rx_ax, num_tx_ax,
                 rng, rx_antenna_mode, ax_panel_polarization):

    Outputs (get_ch_tr_38_901):
    ch_outcomes_rx: Dictionary with the main channel outputs: {"t": d_correlation_map_rx["t"], "x": d_correlation_map_rx["x"],
    "y": d_correlation_map_rx["y"], "o2i": d_correlation_map_rx["o2i"], "los": d_correlation_map_rx["los"],
    "o2i_loss": d_correlation_map_rx["o2i_loss"], "shadowing": d_correlation_map_rx["shadowing"], "path_loss": path_loss,
    "angle_att": angle_att, "hb_attenuation": hb_attenuation, "fast_fading_att": round(np.mean(fast_fading_att), 2),
    "sinr": sinr}
    d_correlation_map_rx: Matrix for tracking the muvement of each rx and cheking the correlation distance over time.
    hb_map_rx: Mapping for tracking if a user is in hb conditions from t-1 (t_old) to t (t_now).
    jakes_map: Matrix for storing the multipath components from t-1 (t_old) to t (t_now) avoiding abrupt changes in the
    resulting fast-fading attenuation.


    """

    def __init__(self, channel_model, tx_antenna_mode, shadowing, dynamic_los, dynamic_hb, outdoor_to_indoor, inside_what_o2i, penetration_loss_model,
                 d_2d, d_3d, h_rx, h_tx, h_ceiling, block_density, fc, d_correlation_map_rx, t_now, t_old,
                 speed_rx, speed_tx, rx_coord, tx_coord, h_angle, v_angle, ds_angle, v_tilt, n_rb, jakes_map, fast_fading_model, hb_map_rx,
                 cable_loss_tx, thermal_noise, bw_rb, rx_noise_figure, fast_fading, tx_power, antenna_gain_tx, antenna_gain_rx,
                 atmospheric_absorption, desired_delay_spread, fast_fading_los_type, fast_fading_nlos_type, num_rx_ax, num_tx_ax,
                 rng, rx_antenna_mode, ax_panel_polarization):

        assert h_ceiling <= 10, f"The defined ceiling height: {h_ceiling} for InF link modes,  must be equal or lower than 10 meters"

        self.channel_model = channel_model  # string with the selected link channel to be modeled from the tr_138_901: Just valid for UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH.
        self.tx_antenna_mode = tx_antenna_mode  # string with the selected tx antenna mode: omni, three_sectors, four_sectors.

        self.shadowing = shadowing
        self.dynamic_los = dynamic_los  # True or False variable for enabling or not the dynamic line-of-sight (los) mode of the end devices.
        self.dynamic_hb = dynamic_hb  # True or False variable for enabling or not the dynamic human blockage (hb) mode of the end devices.
        self.outdoor_to_indoor = outdoor_to_indoor  # True or False values for enabling if the ue is modeled as inside a building/car or not.
        self.inside_what_o2i = inside_what_o2i  # String (building, car, dynamic) for defining if a user is inside a car a building or can change dynamically.
        self.penetration_loss_model = penetration_loss_model  # string (low-loss, high-loss) for defining the penetration_loss_model: Low-loss, High-loss, for indoor penetratio, or can change dynamically.

        self.d_2d = d_2d  # two-dimensional distance between rx and tx
        self.d_3d = d_3d  # Three-dimensional distance between rx and tx
        self.h_rx = h_rx  # rx height
        self.h_tx = h_tx  # tx height
        self.h_ceiling = h_ceiling  # Ceiling height, < 0-10 m. For InF
        self.block_density = block_density  # human block density, valid when dynamic_hb is True

        self.fc = fc  # Simulation Frequency in GHz


        self.t_now = t_now  # current time step
        self.t_old = t_old  # previous time step
        self.speed_rx = speed_rx
        self.speed_tx = speed_tx
        self.rx_coord = rx_coord
        self.tx_coord = tx_coord

        self.h_angle = h_angle
        self.v_angle = v_angle
        self.ds_angle = ds_angle
        self.v_tilt = v_tilt

        self.n_rb = n_rb  # Nuber of rb assigned to the ue
        self.fast_fading_model = fast_fading_model  # "jakes"
        self.d_correlation_map_rx = d_correlation_map_rx  # Array of 1*2: [t, x, y, std], For tracking the shadowing of the user over time
        self.jakes_map = jakes_map  # array of the resulting jackFadingMap of the K users respect to the BS and each proximity user in D2D comm, fadingTaps X [delay amp, AoA]

        self.hb_map_rx = hb_map_rx

        self.cable_loss_tx = cable_loss_tx
        self.thermal_noise = thermal_noise
        self.bw_rb = bw_rb
        self.rx_noise_figure = rx_noise_figure

        self.fast_fading = fast_fading
        self.tx_power = tx_power
        self.antenna_gain_tx = antenna_gain_tx
        self.antenna_gain_rx = antenna_gain_rx

        self.atmospheric_absorption = atmospheric_absorption
        self.desired_delay_spread = desired_delay_spread
        self.fast_fading_los_type = fast_fading_los_type
        self.fast_fading_nlos_type = fast_fading_nlos_type
        self.num_rx_ax = num_rx_ax
        self.num_tx_ax = num_tx_ax
        self.bearing_angle = 0
        self.down_tilt_angle = v_tilt

        self.rng = rng
        self.rx_antenna_mode = rx_antenna_mode
        self.ax_panel_polarization = ax_panel_polarization
        self.delta_h = self.h_tx - self.h_rx

    def compute_o2i_probability(self):
        if not self.outdoor_to_indoor:
            o2i = False  # 0 dB
            return o2i
        else:
            o2i_p = random.uniform(0.0, 1.0)
            if self.channel_model == "UMa" or self.channel_model == "UMi" or self.channel_model == "A2G":
                if 0.2 < o2i_p:  # Table 7.2-1: Evaluation parameters for UMi-street canyon and UMa scenarios
                    o2i = True
                else:
                    o2i = False
            elif self.channel_model == "RMa":
                if 0.5 < o2i_p:  # Table 7.2-3: Evaluation parameters for RMa
                    o2i = True
                else:
                    o2i = False
            else: o2i = False
        return o2i

    def compute_los_probability(self, o2i):  # ETSI-TR 138 901 V16.1.0 (2020-11), page 30. Table 7.4.2-1 LOS probability
        # Note: The LOS probability is derived with assuming antenna heights of 3m for indoor, 10m for UMi, and 25m for UMa: (Table 7.4.2-1 LOS probability)

        p = 0.0
        if not self.dynamic_los:
            los = True
            return los

        else:
            if self.channel_model == "UMi":
                if o2i:
                    los = False
                    return los

                if self.d_2d <= 18.0:
                    p = 1
                else:
                    p = (18.0 / self.d_2d) + + ma.exp(-1 * self.d_2d / 36) * (1.0 - (18.0 / self.d_2d))

            elif self.channel_model == "UMa":
                if o2i:
                    los = False
                    return los

                if self.d_2d <= 18.0:
                    p = 1
                else:
                    if self.h_rx <= 13.0:
                        cc = 0
                    else:
                        cc = pow((self.h_rx - 13.0) / 10.0, 1.5)
                    p = ((18.0 / self.d_2d) + ma.exp(-1 * self.d_2d / 63) * (1.0 - (18.0 / self.d_2d))) * (
                            1 + cc * (5.0 / 4.0) * pow((self.d_2d / 100.0), 3) * ma.exp(-1 * self.d_2d / 150.0))

            elif self.channel_model == "RMa":
                if o2i:
                    los = False
                    return los

                if self.d_2d <= 10.0:
                    p = 1
                else:
                    p = ma.exp(-(self.d_2d - 10) / 1000)

            elif self.channel_model == "A2G":  # TODO: I must be clear of the literature behind this and must add the reference.
                if o2i:
                    los = False
                    return los

                a = 12.1
                b = 0.51
                p = 1 / (1 + a * ma.exp(-b * (self.v_angle - a)))

            elif self.channel_model == "InH-Mixed":  # Indoor - Mixed office
                if self.d_2d <= 1.2:
                    p = 1
                elif 1.2 < self.d_2d < 6.5:
                    p = ma.exp(-1 * (self.d_2d - 1.2) / 4.7)
                elif 6.5 <= self.d_2d:
                    p = ma.exp(-1 * (self.d_2d - 6.5) / 32.6)

            elif self.channel_model == "InH-Open":  # Indoor - Open office
                if self.d_2d <= 5:
                    p = 1
                elif 5 < self.d_2d < 49:
                    p = ma.exp(-1 * (self.d_2d - 5) / 70.8)
                elif 49 <= self.d_2d:
                    p = ma.exp(-1 * (self.d_2d - 49) / 211.7)

            elif self.channel_model == "InF-HH":
                p = 1  # Indoor Factory with High Tx and High Rx (both elevated above the clutter)

            elif self.channel_model == "InF-SL":  # Indoor Factory
                d_clutter = 10  # Table 7.2-4: Evaluation parameters for InF
                r = random.uniform(0.0, 0.399)  # Table 7.2-4: Evaluation parameters for InF
                k_subsce = -1 * d_clutter / (ma.log(1 - r))
                p = ma.exp(-1 * (self.d_2d) / k_subsce)

            elif self.channel_model == "InF-DL":  # Indoor Factory
                d_clutter = 2  # Table 7.2-4: Evaluation parameters for InF
                r = random.uniform(0.40, 0.99)  # Table 7.2-4: Evaluation parameters for InF
                k_subsce = -1 * d_clutter / (ma.log(1 - r))
                p = ma.exp(-1 * (self.d_2d) / k_subsce)

            elif self.channel_model == "InF-SH":  # Indoor Factory
                d_clutter = 10  # Table 7.2-4: Evaluation parameters for InF
                r = random.uniform(0.0, 0.399)  # Table 7.2-4: Evaluation parameters for InF
                h_clutter = self.h_ceiling
                k_subsce = (-1 * d_clutter / (ma.log(1 - r))) * ((self.h_tx - self.h_rx) / (h_clutter - self.h_rx))
                p = ma.exp(-1 * (self.d_2d) / k_subsce)

            elif self.channel_model == "InF-DH":  # Indoor Factory
                d_clutter = 2  # Table 7.2-4: Evaluation parameters for InF
                r = random.uniform(0.0, 0.399)  # Table 7.2-4: Evaluation parameters for InF
                h_clutter = self.h_ceiling
                k_subsce = (-1 * d_clutter / (ma.log(1 - r))) * ((self.h_tx - self.h_rx) / (h_clutter - self.h_rx))
                p = ma.exp(-1 * (self.d_2d) / k_subsce)

            elif self.channel_model == "D2D":  # TODO> Check this
                if o2i:
                    los = False
                    return los

                if self.d_2d <= 18.0:
                    p = 1
                else:
                    p = (18.0 / self.d_2d) + ma.exp(-1 * self.d_2d / 36.0) * (1.0 - (18.0 / self.d_2d))

        r = random.uniform(0.0, 1.0)
        if r <= p:
            los = True
        else:
            los = False

        return los

    def compute_hb_probability(self):
        h_bl = 1.7
        r_bl = 0.3
        p = 0.0

        if not self.dynamic_hb:
            hb = False
            return hb
        else:
            if self.h_tx != self.h_rx:
                p = (1 - ma.exp(-2 * self.block_density * r_bl * (
                        ma.sqrt(pow(self.d_3d, 2) - pow(self.h_tx - self.h_rx, 2)) * (
                        (h_bl - self.h_rx) / (self.h_tx - self.h_rx)) + r_bl)))
            else:
                # TODO: For D2D I do not know how change the HB probability in D2D, search in literature to update.
                h_tx_ = self.h_tx + 0.1

                p = (1 - ma.exp(-2 * self.block_density * r_bl * (
                        ma.sqrt(pow(self.d_3d, 2) - pow(h_tx_ - self.h_rx, 2)) * (
                        (h_bl - self.h_rx) / (h_tx_ - self.h_rx)) + r_bl)))

        r = random.uniform(0.0, 1.0)
        if (r <= p):
            h_blockage = True
        else:
            h_blockage = False

        return h_blockage

    def get_hb_attenuation(self):
        d_correlation_hb = 2  # TODO: Check this human blockage correlation value.
        h_attenuation = 15 # TODO: Check this human blockage value.

        if not self.dynamic_hb:
            hb_attenuation = 0
            self.hb_map_rx["t"] = self.t_now
            self.hb_map_rx["x"] = self.rx_coord[0]
            self.hb_map_rx["y"] = self.rx_coord[1]
            self.hb_map_rx["h_blockage"] = False
            return hb_attenuation, self.hb_map_rx

        elif self.t_now == 0:
            self.hb_map_rx["t"] = self.t_now
            self.hb_map_rx["x"] = self.rx_coord[0]
            self.hb_map_rx["y"] = self.rx_coord[1]
            self.hb_map_rx["h_blockage"] = self.compute_hb_probability()

            if self.hb_map_rx["h_blockage"]: hb_attenuation = h_attenuation
            else: hb_attenuation = 0
            return hb_attenuation, self.hb_map_rx

        delta_xy = ma.sqrt((self.rx_coord[0] - self.hb_map_rx["x"])**2 + (self.rx_coord[1] - self.hb_map_rx["y"])**2)
        if delta_xy > d_correlation_hb:
            self.hb_map_rx["t"] = self.t_now
            self.hb_map_rx["x"] = self.rx_coord[0]
            self.hb_map_rx["y"] = self.rx_coord[1]
            self.hb_map_rx["h_blockage"] = self.compute_hb_probability()

        if self.hb_map_rx["h_blockage"]: hb_attenuation = h_attenuation
        else: hb_attenuation = 0

        return hb_attenuation, self.hb_map_rx

    # NOTE: 7.4.3 O2I penetration loss
    def compute_o2i_loss(self, o2i):  # Valid just for Frequency Lower than 6GHz
        o2i_loss = 0
        if not o2i:
            o2i_loss = 0  # 0 dB
            return o2i_loss
        else:
            if self.inside_what_o2i == "dynamic": inside_what = random.choice(["building", "car"])
            else: inside_what = self.inside_what_o2i

        if inside_what == "building":
            # NOTE: 7.4.3.1 O2I building penetration loss
            if self.channel_model == "UMa":
                d_2d_in = min(random.uniform(0, 25), random.uniform(0, 25))
            elif self.channel_model == "UMi":
                d_2d_in = min(random.uniform(0, 10), random.uniform(0, 10))
            else:  # TODO: I am not clear of this
                d_2d_in = min(random.uniform(0, 20), random.uniform(0, 20))

            if self.fc < 6:  # NOTE: For backwards compatibility with TR 36.873
                # NOTE: O2I building penetration loss model for single-frequency simulations <6 GHz >>>Table 7.4.3-3.
                ploss_in = 0.5 * d_2d_in
                ploss_tw = 20.0
                o2i_loss = ploss_in + ploss_tw

            elif self.fc >= 6:
                l_glass = 2 + 0.2 * self.fc
                l_iirglass = 23 + 0.3 * self.fc
                l_concrete = 5 + 4 * self.fc
                l_wood = 4.85 + 0.12 * self.fc

                loss_model = self.penetration_loss_model

                if loss_model == "low-loss":
                    ploss_tw = 5 - 10 * ma.log((0.3 * pow(10, -l_glass / 10)) + (0.7 * pow(10, -l_concrete / 10)), 10)
                    ploss_in = 0.5 * d_2d_in
                    normal_dist = normal(loc=0, scale=4.4, size=1)[0]
                    o2i_loss = ploss_tw + ploss_in + normal_dist

                elif loss_model == "high-loss":
                    ploss_tw = 5 - 10 * ma.log((0.7 * pow(10, -l_iirglass / 10)) + (0.3 * pow(10, -l_concrete / 10)),
                                               10)
                    ploss_in = 0.5 * d_2d_in
                    normal_dist = normal(loc=0, scale=6.5, size=1)[0]
                    o2i_loss = ploss_tw + ploss_in + normal_dist

        elif inside_what == "car":
            # NOTE: 7.4.3.2 O2I car penetration loss
            # The O2I car penetration loss models are applicable for at least 0.6-60 GHz.
            mu = 9.0  # 20 for metalized car windows
            sigma_p = 5.0
            o2i_loss = normal(loc=mu, scale=sigma_p, size=1)[0]

        return o2i_loss

    def compute_path_loss(self, los): #valid for UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH

        if self.channel_model == "UMi": path_loss = pl_tn.umi_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)
        elif self.channel_model == "UMa": path_loss = pl_tn.uma_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)
        elif self.channel_model == "RMa": path_loss = pl_tn.rma_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)
        elif self.channel_model == "InH-Mixed": path_loss = pl_tn.inh_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)
        elif self.channel_model == "InH-Open": path_loss = pl_tn.inh_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)
        elif self.channel_model == "InF-HH": path_loss = pl_tn.inf_sl_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)
        elif self.channel_model == "InF-SL": path_loss = pl_tn.inf_sl_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)
        elif self.channel_model == "InF-DL": path_loss = pl_tn.inf_dl_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)
        elif self.channel_model == "InF-SH": path_loss = pl_tn.inf_sh_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)
        elif self.channel_model == "InF-DH": path_loss = pl_tn.inf_dh_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)
        elif self.channel_model == "D2D": path_loss = pl_tn.d2d_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)

        elif self.channel_model == "A2G": path_loss = pl_a2g.a2g_path_loss(self.d_2d, self.d_3d, self.h_rx, self.h_tx, self.fc, los)

        return path_loss

    def get_std(self, los, o2i):

        # los = self.compute_los_probability()
        if self.channel_model == "UMi" or self.channel_model == "D2D":  # TODO> Check this assumption for D2D
            if los:
                std = 4
                d_correlation_sf = 10
            elif not o2i:
                std = 7.82
                d_correlation_sf = 13
            else:
                std = 7.82
                d_correlation_sf = 7

        elif self.channel_model == "UMa" or self.channel_model == "A2G":
            if los:
                std = 4
                d_correlation_sf = 37
            elif not o2i:
                std = 6
                d_correlation_sf = 50
            else:
                std = 6
                d_correlation_sf = 7

        elif self.channel_model == "RMa":
            if los:
                c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
                d_bp = 2 * ma.pi * self.h_tx * self.h_tx * (self.fc * 1000000000) / c  # Break point distance (Table 7.4.1.1, Note 5)
                if 10 <= self.d_2d <= d_bp: std = 4
                elif d_bp <= self.d_2d <= 10000: std = 6
                d_correlation_sf = 37
            elif not o2i:
                std = 8
                d_correlation_sf = 120
            else:
                std = 8
                d_correlation_sf = 120

        elif self.channel_model == "InH-Mixed" or self.channel_model == "InH-Open":
            if los:
                std = 3
                d_correlation_sf = 10
            else:
                std = 8.03
                d_correlation_sf = 6

        elif self.channel_model == "InF-HH" or self.channel_model == "InF-SL":
            if los:
                std = 4
                d_correlation_sf = 10
            else:
                std = 5.7
                d_correlation_sf = 6

        elif self.channel_model == "InF-DL":
            if los:
                std = 4
                d_correlation_sf = 10
            else:
                std = 7.2
                d_correlation_sf = 6

        elif self.channel_model == "InF-SH":
            if los:
                std = 4
                d_correlation_sf = 10
            else:
                std = 5.9
                d_correlation_sf = 6

        elif self.channel_model == "InF-DH":
            if los:
                std = 4
                d_correlation_sf = 10
            else:
                std = 4
                d_correlation_sf = 6

        return std, d_correlation_sf

    def compute_shadowing(self):

        if self.t_now == 0:
            o2i = self.compute_o2i_probability()
            o2i_loss = self.compute_o2i_loss(o2i)
            los = self.compute_los_probability(o2i)
            std, d_correlation_sf = self.get_std(los, o2i)
            self.d_correlation_map_rx["t"] = self.t_now
            self.d_correlation_map_rx["x"] = self.rx_coord[0]
            self.d_correlation_map_rx["y"] = self.rx_coord[1]
            if not self.shadowing: self.d_correlation_map_rx["shadowing"] = 0
            else:
                self.d_correlation_map_rx["shadowing"] = normal(loc=0, scale=std, size=1)[0]
            self.d_correlation_map_rx["o2i"] = o2i
            self.d_correlation_map_rx["o2i_loss"] = o2i_loss
            self.d_correlation_map_rx["los"] = los
            self.d_correlation_map_rx["d_correlation_sf"] = d_correlation_sf
            return self.d_correlation_map_rx


            # return shadowing_map_rx_

        # If the shadowing attenuation has been computed at least one time for this user
        # and the distance traveled by the UE is greater than correlation distance

        delta_xy = ma.sqrt((self.rx_coord[0] - self.d_correlation_map_rx["x"])**2 + (self.rx_coord[1] - self.d_correlation_map_rx["y"])**2)
        if delta_xy > self.d_correlation_map_rx["d_correlation_sf"]:
            o2i = self.compute_o2i_probability()
            o2i_loss = self.compute_o2i_loss(o2i)
            los = self.compute_los_probability(o2i)
            std, d_correlation_sf = self.get_std(los, o2i)

            if not self.shadowing:
                self.d_correlation_map_rx["shadowing"] = 0
            else:
                old_shadowing = self.d_correlation_map_rx["shadowing"]  # Get last shadowing attenuation computed
                a = ma.exp(-0.5 * (delta_xy / d_correlation_sf))  # Compute shadowing with a EAW (Exponential Average Window) (step1)
                log_normal_value = normal(loc=0, scale=std, size=1)[0]
                # Compute shadowing with a EAW (Exponential Average Window) (step2)
                shadowing = a * old_shadowing + ma.sqrt(1 - pow(a, 2)) * log_normal_value
                self.d_correlation_map_rx["shadowing"] = shadowing

            self.d_correlation_map_rx["t"] = self.t_now
            self.d_correlation_map_rx["x"] = self.rx_coord[0]
            self.d_correlation_map_rx["y"] = self.rx_coord[1]
            self.d_correlation_map_rx["o2i"] = o2i
            self.d_correlation_map_rx["o2i_loss"] = o2i_loss
            self.d_correlation_map_rx["los"] = los
            self.d_correlation_map_rx["d_correlation_sf"] = d_correlation_sf

        return self.d_correlation_map_rx

    def compute_angular_attenuation(self):
        if self.tx_antenna_mode == "omni":  # omni, three_sectors, four_sectors
            angle_att = 0

        # Report  ITU-R  M.2135-1 (12/2009) for Simplified antenna pattern
        # Table 7.3-1: Radiation power pattern of a single antenna element (7.7.4.1 Exemplary filters/antenna patterns, tr-38-901)
        elif self.tx_antenna_mode == "three_sectors":  # TODO: Check the assumed h_angle_3dB= 70º for a 120º sector
            max_h_angle_att = 30  # is the 3 dB beamwidth (corresponding to h_angle_3dB= 70º)
            h_angle_att = -min(12 * pow(self.h_angle / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation
            v_angle_att = -min(12 * pow((self.v_angle - self.v_tilt) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation
            angle_att = min(-(h_angle_att + v_angle_att), max_h_angle_att)

        elif self.tx_antenna_mode == "four_sectors":  # TODO: Check the assumed h_angle_3dB= 60º for a 90º sector
            max_h_angle_att = 30  # is the 3 dB beamwidth (corresponding to h_angle_3dB= 60º)
            h_angle_att = -min(12 * pow(self.h_angle / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation
            v_angle_att = -min(12 * pow((self.v_angle - self.v_tilt) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation
            angle_att = min(-(h_angle_att + v_angle_att), max_h_angle_att)

        elif self.tx_antenna_mode == "one_sectors_90_degrees":  # TODO: Check the assumed h_angle_3dB= 60º for a 90º sector
            max_h_angle_att = 30  # is the 3 dB beamwidth (corresponding to h_angle_3dB= 60º)
            h_angle_att = -min(12 * pow((45 - self.h_angle) / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation
            v_angle_att = -min(12 * pow( (self.v_angle - self.v_tilt ) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation
            angle_att = min(-(h_angle_att + v_angle_att), max_h_angle_att)

        return angle_att

    def compute_fast_fading_attenuation(self, los, o2i):
        if self.fast_fading:
            if self.fast_fading_model == "jakes":
                fast_fading_att, jakes_map = jakes.jakes_channel(self.ds_angle, self.speed_rx, self.speed_tx, self.t_now, self.n_rb, self.fc, self.jakes_map,
                self.desired_delay_spread, self.atmospheric_absorption, self.channel_model, los, o2i,  self.d_3d, self.tx_coord[2], self.v_angle
                )
                # print(fast_fading_att)
                # print(np.mean(fast_fading_att))
            elif self.fast_fading_model == "tdl":
                fast_fading_att = np.zeros(self.n_rb, dtype=float)
                jakes_map = self.jakes_map
                if los:
                    fast_fading_type = self.fast_fading_los_type
                else:
                    fast_fading_type = self.fast_fading_nlos_type

                for r in range(self.n_rb):
                    fast_fading_att[r] = tdl.tdl_ff_channel_gain(self.channel_model, los, o2i, self.atmospheric_absorption,
                            self.desired_delay_spread, fast_fading_type,
                            self.num_rx_ax, self.num_tx_ax, self.d_2d, self.d_3d, self.bearing_angle,
                            self.down_tilt_angle, self.h_angle, self.v_angle, self.ds_angle, self.fc, self.speed_rx, self.speed_tx,
                            self.rx_coord, self.tx_coord, self.t_now, self.t_old)
                # print(fast_fading_att)
                # print(np.mean(fast_fading_att))
            elif self.fast_fading_model == "cdl":
                fast_fading_att = np.zeros(self.n_rb, dtype=float)
                jakes_map = self.jakes_map
                if los:
                    fast_fading_type = self.fast_fading_los_type
                else:
                    fast_fading_type = self.fast_fading_nlos_type
                for r in range(self.n_rb):

                    fast_fading_att[r] = cdl.cdl_ff_channel_gain(self.channel_model, los, o2i, self.atmospheric_absorption,
                            self.desired_delay_spread, fast_fading_type, self.rng,
                            self.num_rx_ax, self.num_tx_ax, self.tx_antenna_mode, self.rx_antenna_mode,
                            self.ax_panel_polarization, self.delta_h, self.d_2d, self.d_3d, self.bearing_angle,
                            self.down_tilt_angle, self.h_angle, self.v_angle, self.fc, self.speed_rx, self.rx_coord, self.tx_coord,
                            self.t_now, self.t_old)
            # print(fast_fading_att)
            # print(np.mean(fast_fading_att))
        else:
            fast_fading_att = np.zeros(self.n_rb, dtype=float)
            jakes_map = self.jakes_map

        return fast_fading_att, jakes_map

    def get_sinr(self, d_correlation_map_rx, path_loss, hb_attenuation, angle_att, fast_fading_att):

        o2i_loss = d_correlation_map_rx["o2i_loss"]
        shadowing = d_correlation_map_rx["shadowing"]
        # d_correlation_map_rx = self.compute_shadowing()
        # path_loss_ = self.compute_path_loss(d_correlation_map_rx["los"])
        # hb_attenuation_, hb_map_rx = self.get_hb_attenuation()

        # angle_att_ = self.compute_angular_attenuation()


        # print("d_correlation_map_rx", d_correlation_map_rx)
        # print("path_loss_", path_loss_)
        # print("hb_attenuation_", hb_attenuation_)
        # print("o2i_loss_", o2i_loss_)
        # print("angle_att_", angle_att_)
        # print("shadowing", d_correlation_map_rx["shadowing"])

        rx_power = self.tx_power + self.antenna_gain_tx + self.antenna_gain_rx \
                   - (path_loss + shadowing + angle_att + hb_attenuation + o2i_loss + self.cable_loss_tx)

        thermal_noise_bw = jakes.linear_to_db(jakes.db_to_linear(self.thermal_noise) * self.n_rb * self.bw_rb)  # TODO: Check
        noise_bw = thermal_noise_bw + self.rx_noise_figure

        thermal_noise_rb = jakes.linear_to_db(jakes.db_to_linear(self.thermal_noise) * self.bw_rb)  # TODO: Check
        noise_rb = thermal_noise_rb + self.rx_noise_figure                                    # TODO: Check

        if self.fast_fading:
            # fast_fading_att, jakes_map_ = self.compute_fast_fading_attenuation()
            sinr_vector_n_rb = np.zeros(self.n_rb, dtype=float)
            rx_power_vector_n_rb = np.zeros(self.n_rb, dtype=float)
            sinr = 0
            # print("rx_power", rx_power)
            for i in range(self.n_rb):
                rx_power_vector_n_rb[i] = rx_power + fast_fading_att[i]
                sinr_vector_n_rb[i] = rx_power_vector_n_rb[i] - noise_bw
                sinr = sinr + sinr_vector_n_rb[i]
            # print("rx_power_vector_n_rb", rx_power_vector_n_rb)
            # print("mean rx_power_vector_n_rb", np.mean(rx_power_vector_n_rb))
            sinr = sinr/self.n_rb

        else:
            sinr = rx_power - noise_bw

        return round(sinr, 2)


def get_ch_tr_38_901(channel_model, tx_antenna_mode, shadowing, dynamic_los, dynamic_hb, outdoor_to_indoor, inside_what_o2i, penetration_loss_model,
                 d_2d, d_3d, h_rx, h_tx, h_ceiling, block_density, fc, d_correlation_map_rx, t_now, t_old,
                 speed_rx, speed_tx, rx_coord, tx_coord, h_angle, v_angle, ds_angle, v_tilt, n_rb, jakes_map, fast_fading_model, hb_map_rx,
                 cable_loss_tx, thermal_noise, bw_rb, rx_noise_figure, fast_fading, tx_power, antenna_gain_tx, antenna_gain_rx,
                 atmospheric_absorption, desired_delay_spread, fast_fading_los_type, fast_fading_nlos_type, num_rx_ax, num_tx_ax,
                 rng, rx_antenna_mode, ax_panel_polarization):

        ch_tr_38_901 = Ch_tr_138_901(channel_model, tx_antenna_mode, shadowing, dynamic_los, dynamic_hb, outdoor_to_indoor, inside_what_o2i, penetration_loss_model,
                 d_2d, d_3d, h_rx, h_tx, h_ceiling, block_density, fc, d_correlation_map_rx, t_now, t_old,
                 speed_rx, speed_tx, rx_coord, tx_coord, h_angle, v_angle, ds_angle, v_tilt, n_rb, jakes_map, fast_fading_model, hb_map_rx,
                 cable_loss_tx, thermal_noise, bw_rb, rx_noise_figure, fast_fading, tx_power, antenna_gain_tx, antenna_gain_rx,
                 atmospheric_absorption, desired_delay_spread, fast_fading_los_type, fast_fading_nlos_type, num_rx_ax, num_tx_ax,
                 rng, rx_antenna_mode, ax_panel_polarization)

        d_correlation_map_rx = ch_tr_38_901.compute_shadowing()
        path_loss = ch_tr_38_901.compute_path_loss(d_correlation_map_rx["los"])
        hb_attenuation, hb_map_rx = ch_tr_38_901.get_hb_attenuation()
        angle_att = ch_tr_38_901.compute_angular_attenuation()
        fast_fading_att, jakes_map = ch_tr_38_901.compute_fast_fading_attenuation(d_correlation_map_rx["los"], d_correlation_map_rx["o2i"])
        sinr = ch_tr_38_901.get_sinr(d_correlation_map_rx, path_loss, hb_attenuation, angle_att, fast_fading_att)
        ch_outcomes_rx = {"t": d_correlation_map_rx["t"],
                          "x": d_correlation_map_rx["x"],
                          "y": d_correlation_map_rx["y"],
                          "o2i": d_correlation_map_rx["o2i"],
                          "los": d_correlation_map_rx["los"],
                          "o2i_loss": d_correlation_map_rx["o2i_loss"],
                          "shadowing": d_correlation_map_rx["shadowing"],
                          "path_loss": path_loss,
                          "angle_att": angle_att,
                          "hb_attenuation": hb_attenuation,
                          "fast_fading_att": round(np.mean(fast_fading_att), 2),
                          "sinr": sinr
                          }

        return ch_outcomes_rx, d_correlation_map_rx, hb_map_rx, jakes_map

    ########################## testing ####################################################################



# import numpy as np
# import matplotlib.pyplot as plt
#
# # Parameters for the normal distribution
# mu = 0  # Mean of the normal distribution (dB)
# sigma = 8  # Standard deviation of the normal distribution (dB), a typical value for shadow fading
#
# # Generate random shadow fading values
# num_samples = 1000
# shadow_fading = normal(mu, sigma, num_samples)
#
#
# # Plot the histogram of the shadow fading values
# plt.figure(figsize=(10, 6))
# plt.hist(shadow_fading, bins=30, density=True, alpha=0.6, color='b')
# plt.title('Shadow Fading Loss Distribution')
# plt.xlabel('Shadow Fading Loss (dB)')
# plt.ylabel('Density')
# plt.grid(True)
# plt.show()
#
# # Print basic statistics
# mean = np.mean(shadow_fading)
# std_dev = np.std(shadow_fading)
# print(f"Generated Shadow Fading Loss - Mean: {mean:.2f} dB, Standard Deviation: {std_dev:.2f} dB")