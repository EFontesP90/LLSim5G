import math as ma

import numpy as np
from numpy import random
from numpy.random import normal
from scipy.stats import norm
from scipy.special import jn

from channel_models import path_loss_models_tr_38_811 as pl_ntn
import channel_models.spectroscopic_data as sd
import channel_models.ff_model_jakes as jakes
from channel_models.ff_models_tr_38_901and811 import tdl_mdels as tdl
from channel_models.ff_models_tr_38_901and811 import cdl_models as cdl

class Ch_tr_38_811(object):
    """
    14/05/2024
    Channel implementation according to 3gpp tr-38-811.

    Required attributes:
    ():

    """

    def __init__(self, t_now, t_old, speed_rx, speed_tx, ds_angle,  rx_coord, tx_coord, channel_model, rx_scenario, tx_antenna_mode, dynamic_los,
                 elevation_angle, d_sat, h_sat, fc, f_band_rx, outdoor_to_indoor, inside_what, penetration_loss_model,
                 d_correlation_map_rx, shadowing, n_rb, jakes_map, fast_fading_model,
                 cable_loss_tx, thermal_noise, bw_rb, rx_noise_figure, fast_fading, tx_power, antenna_gain_tx,
                 antenna_gain_rx, atmospheric_absorption, desired_delay_spread, fast_fading_los_type, fast_fading_nlos_type, num_rx_ax, num_tx_ax,
                 rng, rx_antenna_mode, ax_panel_polarization):

        if elevation_angle == 0:
            elevation_angle = 0.01

        self.t_now = t_now  # current time step
        self.t_old = t_old  # previous time step
        self.speed_rx = speed_rx
        self.speed_tx = speed_tx
        self.ds_angle = ds_angle
        self.rx_coord = rx_coord
        self.tx_coord = tx_coord

        self.channel_model = channel_model  # "Geo", "Non-Geo", "HAPS". String with the selected ntn link channel to be modeled from the tr.38.8111: Sat.
        self.rx_scenario = rx_scenario  # "open", "rural", "suburban", "urban" and "dense urban"
        self.tx_antenna_mode = tx_antenna_mode  # string with the selected tx antenna mode: omni, Sat_ax.

        self.shadowing = shadowing
        self.dynamic_los = dynamic_los  # True or False variable for enabling or not the dynamic line-of-sight (los) mode of the end devices.
        self.outdoor_to_indoor = outdoor_to_indoor  # True or False values for enabling if the ue is modeled as inside a building/car or not.
        self.inside_what = inside_what  # String (building, car, dynamic) for defining if a user is inside a car a building or can change dynamically.
        self.penetration_loss_model = penetration_loss_model  # ("high-loss",  "low-loss") equivalent to "Traditional", "Thermally-efficient"

        self.d_sat = d_sat  # For a ground terminal, the distance d (a.k.a. slant range)
        self.h_sat = h_sat  # satellite/HAPS altitude
        self.elevation_angle = elevation_angle  # string with the selected tx antenna mode: omni, three_sectors, four_sectors.

        self.fc = fc  # Simulation Frequency in GHz
        self.f_band_rx = f_band_rx  # "S-band", "Ka-band"

        self.n_rb = n_rb  # Nuber of rb assigned to the ue
        self.fast_fading_model = fast_fading_model  # "jakes"
        self.d_correlation_map_rx = d_correlation_map_rx  # Array of 1*2: [t, x, y, std], For tracking the shadowing of the user over time
        self.jakes_map = jakes_map  # array of the resulting jackFadingMap of the K users respect to the BS and each proximity user in D2D comm, fadingTaps X [delay amp, AoA]

        self.cable_loss_tx = cable_loss_tx
        self.thermal_noise = thermal_noise
        self.bw_rb = bw_rb
        self.rx_noise_figure = rx_noise_figure

        self.fast_fading = fast_fading
        self.tx_power = tx_power
        self.antenna_gain_tx = antenna_gain_tx
        self.antenna_gain_rx = antenna_gain_rx

        self.d_correlation_map_rx = d_correlation_map_rx

        self.atmospheric_absorption = atmospheric_absorption
        self.desired_delay_spread = desired_delay_spread
        self.fast_fading_los_type = fast_fading_los_type
        self.fast_fading_nlos_type = fast_fading_nlos_type
        self.num_rx_ax = num_rx_ax
        self.num_tx_ax = num_tx_ax
        self.bearing_angle = 0
        self.down_tilt_angle = 0

        self.rng = rng
        self.rx_antenna_mode = rx_antenna_mode
        self.ax_panel_polarization = ax_panel_polarization
        self.delta_h = self.h_sat



    def compute_o2i_probability(self):
        if not self.outdoor_to_indoor:
            o2i = False  # 0 dB
            return o2i
        elif self.channel_model == "HAPS":
            o2i_p = random.uniform(0.0, 1.0)
            if 0.2 < o2i_p:  # TODO, I am following the same approach as for UMa, UMi in Table 7.2-1: tr.38.901
                o2i = True
            else:
                o2i = False
        else:
            o2i = False

        return o2i

    def compute_los_probability(self, o2i):  # ETSI-TR.38.811 (2020-11), page 48. 6.6.1 LOS probability, Table 6.6.1-1 LOS probability

        if o2i:
            los = False
            return los

        if self.dynamic_los == False:
            los = True
        else:
            if self.rx_scenario == "dense urban":
                if self.elevation_angle <= 10:
                    p = 28.2
                elif self.elevation_angle <= 20:
                    p = 33.1
                elif self.elevation_angle <= 30:
                    p = 39.8
                elif self.elevation_angle <= 40:
                    p = 46.8
                elif self.elevation_angle <= 50:
                    p = 53.7
                elif self.elevation_angle <= 60:
                    p = 61.2
                elif self.elevation_angle <= 70:
                    p = 73.8
                elif self.elevation_angle <= 80:
                    p = 82.0
                elif self.elevation_angle <= 90:
                    p = 98.1
            elif self.rx_scenario == "urban":
                if self.elevation_angle <= 10:
                    p = 24.6
                elif self.elevation_angle <= 20:
                    p = 38.6
                elif self.elevation_angle <= 30:
                    p = 49.3
                elif self.elevation_angle <= 40:
                    p = 61.3
                elif self.elevation_angle <= 50:
                    p = 72.6
                elif self.elevation_angle <= 60:
                    p = 80.5
                elif self.elevation_angle <= 70:
                    p = 91.9
                elif self.elevation_angle <= 80:
                    p = 96.8
                elif self.elevation_angle <= 90:
                    p = 99.2
            elif self.rx_scenario == "suburban" or self.rx_scenario == "rural":
                if self.elevation_angle <= 10:
                    p = 78.2
                elif self.elevation_angle <= 20:
                    p = 86.9
                elif self.elevation_angle <= 30:
                    p = 91.9
                elif self.elevation_angle <= 40:
                    p = 92.9
                elif self.elevation_angle <= 50:
                    p = 93.5
                elif self.elevation_angle <= 60:
                    p = 94.0
                elif self.elevation_angle <= 70:
                    p = 94.9
                elif self.elevation_angle <= 80:
                    p = 95.2
                elif self.elevation_angle <= 90:
                    p = 99.8
            elif self.rx_scenario == "open":
                p = 1

            r = random.uniform(0.0, 1.0)
            if r <= p:
                los = True
            else:
                los = False

        return los

    def compute_path_loss(self):  # valid for Sat
        path_loss = pl_ntn.ntn_sat_path_loss(self.d_sat, self.h_sat, self.fc)

        return path_loss

    def get_sf_std_and_clutter(self, los):

        if los:
            cl = 0
            if self.f_band_rx == "S-band":
                if self.rx_scenario == "dense urban":
                    if self.elevation_angle <= 10:
                        std = 3.5
                    elif self.elevation_angle <= 20:
                        std = 3.4
                    elif self.elevation_angle <= 30:
                        std = 2.9
                    elif self.elevation_angle <= 40:
                        std = 3.0
                    elif self.elevation_angle <= 50:
                        std = 3.1
                    elif self.elevation_angle <= 60:
                        std = 2.7
                    elif self.elevation_angle <= 70:
                        std = 2.5
                    elif self.elevation_angle <= 80:
                        std = 2.3
                    elif self.elevation_angle <= 90:
                        std = 1.2
                elif self.rx_scenario == "urban":
                    std = 4

                elif self.rx_scenario == "suburban" or self.rx_scenario == "rural":
                    if self.elevation_angle <= 10:
                        std = 1.79
                    elif self.elevation_angle <= 20:
                        std = 1.14
                    elif self.elevation_angle <= 30:
                        std = 1.14
                    elif self.elevation_angle <= 40:
                        std = 0.92
                    elif self.elevation_angle <= 50:
                        std = 1.42
                    elif self.elevation_angle <= 60:
                        std = 1.56
                    elif self.elevation_angle <= 70:
                        std = 0.85
                    elif self.elevation_angle <= 80:
                        std = 0.72
                    elif self.elevation_angle <= 90:
                        std = 0.72
                elif self.rx_scenario == "open":
                    std = 0

            elif self.f_band_rx == "Ka-band":
                if self.rx_scenario == "dense urban":
                    if self.elevation_angle <= 10:
                        std = 2.9
                    elif self.elevation_angle <= 20:
                        std = 2.4
                    elif self.elevation_angle <= 30:
                        std = 2.7
                    elif self.elevation_angle <= 40:
                        std = 2.4
                    elif self.elevation_angle <= 50:
                        std = 2.4
                    elif self.elevation_angle <= 60:
                        std = 2.7
                    elif self.elevation_angle <= 70:
                        std = 2.6
                    elif self.elevation_angle <= 80:
                        std = 2.8
                    elif self.elevation_angle <= 90:
                        std = 0.6
                elif self.rx_scenario == "urban":
                    std = 4

                elif self.rx_scenario == "suburban" or self.rx_scenario == "rural":
                    if self.elevation_angle <= 10:
                        std = 1.9
                    elif self.elevation_angle <= 20:
                        std = 1.6
                    elif self.elevation_angle <= 30:
                        std = 1.9
                    elif self.elevation_angle <= 40:
                        std = 2.3
                    elif self.elevation_angle <= 50:
                        std = 2.7
                    elif self.elevation_angle <= 60:
                        std = 3.1
                    elif self.elevation_angle <= 70:
                        std = 3.0
                    elif self.elevation_angle <= 80:
                        std = 3.6
                    elif self.elevation_angle <= 90:
                        std = 0.4
                elif self.rx_scenario == "open":
                    std = 0

        else:
            if self.f_band_rx == "S-band":
                if self.rx_scenario == "dense urban":
                    if self.elevation_angle <= 10:
                        std = 3.5
                        cl = 34.3
                    elif self.elevation_angle <= 20:
                        std = 3.4
                        cl = 30.9
                    elif self.elevation_angle <= 30:
                        std = 2.9
                        cl = 29.0
                    elif self.elevation_angle <= 40:
                        std = 3.0
                        cl = 27.7
                    elif self.elevation_angle <= 50:
                        std = 3.1
                        cl = 26.8
                    elif self.elevation_angle <= 60:
                        std = 2.7
                        cl = 26.2
                    elif self.elevation_angle <= 70:
                        std = 2.5
                        cl = 25.8
                    elif self.elevation_angle <= 80:
                        std = 2.3
                        cl = 25.5
                    elif self.elevation_angle <= 90:
                        std = 1.2
                        cl = 25.5

                elif self.rx_scenario == "urban":
                    std = 6
                    if self.elevation_angle <= 10:
                        cl = 34.3
                    elif self.elevation_angle <= 20:
                        cl = 30.9
                    elif self.elevation_angle <= 30:
                        cl = 29.0
                    elif self.elevation_angle <= 40:
                        cl = 27.7
                    elif self.elevation_angle <= 50:
                        cl = 26.8
                    elif self.elevation_angle <= 60:
                        cl = 26.2
                    elif self.elevation_angle <= 70:
                        cl = 25.8
                    elif self.elevation_angle <= 80:
                        cl = 25.5
                    elif self.elevation_angle <= 90:
                        cl = 25.5

                elif self.rx_scenario == "suburban" or self.rx_scenario == "rural":
                    if self.elevation_angle <= 10:
                        std = 1.79
                        cl = 19.52
                    elif self.elevation_angle <= 20:
                        std = 1.14
                        cl = 18.17
                    elif self.elevation_angle <= 30:
                        std = 1.14
                        cl = 18.42
                    elif self.elevation_angle <= 40:
                        std = 0.92
                        cl = 18.28
                    elif self.elevation_angle <= 50:
                        std = 1.42
                        cl = 18.63
                    elif self.elevation_angle <= 60:
                        std = 1.56
                        cl = 17.68
                    elif self.elevation_angle <= 70:
                        std = 0.85
                        cl = 16.50
                    elif self.elevation_angle <= 80:
                        std = 0.72
                        cl = 16.30
                    elif self.elevation_angle <= 90:
                        std = 0.72
                        cl = 16.30
                elif self.rx_scenario == "open":
                    std = 0
                    cl = 0

            elif self.f_band_rx == "Ka-band":
                if self.rx_scenario == "dense urban":
                    if self.elevation_angle <= 10:
                        std = 2.9
                        cl = 44.3
                    elif self.elevation_angle <= 20:
                        std = 2.4
                        cl = 39.9
                    elif self.elevation_angle <= 30:
                        std = 2.7
                        cl = 37.5
                    elif self.elevation_angle <= 40:
                        std = 2.4
                        cl = 35.8
                    elif self.elevation_angle <= 50:
                        std = 2.4
                        cl = 34.6
                    elif self.elevation_angle <= 60:
                        std = 2.7
                        cl = 33.8
                    elif self.elevation_angle <= 70:
                        std = 2.6
                        cl = 33.3
                    elif self.elevation_angle <= 80:
                        std = 2.8
                        cl = 33.0
                    elif self.elevation_angle <= 90:
                        std = 0.6
                        cl = 32.9
                elif self.rx_scenario == "urban":
                    std = 6
                    if self.elevation_angle <= 10:
                        cl = 44.3
                    elif self.elevation_angle <= 20:
                        cl = 39.9
                    elif self.elevation_angle <= 30:
                        cl = 37.5
                    elif self.elevation_angle <= 40:
                        cl = 35.8
                    elif self.elevation_angle <= 50:
                        cl = 34.6
                    elif self.elevation_angle <= 60:
                        cl = 33.8
                    elif self.elevation_angle <= 70:
                        cl = 33.3
                    elif self.elevation_angle <= 80:
                        cl = 33.0
                    elif self.elevation_angle <= 90:
                        cl = 32.9

                elif self.rx_scenario == "suburban" or self.rx_scenario == "rural":
                    if self.elevation_angle <= 10:
                        std = 1.9
                        cl = 29.5
                    elif self.elevation_angle <= 20:
                        std = 1.6
                        cl = 24.6
                    elif self.elevation_angle <= 30:
                        std = 1.9
                        cl = 21.9
                    elif self.elevation_angle <= 40:
                        std = 2.3
                        cl = 20.0
                    elif self.elevation_angle <= 50:
                        std = 2.7
                        cl = 18.7
                    elif self.elevation_angle <= 60:
                        std = 3.1
                        cl = 17.8
                    elif self.elevation_angle <= 70:
                        std = 3.0
                        cl = 17.2
                    elif self.elevation_angle <= 80:
                        std = 3.6
                        cl = 16.9
                    elif self.elevation_angle <= 90:
                        std = 0.4
                        cl = 16.8
                elif self.rx_scenario == "open":
                    std = 0
                    cl = 0

        # sf = normal(loc=0, scale=std, size=1)[0]
        return std, cl

    def get_d_correlation(self, los):
        if los:
            d_correlation_sf = 37
        elif self.rx_scenario == "rural":
            d_correlation_sf = 120
        else:
            d_correlation_sf = 50
        return d_correlation_sf

    def compute_o2i_loss(self, o2i):  # TODO, how to handle if the user are inside the car, is not standardized. P = probability that loss is not exceeded (0.0 < P < 1.0) ) in 6.6.3 O2I penetration loss
        l_bel_p = 0
        o2i_loss = 0
        if not o2i:
            l_bel_p = 0  # 0 dB
            o2i_loss = l_bel_p
            return o2i_loss
        else:
            if self.inside_what == "building":

                p = random.uniform(0.0, 1.0)  # TODO, I am not clear if this is the propper why to handle this variable.

                if self.penetration_loss_model == "high-loss":
                    r = 12.64
                    s = 3.72
                    t = 0.96
                    w = 9.1
                    u = 9.6
                    v = 2.0
                    y = 4.5
                    z = -2.0
                    x = -3.0
                elif self.penetration_loss_model == "low-loss":
                    r = 12.64
                    s = 3.72
                    t = 0.96
                    w = 9.1
                    u = 9.6
                    v = 2.0
                    y = 4.5
                    z = -2.0
                    x = -2.9

                l_h = r + s * ma.log(self.fc, 10) + t * (ma.log(self.fc, 10)) ** 2
                l_e = 0.212 * abs(self.elevation_angle)

                u_1 = l_h + l_e
                u_2 = w + x * ma.log(self.fc, 10)
                o_1 = u + v * ma.log(self.fc, 10)
                o_2 = y + z * ma.log(self.fc, 10)

                a_p = norm.ppf(p, loc=0, scale=1) * o_1 + u_1
                b_p = norm.ppf(p, loc=0, scale=1) * o_2 + u_2
                c = -3.0

                l_bel_p = 10 * ma.log(10 ** (0.1 * a_p) + 10 ** (0.1 * b_p) + 10 ** (0.1 * c), 10)

            elif self.inside_what == "car":
                # NOTE: 7.4.3.2 O2I car penetration loss
                # The O2I car penetration loss models are applicable for at least 0.6-60 GHz.
                mu = 9.0  # 20 for metalized car windows
                sigma_p = 5.0
                l_bel_p = normal(loc=mu, scale=sigma_p, size=1)[0]

            o2i_loss = l_bel_p
        return o2i_loss

    def compute_shadowing(self):

        if self.t_now == 0:
            o2i = self.compute_o2i_probability()
            o2i_loss = self.compute_o2i_loss(o2i)
            los = self.compute_los_probability(o2i)
            std, cl = self.get_sf_std_and_clutter(los)
            d_correlation_sf = self.get_d_correlation(los)
            self.d_correlation_map_rx["t"] = self.t_now
            self.d_correlation_map_rx["x"] = self.rx_coord[0]
            self.d_correlation_map_rx["y"] = self.rx_coord[1]
            if not self.shadowing:
                self.d_correlation_map_rx["shadowing"] = 0
            else:
                self.d_correlation_map_rx["shadowing"] = normal(loc=0, scale=std, size=1)[0]
            self.d_correlation_map_rx["cl"] = cl
            self.d_correlation_map_rx["o2i"] = o2i
            self.d_correlation_map_rx["o2i_loss"] = o2i_loss
            self.d_correlation_map_rx["los"] = los
            self.d_correlation_map_rx["d_correlation_sf"] = d_correlation_sf
            return self.d_correlation_map_rx

        delta_xy = ma.sqrt((self.rx_coord[0] - self.d_correlation_map_rx["x"]) ** 2 + (
                self.rx_coord[1] - self.d_correlation_map_rx["y"]) ** 2)
        if delta_xy > self.d_correlation_map_rx["d_correlation_sf"]:
            o2i = self.compute_o2i_probability()
            o2i_loss = self.compute_o2i_loss(o2i)
            los = self.compute_los_probability(o2i)
            std, cl = self.get_sf_std_and_clutter(los)
            d_correlation_sf = self.get_d_correlation(los)

            if not self.shadowing:
                self.d_correlation_map_rx["shadowing"] = 0
            else:
                old_shadowing = self.d_correlation_map_rx["shadowing"]  # Get last shadowing attenuation computed
                a = ma.exp(-0.5 * (
                        delta_xy / d_correlation_sf))  # Compute shadowing with a EAW (Exponential Average Window) (step1)
                log_normal_value = normal(loc=0, scale=std, size=1)[0]
                # Compute shadowing with a EAW (Exponential Average Window) (step2)
                shadowing = a * old_shadowing + ma.sqrt(1 - pow(a, 2)) * log_normal_value
                self.d_correlation_map_rx["shadowing"] = shadowing

            self.d_correlation_map_rx["cl"] = cl
            self.d_correlation_map_rx["t"] = self.t_now
            self.d_correlation_map_rx["x"] = self.rx_coord[0]
            self.d_correlation_map_rx["y"] = self.rx_coord[1]
            self.d_correlation_map_rx["o2i"] = o2i
            self.d_correlation_map_rx["o2i_loss"] = o2i_loss
            self.d_correlation_map_rx["los"] = los
            self.d_correlation_map_rx["d_correlation_sf"] = d_correlation_sf

        return self.d_correlation_map_rx

    def compute_angular_attenuation(self):
        if self.tx_antenna_mode == "omni":  # omni, Sat_ax,
            angle_att = 0
        # 6.4 Antenna modelling
        # 6.4.1 HAPS/Satellite antenna , tr-38-811
        else:
            if self.elevation_angle == 90:
                angle_att = 1
            else:
                c = 300000000  # 3.0×108 m/s is the propagation velocity in free space
                k = 2 * ma.pi * self.fc / c
                a = 10 * c / self.fc  # The normalized gain pattern for a = 10 c/f (aperture radius of 10 wavelengths) is shown in Figure 6.4.1-1.

                angle = ma.radians(self.elevation_angle-90)
                x = k * a * ma.sin(angle)
                J1_x = jn(1, x)
                angle_att = jakes.linear_to_db(4 * abs(J1_x / (k * a * ma.sin(angle))) ** 2)
        return abs(angle_att)

    def get_atmospheric_absorption(self):

        if self.fc < 10 and self.elevation_angle > 10:
            atmos_att = 0
            att_zenith_total = 0
        elif self.fc < 1 and self.elevation_angle <= 10:
            atmos_att = 0
            att_zenith_total = 0
        else:
            T = 288.15  # K
            p = 1013.25  # hPa
            pp = 7.5  # g/m**3
            e = 9.98  # hPa (pp*T/216.5)
            r_p = (p + e) / 1013.25

            t_1 = (4.64 / (1 + 0.066 * r_p ** (-2.3))) * ma.exp(
                -((self.fc - 59.7) / (2.87 + 12.4 * ma.exp(-7.9 * r_p))) ** 2)
            t_2 = (0.14 * ma.exp(2.12 * r_p)) / ((self.fc - 118.75) ** 2 + 0.031 * ma.exp(2.2 * r_p))
            t_3 = (0.0114 / (1 + 0.14 * r_p ** (-2.6))) * self.fc * (
                    (-0.0247 + 0.0001 * self.fc + 1.61 * 1e-6 * (self.fc ** 2))
                    / (1 - 0.0169 * self.fc + 4.1 * 1e-5 * (self.fc ** 2) + 3.2 * 1e-7 * (self.fc ** 3)))

            h_o = (6.1 / (1 + 0.17 * r_p ** (-1.1))) * (1 + t_1 + t_2 + t_3)

            o_w = 1.013 / (1 + ma.exp(-8.6 * (r_p - 0.57)))
            h_w = 1.66 * (1 + (1.39 * o_w / ((self.fc - 22.235) ** 2 + 2.56 * o_w))
                          + (3.37 * o_w / ((self.fc - 183.31) ** 2 + 4.69 * o_w))
                          + (1.58 * o_w / ((self.fc - 325.1) ** 2 + 2.89 * o_w)))

            length_o = int(len(sd.sd_oxygen_att) / 7)
            length_w = int(len(sd.sd_water_att) / 7)

            s_f_i_o = 0
            s_f_i_w = 0

            for i in range(length_o):
                ff_i = sd.sd_oxygen_att[i * 7]
                a_1 = sd.sd_oxygen_att[i * 7 + 1]
                a_2 = sd.sd_oxygen_att[i * 7 + 2]
                a_3 = sd.sd_oxygen_att[i * 7 + 3]
                a_4 = sd.sd_oxygen_att[i * 7 + 4]
                a_5 = sd.sd_oxygen_att[i * 7 + 5]
                a_6 = sd.sd_oxygen_att[i * 7 + 6]

                s_i_o = a_1 * 1e-7 * p * ((300 / T) ** 3) * ma.exp(a_2 * (1 - (300 / T)))  # (3)
                d_f_o = a_3 * 1e-4 * (p * (300 / T) ** (0.8 - a_4) + 1.1 * e * (300 / T))  # (6.a)
                delta_o = (a_5 + a_6 * (300 / T)) * 1e-4 * (p + e) * (300 / T) ** 0.8  # (7)
                f_i_o = (self.fc / ff_i) * (
                        ((d_f_o - delta_o * (ff_i - self.fc)) / ((ff_i - self.fc) ** 2 + d_f_o ** 2)) + (
                        (d_f_o - delta_o * (ff_i + self.fc)) / ((ff_i + self.fc) ** 2 + d_f_o ** 2)))  # (5)
                s_f_i_o += s_i_o * f_i_o

            for i in range(length_w):
                ff_i = sd.sd_water_att[i * 7]
                b_1 = sd.sd_water_att[i * 7 + 1]
                b_2 = sd.sd_water_att[i * 7 + 2]
                b_3 = sd.sd_water_att[i * 7 + 3]
                b_4 = sd.sd_water_att[i * 7 + 4]
                b_5 = sd.sd_water_att[i * 7 + 5]
                b_6 = sd.sd_water_att[i * 7 + 6]

                s_i_w = b_1 * 1e-1 * e * ((300 / T) ** 3.5) * ma.exp(b_2 * (1 - (300 / T)))  # (3)
                delta_w = 0  # (7)
                d_f_w = b_3 * 1e-4 * (p * (300 / T) ** b_4 + b_5 * e * (300 / T) ** b_6)  # (6.a)
                f_i_w = (self.fc / ff_i) * (
                        ((d_f_w - delta_w * (ff_i - self.fc)) / ((ff_i - self.fc) ** 2 + d_f_w ** 2)) + (
                        (d_f_w - delta_w * (ff_i + self.fc)) / ((ff_i + self.fc) ** 2 + d_f_w ** 2)))  # (5)
                s_f_i_w += s_i_w * f_i_w

            d = 5.6 * 1e-4 * (p + e) * (300 / T) ** 0.8  # (9)

            n_d_f = self.fc * p * (300 / T) ** 2 * (
                    (6.14 * 1e-5) / (d * (1 + (self.fc / d) ** 2)) + (1.4 * 1e-12 * p * (300 / T) ** 1.5) / (
                    1 + 1.9 * 1e-5 * self.fc ** 1.5))  # (8)

            y_o = 0.1820 * self.fc * (s_f_i_o + n_d_f)
            # y_o = 0
            y_w = 0.1820 * self.fc * s_f_i_w
            # y_w = 0

            if 50 < self.fc < 70: att_zenith_total = y_w * h_w
            else: att_zenith_total = y_o * h_o + y_w * h_w
            # att_zenith_total_d = (y_o + y_w) * d_sat
            angle = ma.radians(self.elevation_angle)
            atmos_att = att_zenith_total / ma.sin(angle)

        return atmos_att

    def compute_fast_fading_attenuation(self, los, o2i):
        if self.fast_fading:
            if self.fast_fading_model == "jakes":
                fast_fading_att, jakes_map = jakes.jakes_channel(self.ds_angle, self.speed_rx, self.speed_tx, self.t_now, self.n_rb, self.fc, self.jakes_map,
                self.desired_delay_spread, False, self.channel_model, los, o2i,  self.d_sat, self.h_sat, self.elevation_angle
                )
            elif self.fast_fading_model == "tdl":
                fast_fading_att = np.zeros(self.n_rb, dtype=float)
                jakes_map = self.jakes_map
                if los:
                    fast_fading_type = self.fast_fading_los_type
                else:
                    fast_fading_type = self.fast_fading_nlos_type

                for r in range(self.n_rb):
                    fast_fading_att[r] = tdl.tdl_ff_channel_gain(self.channel_model, los, o2i, False,
                            self.desired_delay_spread, fast_fading_type,
                            self.num_rx_ax, self.num_tx_ax, None, self.d_sat, self.bearing_angle,
                            self.down_tilt_angle, 0, self.elevation_angle, self.ds_angle, self.fc, self.speed_rx, self.speed_tx,
                            self.rx_coord, self.tx_coord, self.t_now, self.t_old)
            elif self.fast_fading_model == "cdl":
                fast_fading_att = np.zeros(self.n_rb, dtype=float)
                jakes_map = self.jakes_map
                if los:
                    fast_fading_type = self.fast_fading_los_type
                else:
                    fast_fading_type = self.fast_fading_nlos_type
                for r in range(self.n_rb):

                    fast_fading_att[r] = cdl.cdl_ff_channel_gain(self.channel_model, los, o2i, False,
                            self.desired_delay_spread, fast_fading_type, self.rng,
                            self.num_rx_ax, self.num_tx_ax, self.tx_antenna_mode, self.rx_antenna_mode,
                            self.ax_panel_polarization, self.delta_h, None, self.d_sat, self.bearing_angle,
                            self.down_tilt_angle, 0, self.elevation_angle, self.fc, self.speed_rx, self.rx_coord, self.tx_coord,
                            self.t_now, self.t_old)
        else:
            fast_fading_att = np.zeros(self.n_rb, dtype=float)
            jakes_map = self.jakes_map

        return fast_fading_att, jakes_map

    def get_sinr(self, d_correlation_map_rx, path_loss, angle_att, atmos_att, fast_fading_att):

        polarization_mismatch = 3  #we assume that the User Equipment has an omni-directional antenna of linear polarization,
                                   # while the antenna on board space-borne or airborne platforms features typically employs circular
                                   #polarization.

        o2i_loss = d_correlation_map_rx["o2i_loss"]
        shadowing = d_correlation_map_rx["shadowing"]

        rx_power = self.tx_power + self.antenna_gain_tx + self.antenna_gain_rx \
                   - (path_loss + shadowing + angle_att + atmos_att + o2i_loss + self.cable_loss_tx + polarization_mismatch)

        thermal_noise_bw = jakes.linear_to_db(jakes.db_to_linear(self.thermal_noise) * self.n_rb * self.bw_rb)  # TODO: Check
        noise_bw = thermal_noise_bw + self.rx_noise_figure

        thermal_noise_rb = jakes.linear_to_db(jakes.db_to_linear(self.thermal_noise) * self.bw_rb)  # TODO: Check
        noise_rb = thermal_noise_rb + self.rx_noise_figure                                    # TODO: Check

        if self.fast_fading:
            # fast_fading_att, jakes_map_ = self.compute_fast_fading_attenuation()
            sinr_vector_n_rb = np.zeros(self.n_rb, dtype=float)
            rx_power_vector_n_rb = np.zeros(self.n_rb, dtype=float)
            sinr = 0
            for i in range(self.n_rb):
                rx_power_vector_n_rb[i] = rx_power + fast_fading_att[i]
                sinr_vector_n_rb[i] = rx_power_vector_n_rb[i] - noise_bw
                sinr = sinr + sinr_vector_n_rb[i]
            sinr = sinr/self.n_rb

        else:
            sinr = rx_power - noise_bw


        return round(sinr, 2)

def get_ch_tr_38_811(t_now, t_old, speed_rx, speed_tx, ds_angle, rx_coord, tx_coord, channel_model, rx_scenario, tx_antenna_mode, dynamic_los,
             elevation_angle, d_sat, h_sat, fc, f_band_rx, outdoor_to_indoor, inside_what, penetration_loss_model,
             d_correlation_map_rx, shadowing, n_rb, jakes_map, fast_fading_model,
             cable_loss_tx, thermal_noise, bw_rb, rx_noise_figure, fast_fading, tx_power, antenna_gain_tx,
             antenna_gain_rx, atmospheric_absorption, desired_delay_spread, fast_fading_los_type, fast_fading_nlos_type, num_rx_ax, num_tx_ax,
                 rng, rx_antenna_mode, ax_panel_polarization):

    ch_tr_38_811 = Ch_tr_38_811(t_now, t_old, speed_rx, speed_tx, ds_angle, rx_coord, tx_coord, channel_model, rx_scenario, tx_antenna_mode, dynamic_los,
                  elevation_angle, d_sat, h_sat, fc, f_band_rx, outdoor_to_indoor, inside_what, penetration_loss_model,
                  d_correlation_map_rx, shadowing, n_rb, jakes_map, fast_fading_model,
                  cable_loss_tx, thermal_noise, bw_rb, rx_noise_figure, fast_fading, tx_power, antenna_gain_tx,
                  antenna_gain_rx, atmospheric_absorption, desired_delay_spread, fast_fading_los_type, fast_fading_nlos_type, num_rx_ax, num_tx_ax,
                 rng, rx_antenna_mode, ax_panel_polarization)

    d_correlation_map_rx = ch_tr_38_811.compute_shadowing()
    path_loss = ch_tr_38_811.compute_path_loss()
    angle_att = ch_tr_38_811.compute_angular_attenuation()
    atmos_att = ch_tr_38_811.get_atmospheric_absorption()
    fast_fading_att, jakes_map = ch_tr_38_811.compute_fast_fading_attenuation(d_correlation_map_rx["los"], d_correlation_map_rx["o2i"])
    sinr = ch_tr_38_811.get_sinr(d_correlation_map_rx, path_loss, angle_att, atmos_att, fast_fading_att)
    ch_outcomes_rx = {"t": d_correlation_map_rx["t"],
                      "x": d_correlation_map_rx["x"],
                      "y": d_correlation_map_rx["y"],
                      "elevation_angle": elevation_angle,
                      "d_3d": d_sat,
                      "o2i": d_correlation_map_rx["o2i"],
                      "los": d_correlation_map_rx["los"],
                      "o2i_loss": d_correlation_map_rx["o2i_loss"],
                      "shadowing": d_correlation_map_rx["shadowing"],
                      "path_loss": path_loss,
                      "angle_att": angle_att,
                      "atmos_att": atmos_att,
                      "fast_fading_att": round(np.mean(fast_fading_att), 2),
                      "sinr": sinr
                      }

    return ch_outcomes_rx, d_correlation_map_rx, jakes_map


# channel_model = "HAPS"
# rx_scenario = "urban"
# dynamic_los = False
# elevation_angle = 90
# h_sat = 8000  # Deployment-D5, HAPS between 8 km and 50 km
# r_earth = 6371*1e3 # in meters
#
# elevation_angle_rad = ma.radians(elevation_angle)
# d_sat = ma.sqrt((r_earth**2) * ((ma.sin(elevation_angle_rad))**2) + h_sat**2 + 2*h_sat*r_earth) - r_earth*ma.sin(elevation_angle_rad)
# print(d_sat)
#
# fc = 2
# f_band_rx = "S-band"
# outdoor_to_indoor = False
# inside_what = "building"
# penetration_loss_model = "low-loss"
#
# t_now = 0
# t_old = 0
# speed_rx = 3
# rx_coord = [50, 50, 1.5]
# tx_antenna_mode = "Sat_ax"
# d_correlation_map_rx = {"t": None, "x": None, "y": None, "shadowing": None, "cl": None, "o2i_loss": None, "los": None,
#                         "d_correlation_sf": None}
# n_rb = 10
# jakes_map = np.zeros([n_rb * 6, 3], dtype=float)
# shadowing = True
# fast_fading_model = "jakes"
# cable_loss_tx = 2
# thermal_noise = -174
# bw_rb = 720e3
# rx_noise_figure = 7
# fast_fading = True
# tx_power = 36
# antenna_gain_tx = 30
# antenna_gain_rx = 10
# ds_angle = 0
#
# ch_outcomes_rx, d_correlation_map_rx, jakes_map = get_ch_tr_38_811(t_now, t_old, speed_rx, speed_tx, ds_angle, rx_coord, channel_model, rx_scenario, tx_antenna_mode, dynamic_los,
#              elevation_angle, d_sat, h_sat, fc, f_band_rx, outdoor_to_indoor, inside_what, penetration_loss_model,
#              d_correlation_map_rx, shadowing, n_rb, jakes_map, fast_fading_model,
#              cable_loss_tx, thermal_noise, bw_rb, rx_noise_figure, fast_fading, tx_power, antenna_gain_tx,
#              antenna_gain_rx)
#
#
# print("ch_outcomes_rx", ch_outcomes_rx)


