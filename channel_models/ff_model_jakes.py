"""
File: ff_model_jakes.py

Purpose:
This file comprises the Jakes fast fading model (as implemented in Simu5G OmNet++) s.t.
P. Dent, G. E. Bottomley, and T. Croft, “Jakes fading model revisited,”
Electronics letters, vol. 13, no. 29, pp. 1162–1163, 1993.

Authors: Ernesto Fontes Pupo / Claudia Carballo González
        University of Cagliari
Date: 2024-10-30
Version: 1.0.0
                   GNU LESSER GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

    LLSim5G is a link-level simulator for HetNet 5G use cases.
    Copyright (C) 2024  Ernesto Fontes, Claudia Carballo

"""


# Third-party imports
import math as ma
import numpy as np
from numpy import random
from numpy.random import exponential

# Local application/library-specific imports
import channel_models.ff_models_tr_38_901and811.cdl_matrixs as cdl_ma


def linear_to_db(linear):
    return 10*ma.log(linear,10)

def db_to_linear(dB):
    return pow(10, dB/10)

def clip_outliers(data, lower_percentile=5, upper_percentile=95):
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return np.clip(data, lower_bound, upper_bound)

class Jakes_channel():
    """
    07/05/2024

    Required attributes:
    (ds_angle, rx_speed, tx_speed, t_now, n_rb, fc, jakes_map, desired_delay_spread, atmospheric_absorption,
     channel_model, los, o2i, d_3d, h_sat, v_angle):

    Outputs (jakes_channel):
    fast_fading_att: Fast fading attenuation in dB and for each resource block.
    jakes_map: Matrix for storing the multipath components from t-1 (t_old) to t (t_now) avoiding abrupt changes in the
    resulting fast-fading attenuation.

    """

    def __init__(self, ds_angle, rx_speed, tx_speed, t_now, n_rb, fc, jakes_map, desired_delay_spread, atmospheric_absorption,
                 channel_model, los, o2i, d_3d, h_sat, v_angle):

        self.ds_angle = ds_angle
        self.rx_speed = rx_speed  # ue speed
        self.tx_speed = tx_speed  # ue speed
        self.t_now = t_now  # current time step
        #self.rb = rb  # actual rb for iterating over all b_rb
        self.n_rb = n_rb  # Nuber of rb assigned to the ue

        self.fc = fc  # carrier frequency

        self.jakes_map = jakes_map  # array of the resulting jackFadingMap of the K users respect to the BS and each proximity user in D2D comm, fadingTaps X [delay amp, AoA]

        self.desired_delay_spread = desired_delay_spread
        self.atmospheric_absorption = atmospheric_absorption
        self.channel_model = channel_model
        self.los = los
        self.o2i = o2i
        self.d_3d = d_3d
        self.h_sat = h_sat
        self.v_angle = v_angle
        #JakesFading_Map = np.zeros([K, K + 1, NRb * fadingPaths, 3], dtype=float)

    def get_desired_delay_spread(self):
        # valid for UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH
        if self.desired_delay_spread == "Very short":
            dds = cdl_ma.desired_delay_spread[0]
        elif self.desired_delay_spread == "Short":
            dds = cdl_ma.desired_delay_spread[1]
        elif self.desired_delay_spread == "Nominal":
            dds = cdl_ma.desired_delay_spread[2]
        elif self.desired_delay_spread == "Long":
            dds = cdl_ma.desired_delay_spread[3]
        elif self.desired_delay_spread == "Very long":
            dds = cdl_ma.desired_delay_spread[4]
        elif self.desired_delay_spread == "None":
            dds = 1e-9

        return dds

    def delay_scaling(self):
        # valid for UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH
        if self.channel_model == "UMi" or self.channel_model == "A2G" or self.channel_model == "D2D":
            if self.o2i:
                r_tau = cdl_ma.delay_scaling["UMi"][2]
            else:
                if self.los:
                    r_tau = cdl_ma.delay_scaling["UMi"][0]
                else:
                    r_tau = cdl_ma.delay_scaling["UMi"][1]
        elif self.channel_model == "UMa":
            if self.o2i:
                r_tau = cdl_ma.delay_scaling["UMa"][2]
            else:
                if self.los:
                    r_tau = cdl_ma.delay_scaling["UMa"][0]
                else:
                    r_tau = cdl_ma.delay_scaling["UMa"][1]
        elif self.channel_model == "RMa":
            if self.o2i:
                r_tau = cdl_ma.delay_scaling["RMa"][2]
            else:
                if self.los:
                    r_tau = cdl_ma.delay_scaling["RMa"][0]
                else:
                    r_tau = cdl_ma.delay_scaling["RMa"][1]
        elif self.channel_model == "InH-Mixed" or self.channel_model == "InH-Open":
            if self.los:
                r_tau = cdl_ma.delay_scaling["InH"][0]
            else:
                r_tau = cdl_ma.delay_scaling["InH"][1]
        else:
            if self.los:
                r_tau = cdl_ma.delay_scaling["InF"][0]
            else:
                r_tau = cdl_ma.delay_scaling["InF"][1]
        return r_tau

    def calculate_oxygen_loss(self, c_delay_normalized):
        dds = self.get_desired_delay_spread()
        ol_n_fc = 0
        if not self.atmospheric_absorption:
            ol_n_fc = ol_n_fc
        else:
            # Speed of light
            c = 3e8  # m/s
            # Calculate OL_n(fc)
            if self.fc <= 52:
                alpha_fc = 0
            elif self.fc <= 53:
                alpha_fc = 1
            elif self.fc <= 54:
                alpha_fc = 2.2
            elif self.fc <= 55:
                alpha_fc = 4
            elif self.fc <= 56:
                alpha_fc = 6.6
            elif self.fc <= 57:
                alpha_fc = 9.7
            elif self.fc <= 58:
                alpha_fc = 12.6
            elif self.fc <= 59:
                alpha_fc = 14.6
            elif self.fc <= 60:
                alpha_fc = 15
            elif self.fc <= 61:
                alpha_fc = 14.6
            elif self.fc <= 62:
                alpha_fc = 14.3
            elif self.fc <= 63:
                alpha_fc = 10.5
            elif self.fc <= 64:
                alpha_fc = 6.8
            elif self.fc <= 65:
                alpha_fc = 3.9
            elif self.fc <= 66:
                alpha_fc = 1.9
            elif self.fc <= 67:
                alpha_fc = 1
            elif self.fc > 67:
                alpha_fc = 0

            r_tau = self.delay_scaling()
            ds = 363e-9

            if self.los:
                tau_delta = 0
            else:
                x_n = np.random.uniform(0, 1)
                tau_prime_n = -r_tau * ds * np.log(x_n)
                tau_delta = tau_prime_n
            ol_n_fc = -(alpha_fc / 1000) * (self.d_3d + c * (dds * c_delay_normalized + tau_delta))

        return ol_n_fc

    def compute_jakes_faiding(self):
        delay_rms = 363e-9
        fading_paths = 6
        f = self.fc * 1000000000
        # t = random.uniform(0, 1)
        t = self.t_now
        # t = 1
        c = 300000000  # 3.0×10^8 m/s is the propagation velocity in free space
        # //if this is the first time that we compute fading for current user
        if self.t_now == 0 and np.all((self.jakes_map == 0)):
            for i in range(self.n_rb):
                for ii in range(fading_paths):
                    self.jakes_map[i * fading_paths + ii][2] = ma.cos(random.uniform(0, np.pi))
                    self.jakes_map[i * fading_paths + ii][0] = exponential(delay_rms, 1)[0]



        ds_angle_rad = ma.radians(self.ds_angle)
        # "Geo", "Non-Geo", "HAPS"
        if self.channel_model == "Geo" or self.channel_model == "Non-Geo" or self.channel_model == "HAPS":
            r_earth = 6371 * 1e3  # in meters
            angle = np.radians(self.v_angle)
            h = self.h_sat
            # fd = self.rx_speed * np.cos(ds_angle_rad) / self.lambda_0
            doppler_shift = ((self.rx_speed * f * ma.cos(ds_angle_rad))/c) + (self.tx_speed * c/ self.fc) * (r_earth*np.cos(angle)/(r_earth + h))
        else:
            doppler_shift = (self.rx_speed * f * ma.cos(ds_angle_rad)) / c
            # doppler_shift = (self.rx_speed * f) / c
            # print("self.rx_speed", self.rx_speed)
            # print("doppler_shift", doppler_shift)
        fast_fading_att = np.zeros(self.n_rb, dtype=float)
        for i in range(self.n_rb):
            re_h = 0
            im_h = 0
            for ii in range(fading_paths):

                phi_d = self.jakes_map[i * fading_paths + ii][2] * doppler_shift  # Phase shift due to Doppler => t-selectivity
                phi_i = self.jakes_map[i * fading_paths + ii][0] * f  # Phase shift due to delay spread => f-selectivity
                phi = 2.00 * 3.1416 * (phi_d * t - phi_i)  # Calculate resulting phase due to t-selective and f-selective fading.
                attenuation = (1.00 / ma.sqrt(fading_paths))  # attenuation per path.
                ol_n_fc = self.calculate_oxygen_loss(phi_i)
                oxygen_loss = 10 ** (ol_n_fc/ 10)
                self.jakes_map[i * fading_paths + ii][1] = attenuation

                # // Convert to cartesian form and aggregate {Re, Im} over all fading paths.
                re_h = re_h + oxygen_loss*attenuation * ma.cos(phi)
                im_h = im_h - oxygen_loss*attenuation * ma.sin(phi)
            fast_fading_att[i] = linear_to_db(re_h * re_h + im_h * im_h)
        # print(fast_fading_att)
        fast_fading_att_cliped = clip_outliers(fast_fading_att)
        return fast_fading_att_cliped, self.jakes_map

def jakes_channel(ds_angle, rx_speed, tx_speed, t_now, n_rb, fc, jakes_map, desired_delay_spread, atmospheric_absorption,
                 channel_model, los, o2i, d_3d, h_sat, v_angle):
    jakes = Jakes_channel(ds_angle, rx_speed, tx_speed, t_now, n_rb, fc, jakes_map, desired_delay_spread, atmospheric_absorption,
                 channel_model, los, o2i, d_3d, h_sat, v_angle)

    fast_fading_att, jakes_map = jakes.compute_jakes_faiding()
    return fast_fading_att, jakes_map
