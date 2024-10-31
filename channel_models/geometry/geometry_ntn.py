"""
File: geometry.py

Purpose:
This file allows computing for a pair rx (receiver, e.g., user equipment) and tx (transmitter, a Satellite) their
three-dimensional distance, their elevation angle, their relative speeds and Doppler shift angle.

Author: Ernesto Fontes Pupo / Claudia Carballo González
Date: 2024-10-30
Version: 1.0.0
SPDX-License-Identifier: Apache-2.0

"""

# Third-party imports
import math as ma
import numpy as np

# Local application/library-specific imports
import scenario.sattelites_lla_info as sat_pos


class Geometry_ntn(object):

    """
    06/05/2024
    The class Geometry_ntn allows computing for a pair rx (receiver, e.g., user equipment) and tx (transmitter, a Satellite) their
    three-dimensional distance, their elevation angle, their relative speeds and Doppler shift angle.

    Required attributes:
    (grid_lla, grid_xy, rx_coord, tx_lla,  rx_coord_old, tx_lla_old, desired_elevation_angle, t_now, t_old):

    Outputs (geometry):
    d_3d: Three dimensional distance between the tx and rx.
    speed_rx: Speed of the rx regarding the time interval from t-1 (t_old) to t (t_now).
    speed_tx: Speed of the tx regarding the time interval from t-1 (t_old) to t (t_now).
    elevation_angle: Vertical angle between the tx and the rx.
    ds_angle: Doppler shift angle between the tx and rx. The Doppler shift depends on the angle between the direction of
     movement of the source/receiver and the line of sight.

    """

    def __init__(self, grid_lla, grid_xy, rx_coord, tx_lla,  rx_coord_old, tx_lla_old, desired_elevation_angle, t_now, t_old):

        self.grid_lla = grid_lla
        self.grid_xy = grid_xy
        self.rx_coord = rx_coord  # array of 3x1 ([x,y,z]) with coordinates of the rx e.g., [10, 10, 1.5] in meters
        self.tx_lla = tx_lla  # array of 3x1 ([x,y,z]) with coordinates of the tx e.g., [50, 50, 20] in meters

        self.rx_coord_old = rx_coord_old  # array of 3x1 ([x,y,z]) with coordinates of the rx in the time steap (t-1 = t_old) e.g., [15,15,1.5] in meters
        self.tx_lla_old = tx_lla_old  # array of 3x1 ([x,y,z]) with coordinates of the tx in the time steap (t-1 = t_old) e.g., [50, 50, 20] in meters. It will be used for aerial base stations (abs)

        self.desired_elevation_angle = desired_elevation_angle  # string with the selected tx antenna mode: omni, three_sectors, four_sectors.

        self.x_rx = rx_coord[0]  # x coordinate of the receiver (rx)
        self.y_rx = rx_coord[1]  # y coordinate of the receiver (rx)
        self.h_rx = rx_coord[2]  # z coordinate of the receiver (rx)
        self.lat_tx = self.tx_lla[0]  # latitude of the transmitter (tx)
        self.long_tx = self.tx_lla[1]  # longitude of the transmitter (tx)
        self.h_tx = self.tx_lla[2]  # altitude of the transmitter (tx)

        self.x_rx_old = rx_coord_old[0]  # old x coordinate of the receiver (rx)
        self.y_rx_old = rx_coord_old[1]  # old y coordinate of the receiver (rx)
        self.h_rx_old = rx_coord_old[2]  # old h coordinate of the receiver (rx)
        self.lat_tx_old = self.tx_lla_old[0]  # old latitude of the transmitter (tx)
        self.long_tx_old = self.tx_lla_old[1]  # old longitude of the transmitter (tx)
        self.h_tx_old = self.tx_lla_old[2]  # old altitude of the transmitter (tx)

        self.t_now = t_now  # current time step
        self.t_old = t_old  # previous time step

        self.delta_z = self.h_tx - self.h_rx

        self.r_earth = 6371*1e3 # in meters

    def get_d_2d(self):
        return None

    def get_elevation_angle(self):

        # Satellite position in LLA (latitude, longitude, altitude)
        satellite_lat = self.lat_tx  # degrees
        satellite_lon = self.long_tx  # degrees
        satellite_alt = self.h_tx  # altitude in meters

        x_s, y_s, z_s = sat_pos.lla_to_ecef(satellite_lat, satellite_lon, satellite_alt)

        # Grid center in LLA
        grid_lat = self.grid_lla[0]  # degrees
        grid_lon = self.grid_lla[1]  # degrees
        grid_alt = self.grid_lla[2]

        x_g_center, y_g_center, z_g_center = sat_pos.lla_to_ecef(grid_lat, grid_lon, grid_alt)
        x_ecef_rx = x_g_center + (self.x_rx - self.grid_xy[0]/2)
        y_ecef_rx = y_g_center + (self.y_rx - self.grid_xy[0] / 2)
        z_ecef_rx = z_g_center + self.h_rx

        # grid_size = 1000.0  # grid size in meters
        elevation_angle = sat_pos.sat_elevation_angle_from_ecef(x_s, y_s, z_s,
                                                                  x_ecef_rx, y_ecef_rx,
                                                                              z_ecef_rx)

        return round(elevation_angle, 2)

    def get_d_3d(self):
        elevation_angle = self.get_elevation_angle()
        # d_3d = ma.sqrt(pow(self.h_tx - self.h_rx, 2) + pow(self.d_2d, 2))
        elevation_angle_rad = ma.radians(elevation_angle)
        d_3d = ma.sqrt((self.r_earth**2) * ((ma.sin(elevation_angle_rad))**2) + self.h_tx**2 + 2*self.h_tx*self.r_earth) - self.r_earth*ma.sin(elevation_angle_rad)

        return round(d_3d, 2)

    def get_rx_speed(self):
        if self.t_now == 0:
            speed_rx = 0
        elif self.t_now > 0:
            delta_x_rx = self.x_rx - self.x_rx_old
            delta_y_rx = self.y_rx - self.y_rx_old
            d_2d_rx = ma.sqrt(pow(delta_x_rx, 2) + pow(delta_y_rx, 2))
            speed_rx = d_2d_rx / (self.t_now - self.t_old)
        return round(speed_rx, 2)

    def get_tx_speed(self):  # TODO: This is not correct, this implementation is for TN, I must include this
        if self.t_now == 0:
            speed_tx = 0
        elif self.t_now > 0:

            # Satellite position in LLA (latitude, longitude, altitude)
            x_s, y_s, z_s = sat_pos.lla_to_ecef(self.lat_tx, self.long_tx, self.h_tx)
            x_s_old, y_s_old, z_s_old = sat_pos.lla_to_ecef(self.lat_tx_old, self.long_tx_old, self.h_tx_old)

            delta_x_tx = x_s - x_s_old
            delta_y_tx = y_s - y_s_old
            d_2d_tx = ma.sqrt(pow(delta_x_tx, 2) + pow(delta_y_tx, 2))
            speed_tx = d_2d_tx / (self.t_now - self.t_old)
        return round(speed_tx, 2)



    def get_dopple_shift_angle(self):
        delta_x_rx = self.x_rx - self.x_rx_old
        delta_y_rx = self.y_rx - self.y_rx_old
        delta_z_rx = self.h_rx - self.h_rx_old
        delta_t = self.t_now - self.t_old

        if self.t_now == 0:
            ds_angle = 0  # m/s.
        else:
            v_vector = [delta_x_rx / delta_t, delta_y_rx / delta_t, delta_z_rx / delta_t]  # m/s.
            x_s, y_s, z_s = sat_pos.lla_to_ecef(self.lat_tx, self.long_tx, self.h_tx)

            # Grid center in LLA
            grid_lat = self.grid_lla[0]  # degrees
            grid_lon = self.grid_lla[1]  # degrees
            grid_alt = self.grid_lla[2]
            x_g_center, y_g_center, z_g_center = sat_pos.lla_to_ecef(grid_lat, grid_lon, grid_alt)
            x_ecef_rx = x_g_center + (self.x_rx - self.grid_xy[0] / 2)
            y_ecef_rx = y_g_center + (self.y_rx - self.grid_xy[0] / 2)
            z_ecef_rx = z_g_center + self.h_rx

            relative_rx_tx = np.array([x_ecef_rx-x_s, y_ecef_rx-y_s, z_ecef_rx-z_s])

            magnitude_relative_rx_tx = np.linalg.norm(relative_rx_tx)
            magnitude_v_vector = np.linalg.norm(np.array(v_vector))

            normalized_relative_rx_tx = relative_rx_tx / magnitude_relative_rx_tx
            normalized_v_vector = np.array(v_vector) / magnitude_v_vector

            dot_product = np.dot(normalized_relative_rx_tx, normalized_v_vector)
            ds_angle = np.arccos(dot_product)

        return round(ds_angle, 2)


def geometry(grid_lla, grid_xy, rx_coord, tx_coord,  rx_coord_old, tx_coord_old, elevation_angle, t_now, t_old):

    gm = Geometry_ntn(grid_lla, grid_xy, rx_coord, tx_coord,  rx_coord_old, tx_coord_old, elevation_angle, t_now, t_old)

    d_3d = gm.get_d_3d()
    speed_rx = gm.get_rx_speed()
    speed_tx = gm.get_tx_speed()
    elevation_angle = gm.get_elevation_angle()
    ds_angle = gm.get_dopple_shift_angle()
    return d_3d, speed_rx, speed_tx, elevation_angle, ds_angle
