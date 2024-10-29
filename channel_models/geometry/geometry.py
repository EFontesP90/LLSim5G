import math as ma
import numpy as np


class Geometry(object):

    """
    06/05/2024
    The class Geometry allows us to compute for a pair rx (receiver, e.g., user equipment) and tx (transmitter, e.g.,
    tbs, abs, or d2d possible forwarding user) their two/three-dimensional distance, their horizontal and vertical angle,
    and their relative speeds. In the case of the horizontal angle,  the angle computation is implemented for a transmitter
    with three symmetric sectors of 120 degrees or four sectors of four symmetric sectors of 90 degrees. In the case of
    d2d communication, the antenna pattern is assumed to be omnidirectional.

    Required attributes:
    (rx_coord, tx_coord,  rx_coord_old, tx_coord_old, t_now, t_old):
    """

    def __init__(self, tx_antenna_mode, rx_coord, tx_coord,  rx_coord_old, tx_coord_old, t_now, t_old):

        if tx_antenna_mode == "one_sectors_90_degrees":
            assert tx_coord[0] == 0 and tx_coord[1] == 0, f"The x, y coordinates of the tbs = X, Y: {tx_coord[0], tx_coord[1]},  must be equal to 0, because in this mode the tbs is assumed in the corner of the grid"


        self.tx_antenna_mode = tx_antenna_mode  # string with the selected tx antenna mode: omni, three_sectors, four_sectors.

        self.rx_coord = rx_coord  # array of 3x1 ([x,y,z]) with coordinates of the rx e.g., [10, 10, 1.5] in meters
        self.tx_coord = rx_coord  # array of 3x1 ([x,y,z]) with coordinates of the tx e.g., [50, 50, 20] in meters

        self.rx_coord_old = rx_coord  # array of 3x1 ([x,y,z]) with coordinates of the rx in the time steap (t-1 = t_old) e.g., [15,15,1.5] in meters
        self.tx_coord_old = rx_coord  # array of 3x1 ([x,y,z]) with coordinates of the tx in the time steap (t-1 = t_old) e.g., [50, 50, 20] in meters. It will be used for aerial base stations (abs)

        self.x_rx = rx_coord[0]  # x coordinate of the receiver (rx)
        self.y_rx = rx_coord[1]  # y coordinate of the receiver (rx)
        self.h_rx = rx_coord[2]  # z coordinate of the receiver (rx)
        self.x_tx = tx_coord[0]  # x coordinate of the transmitter (tx)
        self.y_tx = tx_coord[1]  # y coordinate of the transmitter (tx)
        self.h_tx = tx_coord[2]  # z coordinate of the transmitter (tx)

        self.x_rx_old = rx_coord_old[0]  # old x coordinate of the receiver (rx)
        self.y_rx_old = rx_coord_old[1]  # old y coordinate of the receiver (rx)
        self.h_rx_old = rx_coord_old[2]  # old h coordinate of the receiver (rx)
        self.x_tx_old = tx_coord_old[0]  # old x coordinate of the transmitter (tx)
        self.y_tx_old = tx_coord_old[1]  # old y coordinate of the transmitter (tx)
        self.h_tx_old = tx_coord_old[2]  # old h coordinate of the transmitter (tx)

        self.t_now = t_now  # current time step
        self.t_old = t_old  # previous time step

        self.delta_x = self.x_tx - self.x_rx
        self.delta_y = self.y_tx - self.y_rx
        self.delta_z = self.h_tx - self.h_rx

        self.d_2d = ma.sqrt(pow(self.delta_x, 2) + pow(self.delta_y, 2))  # definition of the two-dimensional distance as an attribute

    def get_d_2d(self):
        return round(self.d_2d, 2)

    def get_d_3d(self):
        d_3d = ma.sqrt(pow(self.h_tx - self.h_rx, 2) + pow(self.d_2d, 2))
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

    def get_tx_speed(self):
        if self.t_now == 0:
            speed_tx = 0
        elif self.t_now > 0:
            delta_x_tx = self.x_tx - self.x_tx_old
            delta_y_tx = self.y_tx - self.y_tx_old
            d_2d_tx = ma.sqrt(pow(delta_x_tx, 2) + pow(delta_y_tx, 2))
            speed_tx = d_2d_tx / (self.t_now - self.t_old)
        return round(speed_tx, 2)

    def get_h_angle(self):  # return the horizontal angle in degrees for a base station with three_sectors (of 120 degrees) or four sectors (of 90 degrees)

        if self.tx_antenna_mode == "three_sectors":

            h_angle = 0  # Maximum antenna gain for a sector of 120 degrees
            if self.d_2d != 0:
                if self.delta_y == 0: h_angle = 30  # 2nd or 3rd quadrant  depending on if delta_x is positive or negative
                elif self.delta_y < 0 and self.delta_x == 0: h_angle = 60  # assumed as 2nd quadrant (in the border between 2nd and 3rd quadrant)
                elif self.delta_y < 0:
                    if ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y))) > 60: h_angle = ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y))) - 60  # 2nd or 3rd quadrant depending on if delta_x is positive or negative
                    else: h_angle = 60 - ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y)))  # 2nd or 3rd quadrant depending on if delta_x is positive or negative
                elif self.delta_y > 0:
                    if ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y))) > 60: h_angle = 120 - ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y))) # 2nd or 3rd quadrant  depending on if delta_x is positive or negative
                    else: h_angle = ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y)))  # 1st quadrant

        elif self.tx_antenna_mode == "four_sectors":
            h_angle = 0  # angle for the maximum antenna gain in the horizontal plain for a sector of 120 degrees
            if self.d_2d != 0:
                if self.delta_y == 0 or self.delta_x == 0: h_angle = 0  # 1st, 2nd, 3rd or 4rt quadrant  depending on if delta_x and delta_y
                else:
                    if ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y))) > 45: h_angle = 90 - ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y)))  # 2nd or 4rd quadrant  depending on if delta_x is positive or negative
                    else: h_angle = ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y)))  # 1st or 3rd quadrant depending on if delta_y is positive or negative

        elif self.tx_antenna_mode == "one_sectors_90_degrees":
            h_angle = 0
            if self.d_2d != 0:
                if self.delta_x == 0: h_angle = 90
                else:
                    if self.delta_y == 0: h_angle = 0
                    else: h_angle = ma.degrees(ma.atan(abs(self.delta_y) / abs(self.delta_x)))

        elif self.tx_antenna_mode == "omni":
            h_angle = 0  # if omnidirectional tx antenna model the horizontal angle has not a significant value

        return round(h_angle, 2)

    def get_h_angle_three_sectors(self):  # return the horizontal angle in degrees for a base station with three_sectors of 120 degrees
        h_angle = 60  # Maximum antenna gain for a sector of 120 degrees
        if self.d_2d != 0:
            if self.delta_y == 0:
                h_angle = 30  # 2nd or 3rd quadrant  depending on if delta_x is positive or negative
            elif self.delta_y < 0 and self.delta_x == 0:
                h_angle = 120  # assumed as 2nd quadrant (in the border between 2nd and 3rd quadrant)
            elif self.delta_y < 0:
                h_angle = 90 - ma.degrees(ma.atan(abs(self.delta_x) / abs(
                    self.delta_y))) + 30  # 2nd or 3rd quadrant depending on if delta_x is positive or negative
            elif self.delta_y > 0:
                if ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y))) > 60:
                    h_angle = ma.degrees(ma.atan(abs(self.delta_x) / abs(
                        self.delta_y))) - 60  # 2nd or 3rd quadrant  depending on if delta_x is positive or negative
                else:
                    h_angle = 60 - ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y)))  # 1st quadrant
        # print("h_angle", h_angle)
        return round(h_angle, 2)

    def get_h_angle_four_sectors(self):  # return the horizontal angle in degrees for a base station with four_sectors of 90 degrees
        h_angle = 45  # angle for the maximum antenna gain in the horizontal plain for a sector of 120 degrees
        if self.d_2d != 0:
            if self.delta_y == 0 or self.delta_x == 0: h_angle = 45  # 1st, 2nd, 3rd or 4rt quadrant  depending on if delta_x and delta_y
            else:
                if ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y))) > 45: h_angle = ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y))) - 45  # 2nd or 4rd quadrant  depending on if delta_x is positive or negative
                else: h_angle = 45 - ma.degrees(ma.atan(abs(self.delta_x) / abs(self.delta_y)))  # 1st or 3rd quadrant depending on if delta_y is positive or negative
        return round(h_angle, 2)

    def get_h_angle_one_sectors_90_degrees(self):  # return the horizontal angle in degrees for the particula use case where a base station with a sector 90 degrees is located in the corner of the service area (for mimo use cases)
        h_angle = 0
        if self.d_2d != 0:
            if self.delta_x == 0: h_angle = 90
            else:
                if self.delta_y == 0: h_angle = 0
                else: h_angle = ma.degrees(ma.atan(abs(self.delta_y) / abs(self.delta_x)))
        return round(h_angle, 2)

    def get_v_angle(self):
        delta_x = self.x_tx - self.x_rx
        delta_y = self.y_tx - self.y_rx
        v_angle = 90  # angle for the minimum antenna gain in the vertical plain for a sector of 120 degrees
        d_2d = ma.sqrt(pow(delta_x, 2) + pow(delta_y, 2))
        if d_2d != 0:
            delta_z = self.h_tx - self.h_rx
            v_angle = ma.degrees(ma.atan(abs(delta_z) / abs(d_2d)))
        return round(v_angle, 2)

    def get_dopple_shift_angle(self):
        delta_x_rx = self.x_rx - self.x_rx_old
        delta_y_rx = self.y_rx - self.y_rx_old
        delta_z_rx = self.h_rx - self.h_rx_old
        delta_t = self.t_now - self.t_old

        if self.t_now == 0: ds_angle = 0  # m/s.
        else:
            v_vector = [delta_x_rx/delta_t, delta_y_rx/delta_t, delta_z_rx/delta_t]  # m/s.

            relative_rx_tx = np.array([self.delta_x, self.delta_y, self.delta_z])

            magnitude_relative_rx_tx = np.linalg.norm(relative_rx_tx)
            magnitude_v_vector = np.linalg.norm(np.array(v_vector))

            normalized_relative_rx_tx = relative_rx_tx/magnitude_relative_rx_tx
            normalized_v_vector = np.array(v_vector) / magnitude_v_vector

            dot_product = np.dot(normalized_relative_rx_tx, normalized_v_vector)
            ds_angle = np.arccos(dot_product)

        return round(ds_angle, 2)


def geometry(tx_antenna_mode, rx_coord, tx_coord, rx_coord_old, tx_coord_old, t_now, t_old):

    gm = Geometry(tx_antenna_mode, rx_coord, tx_coord, rx_coord_old, tx_coord_old, t_now, t_old)

    d_2d = gm.get_d_2d()
    d_3d = gm.get_d_3d()
    speed_rx = gm.get_rx_speed()
    speed_tx = gm.get_tx_speed()
    h_angle = gm.get_h_angle()
    v_angle = gm.get_v_angle()
    ds_angle = gm.get_dopple_shift_angle()
    return d_2d, d_3d, speed_rx, speed_tx, h_angle, v_angle, ds_angle


