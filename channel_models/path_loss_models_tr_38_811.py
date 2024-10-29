import math as ma


class NTN_Sat_path_loss(object):
    """
    07/05/2024
    Path loss computation between a satellite or HAPS transmitter and an NTN terminal
    with a link model s.t. 6.6.2 Path loss and Shadow fading, 3GPP TR 38.811 version
    15.4.0 Release 15.

    Required attributes:
    (d_2d, d_3d, h_rx, h_tx, fc, los, outdoor_to_indoor):
    """

    def __init__(self, d_sat, h_sat, fc):

        self.d_sat = d_sat  # For a ground terminal, the distance d (a.k.a. slant range)
        self.h_sat = h_sat  # satellite/HAPS altitude
        self.fc = fc  # Simulation Frequency in GHz

    def compute_path_loss(self):
        # Cumpute Path loss
        path_loss = 32.45 + 20 * ma.log(self.fc, 10) + 20 * ma.log(self.d_sat, 10)
        return path_loss



def ntn_sat_path_loss(d_sat, h_sat, fc):
    sat_pl = NTN_Sat_path_loss(d_sat, h_sat, fc)

    path_loss = sat_pl.compute_path_loss()
    return path_loss


