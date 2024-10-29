import numpy as np


class PFC(object):
    """
    14/05/2024
    Channel implementation according to 3gpp tr-38-901.

    Required attributes:
    (channel_model, antenna_mode, shadowing, dynamic_los, dynamic_hb, outdoor_to_indoor, inside_what_o2i, penetration_loss_model,
                 d_2d, d_3d, h_rx, h_tx, h_ceiling, block_density, fc, d_correlation_map_rx, t_now, t_old,
                 speed_rx, rx_coord, h_angle, v_angle, v_tilt, n_rb, jakes_map, fast_fading_model, hb_map_rx,
                 cable_loss_tx, thermal_noise, bw_rb, rx_noise_figure, fast_fading, tx_power, antenna_gain_tx, antenna_gain_rx):

    """

    def __init__(self, antenna_mode, h_angle, v_angle, ax_panel_polarization, delta_h, d_2d, bearing_angle,
                 down_tilt_angle):
        self.antenna_mode = antenna_mode
        self.h_angle = h_angle
        self.v_angle = v_angle
        self.ax_panel_polarization = ax_panel_polarization  # "single", "dual";  single polarized (P =1) or dual polarized (P =2)

        self.delta_h = delta_h  # difference between the h of the tx and he rx
        self.d_2d = d_2d

        self.bearing_angle = bearing_angle
        self.down_tilt_angle = down_tilt_angle
        self.v_tilt = self.down_tilt_angle
        self.slant_angle = np.degrees(np.arctan(self.delta_h / self.d_2d))



    # TODO, review the angules assumption in the LCS and the GCS, is 0 or 90 the ideal orientation?

    def wrap_to_pi(self, angle_rad):
        return (angle_rad + np.pi) % (2 * np.pi) - np.pi

    def wrap_to_pi_positive(self, angle_rad):
        wrapped_angle = angle_rad % (2 * np.pi)  # Wrap angle between 0 and 2*pi
        if wrapped_angle >= np.pi:
            wrapped_angle -= 2 * np.pi  # Adjust angle to be between -pi and pi
        return wrapped_angle

    def wrap_to_180(self, angle_degrees):
        wrapped_angle = (angle_degrees + 180) % 360 - 180
        return wrapped_angle

    def wrap_to_180_positive(self, angle_degrees):
        wrapped_angle = angle_degrees % 360
        if wrapped_angle > 180:
            wrapped_angle = 360 - wrapped_angle
        return wrapped_angle

    def compute_global_angular_attenuation(self):

        self.v_angle = self.wrap_to_180_positive(self.v_angle)
        self.h_angle = self.wrap_to_180(self.h_angle)

        g_e_max = 8
        max_h_angle_att = 30  # is the 3 dB beamwidth (corresponding to h_angle_3dB= 70º)

        if self.antenna_mode == "omni":  # omni, three_sectors, four_sectors
            angle_att = 0

        # Report  ITU-R  M.2135-1 (12/2009) for Simplified antenna pattern
        # Table 7.3-1: Radiation power pattern of a single antenna element (7.7.4.1 Exemplary filters/antenna patterns, tr-38-901)
        elif self.antenna_mode == "three_sectors":  # TODO: Check the assumed h_angle_3dB= 70º for a 120º sector

            h_angle_att = -min(12 * pow((self.h_angle) / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation
            v_angle_att = -min(12 * pow((self.v_angle - self.v_tilt) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation
            angle_att = -min(-(h_angle_att + v_angle_att), max_h_angle_att) + g_e_max

        elif self.antenna_mode == "four_sectors":  # TODO: Check the assumed h_angle_3dB= 60º for a 90º sector

            h_angle_att = -min(12 * pow(self.h_angle / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation
            v_angle_att = -min(12 * pow((self.v_angle - self.v_tilt) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation
            angle_att = -min(-(h_angle_att + v_angle_att), max_h_angle_att) + g_e_max

        elif self.antenna_mode == "one_sectors_90_degrees":  # TODO: Check the assumed h_angle_3dB= 60º for a 90º sector

            h_angle_att = -min(12 * pow((45 - self.h_angle) / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation
            v_angle_att = -min(12 * pow( (self.v_angle - self.v_tilt ) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation
            angle_att = -min(-(h_angle_att + v_angle_att), max_h_angle_att) + g_e_max

        angle_att_linear = 10**(angle_att/10)
        return angle_att_linear

    def local_angle_zenith_azimuth(self):
        bearing_angle_ = np.radians(self.bearing_angle)  # Bearing angle (𝛼)
        down_tilt_angle_ = np.radians(self.down_tilt_angle)  # Downtilt angle (𝛽)
        slant_angle_ = np.radians(self.slant_angle)  # Slant angle (𝛾)

        v_angle_ = np.radians(self.v_angle)
        h_angle_ = np.radians(self.h_angle)

        # local_angle_zenith/vertical
        theta_prime = np.arccos( np.cos(down_tilt_angle_)*np.cos(slant_angle_)*np.cos(v_angle_)
                                 + ( np.sin(down_tilt_angle_)*np.cos(slant_angle_)*np.cos(h_angle_-bearing_angle_)
                                     - np.sin(slant_angle_)*np.sin(h_angle_-bearing_angle_) )*np.sin(v_angle_) )


        # local_angle_azimuth/horizontal
        phi_prime = np.angle( (np.cos(down_tilt_angle_)*np.sin(v_angle_)*np.cos(h_angle_-bearing_angle_)
                               - np.sin(down_tilt_angle_)*np.cos(v_angle_))
                              + ( 1j * (np.cos(down_tilt_angle_)*np.sin(slant_angle_)*np.cos(v_angle_) + ( np.sin(down_tilt_angle_)*np.sin(slant_angle_)*np.cos(h_angle_-bearing_angle_)
                                     + np.cos(slant_angle_)*np.sin(h_angle_-bearing_angle_) )*np.sin(v_angle_)) ) )
        return theta_prime, phi_prime

    def compute_local_angular_attenuation(self, theta_prime, phi_prime):

        theta_prime = np.degrees(theta_prime)
        # print("theta_prime", theta_prime)
        phi_prime = np.degrees(phi_prime)
        # print("phi_prime", phi_prime)

        theta_prime = self.wrap_to_180_positive(theta_prime)
        phi_prime = self.wrap_to_180(phi_prime)

        g_e_max = 8
        max_h_angle_att = 30  # is the 3 dB beamwidth (corresponding to h_angle_3dB= 70º)

        if self.antenna_mode == "omni":  # omni, three_sectors, four_sectors
            angle_att = 0

        # Report  ITU-R  M.2135-1 (12/2009) for Simplified antenna pattern
        # Table 7.3-1: Radiation power pattern of a single antenna element (7.7.4.1 Exemplary filters/antenna patterns, tr-38-901)
        elif self.antenna_mode == "three_sectors":  # TODO: Check the assumed h_angle_3dB= 70º for a 120º sector

            h_angle_att = -min(12 * pow((phi_prime - 90) / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation

            v_angle_att = -min(12 * pow((theta_prime - 90 - self.v_tilt) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation

            angle_att = -min(-(h_angle_att + v_angle_att), max_h_angle_att) + g_e_max

        elif self.antenna_mode == "four_sectors":  # TODO: Check the assumed h_angle_3dB= 60º for a 90º sector

            h_angle_att = -min(12 * pow((phi_prime - 90) / 65, 2),
                               max_h_angle_att)  # compute horizontal pattern attenuation

            v_angle_att = -min(12 * pow((theta_prime - 90 - self.v_tilt) / 65, 2),
                               max_h_angle_att)  # compute vertical pattern attenuation
            angle_att = -min(-(h_angle_att + v_angle_att), max_h_angle_att) + g_e_max

        elif self.antenna_mode == "one_sectors_90_degrees":  # TODO: Check the assumed h_angle_3dB= 60º for a 90º sector

            h_angle_att = -min(12 * pow((45 - phi_prime - 90) / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation
            v_angle_att = -min(12 * pow( (theta_prime - 90 - self.v_tilt ) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation
            angle_att = -min(-(h_angle_att + v_angle_att), max_h_angle_att) + g_e_max

        angle_att_linear = 10 ** (angle_att / 10)
        return angle_att_linear

    def compute_local_angular_attenuation_los(self, theta_prime, phi_prime):

        theta_prime = np.degrees(theta_prime)
        # print("theta_prime", theta_prime)
        phi_prime = np.degrees(phi_prime)
        # print("phi_prime", phi_prime)

        theta_prime = self.wrap_to_180_positive(theta_prime)
        phi_prime = self.wrap_to_180(phi_prime)

        g_e_max = 8
        max_h_angle_att = 30  # is the 3 dB beamwidth (corresponding to h_angle_3dB= 70º)

        if self.antenna_mode == "omni":  # omni, three_sectors, four_sectors
            angle_att = 0

        # Report  ITU-R  M.2135-1 (12/2009) for Simplified antenna pattern
        # Table 7.3-1: Radiation power pattern of a single antenna element (7.7.4.1 Exemplary filters/antenna patterns, tr-38-901)
        elif self.antenna_mode == "three_sectors":  # TODO: Check the assumed h_angle_3dB= 70º for a 120º sector

            h_angle_att = -min(12 * pow((phi_prime - 90) / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation

            v_angle_att = -min(12 * pow((theta_prime - self.v_tilt) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation

            angle_att = -min(-(h_angle_att + v_angle_att), max_h_angle_att) + g_e_max

        elif self.antenna_mode == "four_sectors":  # TODO: Check the assumed h_angle_3dB= 60º for a 90º sector

            h_angle_att = -min(12 * pow((phi_prime - 90) / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation
            v_angle_att = -min(12 * pow((theta_prime - self.v_tilt) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation
            angle_att = -min(-(h_angle_att + v_angle_att), max_h_angle_att) + g_e_max

        elif self.antenna_mode == "one_sectors_90_degrees":  # TODO: Check the assumed h_angle_3dB= 60º for a 90º sector

            h_angle_att = -min(12 * pow((45 - phi_prime - 90) / 65, 2), max_h_angle_att)  # compute horizontal pattern attenuation
            v_angle_att = -min(12 * pow( (theta_prime - self.v_tilt ) / 65, 2), max_h_angle_att)  # compute vertical pattern attenuation
            angle_att = -min(-(h_angle_att + v_angle_att), max_h_angle_att) + g_e_max

        angle_att_linear = 10 ** (angle_att / 10)
        return angle_att_linear

    def compute_local_pfc(self, angle_att):


        # angle_att_prime_linear = angle_att_prime

        if self.ax_panel_polarization == "dual":
            polarization_slant_angle = 45
        elif self.ax_panel_polarization == "single":
            polarization_slant_angle = 0

        pfc_phi_prime = np.sqrt(angle_att) * np.cos(np.radians(polarization_slant_angle))

        pfc_theta_prime = np.sqrt(angle_att) * np.sin(np.radians(polarization_slant_angle))

        local_pfc = np.array([pfc_theta_prime, pfc_phi_prime])

        return local_pfc

    def compute_global_pfc(self, local_pfc, theta_prime, phi_prime):

        alpha = np.radians(self.bearing_angle)  # Bearing angle (𝛼)
        beta = np.radians(self.down_tilt_angle)  # Downtilt angle (𝛽)
        gamma = np.radians(self.slant_angle)  # Slant angle (𝛾)

        theta = np.radians(self.v_angle)
        phi = np.radians(self.h_angle)

        real_part = (np.sin(gamma) * np.cos(theta) * np.sin(phi - alpha) +
                     np.cos(gamma) * (
                                 np.cos(beta) * np.sin(theta) - np.sin(beta) * np.cos(theta) * np.cos(phi - alpha)))
        imag_part = (np.sin(gamma) * np.cos(phi - alpha) +
                     np.sin(beta) * np.cos(gamma) * np.sin(phi - alpha))
        complex_number = real_part + 1j * imag_part
        psi = np.angle(complex_number)

        # R = np.array([
        #     [np.cos(bearing_angle) * np.cos(down_tilt_angle), np.cos(bearing_angle) * np.sin(down_tilt_angle) * np.sin(slant_angle) - np.sin(bearing_angle) * np.cos(slant_angle),
        #      np.cos(bearing_angle) * np.sin(down_tilt_angle) * np.cos(slant_angle) + np.sin(bearing_angle) * np.sin(slant_angle)],
        #     [np.sin(bearing_angle) * np.cos(down_tilt_angle), np.sin(bearing_angle) * np.sin(down_tilt_angle) * np.sin(slant_angle) + np.cos(bearing_angle) * np.cos(slant_angle),
        #      np.sin(bearing_angle) * np.sin(down_tilt_angle) * np.cos(slant_angle) - np.cos(bearing_angle) * np.sin(slant_angle)],
        #     [-np.sin(down_tilt_angle), np.cos(down_tilt_angle) * np.sin(slant_angle), np.cos(down_tilt_angle) * np.cos(slant_angle)]
        # ])
        #
        # theta_hat = np.array([np.cos(v_angle_)*np.cos(h_angle_), np.cos(v_angle_)*np.sin(h_angle_), -np.sin(v_angle_)])  # unit vectors
        # phi_hat = np.array([-np.sin(h_angle_), np.cos(h_angle_), 0])  # unit vectors
        #
        # theta_hat_prime = np.array([np.cos(theta_prime) * np.cos(phi_prime), np.cos(theta_prime) * np.sin(phi_prime), -np.sin(theta_prime)])  # unit vectors
        #
        # dot_product_theta = np.dot(theta_hat.T, np.dot(R, theta_hat_prime))
        # dot_product_phi = np.dot(phi_hat.T, np.dot(R, theta_hat_prime))
        # psi = np.angle(dot_product_theta + 1j * dot_product_phi)  # angle psi

        transformation_matrix = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi), np.cos(psi)]
        ])
        global_pfc = np.dot(transformation_matrix, local_pfc)
        # print("local_pfc", local_pfc)
        # print("global_pfc", global_pfc)


        #global_pfc = np.array([pfc_theta, pfc_phi])
        return global_pfc



def pfc_los(antenna_mode, h_angle, v_angle, ax_panel_polarization, delta_h, d_2d, bearing_angle,
                 down_tilt_angle):

    pfc_ = PFC(antenna_mode, h_angle, v_angle, ax_panel_polarization, delta_h, d_2d, bearing_angle,
                 down_tilt_angle)

    # angle_att = pfc.compute_global_angular_attenuation()
    theta_prime, phi_prime = pfc_.local_angle_zenith_azimuth()
    # angle_att = pfc_.compute_global_angular_attenuation()
    angle_att = pfc_.compute_local_angular_attenuation_los(theta_prime, phi_prime)
    local_pfc = pfc_.compute_local_pfc(angle_att)
    global_pfc = pfc_.compute_global_pfc(local_pfc, v_angle, h_angle)
    return global_pfc

def pfc_nlos(antenna_mode, h_angle, v_angle, ax_panel_polarization, delta_h, d_2d, bearing_angle,
                 down_tilt_angle):

    pfc_ = PFC(antenna_mode, h_angle, v_angle, ax_panel_polarization, delta_h, d_2d, bearing_angle,
                 down_tilt_angle)

    # angle_att = pfc.compute_global_angular_attenuation()
    theta_prime, phi_prime = pfc_.local_angle_zenith_azimuth()
    # angle_att = pfc_.compute_global_angular_attenuation()
    angle_att = pfc_.compute_local_angular_attenuation(theta_prime, phi_prime)
    local_pfc = pfc_.compute_local_pfc(angle_att)
    global_pfc = pfc_.compute_global_pfc(local_pfc, theta_prime, phi_prime)
    return global_pfc



# antenna_mode = "three_sectors"
# h_angle = 0
# v_angle = 15
# v_tilt = 15
# ax_panel_polarization = "dual"  # "single", "dual";  single polarized (P =1) or dual polarized (P =2)
# delta_h = 20
# d_2d = 100
#
# bearing_angle = 0
# down_tilt_angle = 15

# pfc = Polarized_Field_Component(antenna_mode, h_angle, v_angle, v_tilt, ax_panel_polarization, delta_h, d_2d, bearing_angle,
#                                 down_tilt_angle)
# angle_att = pfc.compute_global_angular_attenuation()
# theta_prime, phi_prime = pfc.local_angle_zenith_azimuth()
# angle_att_prime = pfc.compute_local_angular_attenuation(theta_prime, phi_prime)
# local_pfc = pfc.compute_local_pfc(angle_att_prime)
#
# global_pfc = pfc.compute_global_pfc(local_pfc, theta_prime, phi_prime)

# global_pfc = polarized_field_component(antenna_mode, h_angle, v_angle, ax_panel_polarization, delta_h, d_2d, bearing_angle,
#                  down_tilt_angle)

# print("angle_att", angle_att)
# print("theta_prime", np.degrees(theta_prime))
# print("phi_prime", np.degrees(phi_prime))
# print("angle_att_prime", angle_att_prime)
# print("local_pfc", local_pfc)

# print("global_pfc", global_pfc)

