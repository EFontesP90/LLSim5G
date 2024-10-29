import numpy as np
import scipy.signal
import random

# import channel_models.ff_models_tr_38_901and811.cdl_models
from channel_models.ff_models_tr_38_901and811.cdl_models import CDL_models
# import channel_models.ff_models_tr_38_901and811.cdl_models as cdl
import channel_models.ff_models_tr_38_901and811.cdl_matrixs as cdl_ma
import channel_models.ff_models_tr_38_901and811.tdl_matrixs as tdl_ma
import channel_models.ff_models_tr_38_901and811.polarized_field_component as p_f_c


class TDL_models(CDL_models):
    """
    25/04/2024


    Required attributes:
    ():

    """

    def __init__(self, channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type,
                 num_rx_ax, num_tx_ax, d_2d, d_3d, bearing_angle,
                 down_tilt_angle, h_angle, v_angle, ds_angle, fc, rx_speed, tx_speed, rx_coord, tx_coord, t_now, t_old):

        self.channel_model = channel_model  # string with the selected link channel to be modeled from the tr_138_901: Just valid for UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH.
        self.los = los
        self.o2i = o2i
        self.atmospheric_absorption = atmospheric_absorption
        self.desired_delay_spread = desired_delay_spread

        self.tdl_type = tdl_type  # nlos: A, B, C; los: C, D
        self.cp = tdl_ma.tdl_cp[self.tdl_type]
        self.c_delay_normalized = self.cp[:, 0]
        self.c_power = 10 ** (self.cp[:, 1] / 10)
        self.pcp = cdl_ma.cdl_pcp[self.tdl_type]
        self.xpr_db = self.pcp[4]
        self.tdl_rice_factors = tdl_ma.tdl_rice_factors[self.tdl_type]

        # self.n = 1  # number of clusters 𝑁
        self.n = self.cp.shape[0]  # number of clusters 𝑁
        self.u = num_rx_ax  # Number of receiver antenna elements
        self.s = num_tx_ax  # Number of transmitter antenna elements


        self.d_2d = d_2d
        self.d_3d = d_3d

        self.bearing_angle = bearing_angle
        self.down_tilt_angle = down_tilt_angle
        self.v_tilt = self.down_tilt_angle
        # self.slant_angle = np.degrees(np.arctan(self.delta_h / self.d_2d))

        self.v_angle = v_angle
        self.h_angle = h_angle
        self.ds_angle = ds_angle

        self.fc = fc
        self.lambda_0 = 300000000 / (self.fc * 1e9)

        self.rx_speed = rx_speed
        self.rx_speed = rx_speed
        self.tx_coord = tx_coord
        self.tx_speed = tx_speed

        self.t_now = t_now
        self.t_old = t_old


    # Function to create a delta function using scipy.signal.unit_impulse
    # def delta_function(self, tau, center, dt):
    #     index = int(center / dt)
    #     return scipy.signal.unit_impulse(len(tau), index)

    # def delay_scaling(self):
    #     # valid for UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH
    #     if self.channel_model == "UMi" or self.channel_model == "A2G" or self.channel_model == "D2D":
    #         if self.o2i:
    #             r_tau = cdl_ma.delay_scaling["UMi"][2]
    #         else:
    #             if self.los:
    #                 r_tau = cdl_ma.delay_scaling["UMi"][0]
    #             else:
    #                 r_tau = cdl_ma.delay_scaling["UMi"][1]
    #     elif self.channel_model == "UMa":
    #         if self.o2i:
    #             r_tau = cdl_ma.delay_scaling["UMa"][2]
    #         else:
    #             if self.los:
    #                 r_tau = cdl_ma.delay_scaling["UMa"][0]
    #             else:
    #                 r_tau = cdl_ma.delay_scaling["UMa"][1]
    #     elif self.channel_model == "RMa":
    #         if self.o2i:
    #             r_tau = cdl_ma.delay_scaling["RMa"][2]
    #         else:
    #             if self.los:
    #                 r_tau = cdl_ma.delay_scaling["RMa"][0]
    #             else:
    #                 r_tau = cdl_ma.delay_scaling["RMa"][1]
    #     elif self.channel_model == "InH-Mixed" or self.channel_model == "InH-Open":
    #         if self.los:
    #             r_tau = cdl_ma.delay_scaling["InH"][0]
    #         else:
    #             r_tau = cdl_ma.delay_scaling["InH"][1]
    #     else:
    #         if self.los:
    #             r_tau = cdl_ma.delay_scaling["InF"][0]
    #         else:
    #             r_tau = cdl_ma.delay_scaling["InF"][1]
    #     return r_tau
    #
    # def get_desired_delay_spread(self):
    #     # valid for UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH
    #     if self.desired_delay_spread == "Very short":
    #         dds = cdl_ma.desired_delay_spread[0]
    #     elif self.desired_delay_spread == "Short":
    #         dds = cdl_ma.desired_delay_spread[1]
    #     elif self.desired_delay_spread == "Nominal":
    #         dds = cdl_ma.desired_delay_spread[2]
    #     elif self.desired_delay_spread == "Long":
    #         dds = cdl_ma.desired_delay_spread[3]
    #     elif self.desired_delay_spread == "Very long":
    #         dds = cdl_ma.desired_delay_spread[4]
    #     elif self.desired_delay_spread == "None":
    #         dds = 1e-9
    #
    #     return dds
    #
    # def c_mm(self, large_scale_correlations):
    #     c_cmm_matrix = np.empty((3, 7, 7), dtype=np.float_)
    #     for c, C in enumerate(large_scale_correlations):
    #         c_mm_squared = np.array(
    #             [
    #                 #    DS,    ASD,   ASA,   ZSA,   ZSD,   K,     SF
    #                 [1.0, C[0], C[1], C[15], C[14], C[8], C[4]],  # DS
    #                 [C[0], 1.0, C[5], C[17], C[16], C[6], C[3]],  # ASD
    #                 [C[1], C[5], 1.0, C[19], C[18], C[7], C[2]],  # ASA
    #                 [C[15], C[17], C[19], 1.0, C[20], C[13], C[11]],  # ZSA
    #                 [C[14], C[16], C[18], C[20], 1.0, C[12], C[10]],  # ZSD
    #                 [C[8], C[6], C[7], C[13], C[12], 1.0, C[9]],  # K
    #                 [C[4], C[3], C[2], C[11], C[10], C[9], 1.0],  # SF
    #             ],
    #             dtype=np.float_,
    #         )
    #
    #         # Section 4 of ETSI TR 138.901 v17.0.0 hints at using the cholesky decomposition
    #         # to enforce the expected cross-correlations between the large-scale parameters
    #         c_cmm_matrix[c, ...] = np.linalg.cholesky(c_mm_squared)
    #
    #     return c_cmm_matrix, c_mm_squared
    #
    # def lage_scale_parameters(self):
    #
    #     ds_mean_std = cdl_ma.ds_mean_std(self.channel_model, self.fc, self.los, self.o2i)
    #     ds_mean = ds_mean_std[0]
    #     ds_std = ds_mean_std[1]
    #
    #     if self.channel_model == "UMi" or self.channel_model == "A2G" or self.channel_model == "D2D":
    #         k_mean = cdl_ma.k_mean_std["UMi"][0]
    #         k_std = cdl_ma.k_mean_std["UMi"][1]
    #         large_scale_correlations = cdl_ma.umi_large_scale_correlations
    #     elif self.channel_model == "UMa":
    #         k_mean = cdl_ma.k_mean_std["UMa"][0]
    #         k_std = cdl_ma.k_mean_std["UMa"][1]
    #         large_scale_correlations = cdl_ma.uma_large_scale_correlations
    #     elif self.channel_model == "RMa":
    #         k_mean = cdl_ma.k_mean_std["RMa"][0]
    #         k_std = cdl_ma.k_mean_std["RMa"][1]
    #         large_scale_correlations = cdl_ma.rma_large_scale_correlations
    #     elif self.channel_model == "InH-Mixed" or self.channel_model == "InH-Open":
    #         k_mean = cdl_ma.k_mean_std["InH"][0]
    #         k_std = cdl_ma.k_mean_std["InH"][1]
    #         large_scale_correlations = cdl_ma.inh_large_scale_correlations
    #     else:
    #         k_mean = cdl_ma.k_mean_std["InF"][0]
    #         k_std = cdl_ma.k_mean_std["InF"][1]
    #         large_scale_correlations = cdl_ma.inf_large_scale_correlations
    #
    #     c_cmm_matrix, c_mm_squared = self.c_mm(large_scale_correlations)
    #
    #     num_locations = c_mm_squared.shape[0]
    #     # Generate uncorrelated random variables (Gaussian distribution)
    #     uncorrelated_random_vars = np.random.randn(num_locations)
    #     # Generate correlated random variables
    #     correlated_random_vars = np.dot(c_cmm_matrix,
    #                                     uncorrelated_random_vars)  # TODO, I have the doubt about if i must use this matrix for computing K intead of directly c_cmm_matrix
    #
    #     k_r_linear = 10 ** (k_mean / 10) + 10 ** (k_std / 10) * correlated_random_vars[0][
    #         5]  # TODO, chek all this implementation: 0 means LOS, and 5 the K-factor from the array of 0 to 6 values, the maxiz 7x7
    #     if self.o2i:
    #         ds = ds_mean + ds_std * correlated_random_vars[3][
    #             0]  # TODO, chek all this implementation: 0 means LOS, and 0 the ds-factor from the array of 0 to 6 values, the maxiz 7x7
    #     else:
    #         if self.los:
    #             ds = ds_mean + ds_std * correlated_random_vars[1][0]
    #         else:
    #             ds = ds_mean + ds_std * correlated_random_vars[2][0]
    #     # Ensure K-factors are non-negative
    #     k_r_linear = np.maximum(k_r_linear, 0)
    #     # k_r_db = 10 * np.log10(k_r_linear)
    #     return ds, k_r_linear
    #
    # def calculate_oxygen_loss(self):
    #
    #     dds = self.get_desired_delay_spread()
    #
    #     ol_n_fc = np.zeros(self.n)
    #     if not self.atmospheric_absorption:
    #         ol_n_fc = ol_n_fc
    #     else:
    #         # Speed of light
    #         c = 3e8  # m/s
    #         # Calculate OL_n(fc)
    #         if self.fc <= 52:
    #             alpha_fc = 0
    #         elif self.fc <= 53:
    #             alpha_fc = 1
    #         elif self.fc <= 54:
    #             alpha_fc = 2.2
    #         elif self.fc <= 55:
    #             alpha_fc = 4
    #         elif self.fc <= 56:
    #             alpha_fc = 6.6
    #         elif self.fc <= 57:
    #             alpha_fc = 9.7
    #         elif self.fc <= 58:
    #             alpha_fc = 12.6
    #         elif self.fc <= 59:
    #             alpha_fc = 14.6
    #         elif self.fc <= 60:
    #             alpha_fc = 15
    #         elif self.fc <= 61:
    #             alpha_fc = 14.6
    #         elif self.fc <= 62:
    #             alpha_fc = 14.3
    #         elif self.fc <= 63:
    #             alpha_fc = 10.5
    #         elif self.fc <= 64:
    #             alpha_fc = 6.8
    #         elif self.fc <= 65:
    #             alpha_fc = 3.9
    #         elif self.fc <= 66:
    #             alpha_fc = 1.9
    #         elif self.fc <= 67:
    #             alpha_fc = 1
    #         elif self.fc > 67:
    #             alpha_fc = 0
    #
    #         r_tau = self.delay_scaling()
    #         ds, k_r_linear = self.lage_scale_parameters()
    #
    #         for n in range(self.n):
    #             if self.los:
    #                 tau_delta = 0
    #             else:
    #                 x_n = np.random.uniform(0, 1)
    #                 tau_prime_n = -r_tau * ds * 1e-9 * np.log(x_n)
    #                 tau_delta = tau_prime_n
    #             ol_n_fc[n] = -(alpha_fc / 1000) * (self.d_3d + c * (dds * self.c_delay_normalized[n] + tau_delta))
    #
    #     return ol_n_fc

    # In the NLOS case, determine the LOS channel coefficient by:

    def tdl_channel_sos(self):
        ol_n_fc = self.calculate_oxygen_loss()
        K_lineal = 10**(self.tdl_rice_factors[0]/10)
        num_sinusoids = 20  # number of sinusoids to simulate the channel
        dds = self.get_desired_delay_spread()

        w_c = 2 * np.pi * self.fc
        ds_angle_rad = np.radians(self.ds_angle)
        if self.tdl_type == "A" or self.tdl_type == "B" or self.tdl_type == "C" or self.tdl_type == "D" or self.tdl_type == "E":
            fd = self.rx_speed * np.cos(ds_angle_rad) / self.lambda_0
            # fd = self.rx_speed/ self.lambda_0
        else:
            r_earth = 6371 * 1e3  # in meters
            angle = np.radians(self.v_angle)
            h = self.tx_coord[2]
            # fd = self.rx_speed * np.cos(ds_angle_rad) / self.lambda_0
            fd = self.rx_speed/ self.lambda_0 + (self.tx_speed/ self.lambda_0) * (r_earth*np.cos(angle)/(r_earth + h))

        w_d = 2 * np.pi * fd
        # Generate random phases for the sinusoids

        # NLOS contribution
        h_los = np.zeros((self.u, self.s), dtype=complex)
        h_nlos_u_s_n = np.zeros((self.u, self.s, self.n), dtype=complex)
        for u in range(self.u):
            for s in range(self.s):
                y_c_t = np.zeros(self.n, dtype=complex)
                y_s_t = np.zeros(self.n, dtype=complex)
                for n in range(self.n):

                    oxygen_loss = 10 ** (ol_n_fc[n] / 10)

                    y_c_t_ = 0
                    y_s_t_ = 0
                    for ns in range(num_sinusoids):
                        theta = np.random.uniform(-np.pi, np.pi, 1)
                        alpha = (2 * np.pi * (ns+1) + theta)/num_sinusoids
                        phi = np.random.uniform(-np.pi, np.pi, 1)
                        argument = w_d * (self.t_now - dds * self.c_delay_normalized[n]) * np.cos(alpha) + phi
                        y_c_t_ += np.cos(argument)
                        y_s_t_ += np.sin(argument)

                    # y_c_t[n] = np.sqrt(self.c_power[n] * oxygen_loss/num_sinusoids)*y_c_t_[0]
                    # y_s_t[n] = np.sqrt(self.c_power[n] * oxygen_loss/num_sinusoids)*y_s_t_[0]
                    y_c_t[n] = np.sqrt(self.c_power[n] * oxygen_loss/self.n)*y_c_t_[0]
                    y_s_t[n] = np.sqrt(self.c_power[n] * oxygen_loss/self.n)*y_s_t_[0]
                    h_nlos_u_s_n[u][s][n] = (y_c_t[n] + 1j *y_s_t[n])

                # LOS contribution
                if self.los:
                    theta = np.pi / 4  # Angle of arrival of the LOS component
                    # alpha = (2 * np.pi + theta)/num_sinusoids
                    alpha = (2 * np.pi + theta)
                    phi = np.random.uniform(-np.pi, np.pi, 1)
                    argument = w_d * self.t_now * np.cos(alpha) + phi
                    argument = alpha

                    print("np.cos(argument)", np.cos(argument))
                    print("np.sin(argument)", np.sin(argument))
                    z_c_t = (np.sum(y_c_t))/ np.sqrt(1 + K_lineal) + np.sqrt(K_lineal * oxygen_loss) * np.cos(argument)/np.sqrt(1 + K_lineal)
                    z_s_t = (np.sum(y_s_t))/ np.sqrt(1 + K_lineal) + np.sqrt(K_lineal * oxygen_loss) * np.sin(argument)/np.sqrt(1 + K_lineal)
                    h_los[u][s] = z_c_t + 1j * z_s_t
                    h = h_los
                else:
                    h = h_nlos_u_s_n

        return h

    def step_11_nlos_channel_coefficients(self):
        dds = self.get_desired_delay_spread()
        h_nlos_u_s_n = np.zeros((self.u, self.s, self.n), dtype=complex)
        for u in range(self.u):
            for s in range(self.s):
                ol_n_fc = self.calculate_oxygen_loss()
                for n in range(self.n):
                    w_c = 2 * np.pi * self.fc
                    # ds_angle_rad = np.radians(self.ds_angle)
                    ds_angle_rad = random.uniform(-np.pi, np.pi)

                    fd = self.rx_speed * np.cos(ds_angle_rad)/self.lambda_0
                    w_d = 2 * np.pi * fd
                    h = np.exp(1j * (-w_c * dds * self.c_delay_normalized[n] + w_d*(self.t_now - dds * self.c_delay_normalized[n])))
                    oxygen_loss = 10 ** (ol_n_fc[n] / 10)
                    h_nlos_n = np.sqrt(self.c_power[n] * oxygen_loss) * h
                    h_nlos_u_s_n[u][s][n] = h_nlos_n
        return h_nlos_u_s_n

    # In the LOS case, determine the LOS channel coefficient by:
    def step_11_los_channel_coefficients(self):

        if self.tdl_type == "D":
            p_0 = 10**(-0.2/10)
        elif self.tdl_type == "E":
            p_0 = 10 ** (-0.03 / 10)

        h_los_u_s_1 = np.zeros((self.u, self.s), dtype=complex)
        # h_nlos_u_s_n_m = np.zeros((self.u, self.s, self.n, self.m), dtype=complex)

        for u in range(self.u):
            for s in range(self.s):
                ol_n_fc = self.calculate_oxygen_loss()

                doppler_shift = self.rx_speed / self.lambda_0
                h = np.exp(1j * 2 * np.pi * doppler_shift * self.t_now)
                oxygen_loss = 10 ** (ol_n_fc[0] / 10)
                h_los_1 = np.sqrt(p_0 * oxygen_loss) * h
                # h_los = np.sqrt(oxygen_loss)

                # h_los_u_s_1[u][s] = h_los_1[0]
                h_los_u_s_1[u][s] = h_los_1

        return h_los_u_s_1


    def step_11_nlos_channel_impulse_response(self):

        dds = self.get_desired_delay_spread()
        h_nlos_u_s_n = self.step_11_nlos_channel_coefficients()


        # Define a time vector for tau
        dt = 1e-9  # Time resolution (1 ns)
        tau_length = 5000  # Length of the time vector
        tau = np.arange(tau_length) * dt  # Example time vector from 0 to 1000 ns

        h_nlos_u_s_tau = np.zeros((self.u, self.s, len(tau)), dtype=complex)
        for u in range(self.u):
            for s in range(self.s):
                for n in range(self.n):
                    h_component = h_nlos_u_s_n[u][s][n]
                    # print("h_component", h_component)
                    delta_n = self.delta_function(tau, dds * self.c_delay_normalized[n], dt)
                    h_nlos_u_s_tau[u, s, :] += h_component * delta_n
        # print("h_nlos_u_s_tau", h_nlos_u_s_tau)

        return h_nlos_u_s_tau

    # In the LOS case, the channel impulse response is given by: :
    def step_11_los_channel_impulse_response(self):


        dds = self.get_desired_delay_spread()
        k_r_linear = 10**(self.tdl_rice_factors[0]/10)
        h_nlos_u_s_tau = self.step_11_nlos_channel_impulse_response()
        c_ds = self.step_11_3_get_cluster_delay_spread()

        # Define a time vector for tau
        dt = 1e-9  # Time resolution (1 ns)
        tau_length = 5000  # Length of the time vector
        tau = np.arange(tau_length) * dt  # Example time vector from 0 to 1000 ns

        h_los_u_s_1 = self.step_11_los_channel_coefficients()

        h_los_u_s_tau = np.zeros((self.u, self.s, len(tau)), dtype=complex)

        for u in range(self.u):
            for s in range(self.s):

                delta_ni = self.delta_function(tau, dds * self.c_delay_normalized[0], dt)
                h_los_u_s_tau[u, s, :] = np.sqrt(1 / (k_r_linear + 1)) * h_nlos_u_s_tau[u, s, :] + (
                        np.sqrt(k_r_linear / (k_r_linear + 1)) * h_los_u_s_1[u][s] * delta_ni)

        return h_los_u_s_tau





def tdl_ff_channel_gain_v2(channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type,
                        num_rx_ax, num_tx_ax, d_2d, d_3d, bearing_angle,
                        down_tilt_angle, h_angle, v_angle, ds_angle, fc, rx_speed, speed_tx, rx_coord, tx_coord, t_now, t_old):
    tdl = TDL_models(channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type,
                     num_rx_ax, num_tx_ax,  d_2d, d_3d, bearing_angle,
                     down_tilt_angle, h_angle, v_angle, ds_angle, fc, rx_speed, speed_tx, rx_coord, tx_coord, t_now, t_old)


    channel_gain_lineal = np.zeros((num_rx_ax, num_tx_ax))
    channel_gain_db = np.zeros((num_rx_ax, num_tx_ax))

    if los:

        h_los_u_s_tau = tdl.step_11_los_channel_impulse_response()
        for u in range(num_rx_ax):
            for s in range(num_tx_ax):
                channel_gain_lineal[u][s] = np.sum(np.abs(h_los_u_s_tau[u][s]) ** 2)

                if channel_gain_lineal[u][s] == 0:
                    channel_gain_lineal[u][s] = 0.01  # TODO, I need to fix this error.
                    print("Error: Zero founded")
                channel_gain_db[u][s] = 10 * np.log10(channel_gain_lineal[u][s])

    else:
        h_nlos_u_s_tau = tdl.step_11_nlos_channel_impulse_response()

        for u in range(num_rx_ax):
            for s in range(num_tx_ax):
                channel_gain_lineal[u][s] = np.sum(np.abs(h_nlos_u_s_tau[u][s]) ** 2)
                channel_gain_db[u][s] = 10 * np.log10(channel_gain_lineal[u][s])
    return -channel_gain_db


def tdl_ff_channel_gain(channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type,
                        num_rx_ax, num_tx_ax, d_2d, d_3d, bearing_angle,
                        down_tilt_angle, h_angle, v_angle, ds_angle, fc, rx_speed, speed_tx, rx_coord, tx_coord, t_now, t_old):
    tdl = TDL_models(channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type,
                     num_rx_ax, num_tx_ax, d_2d, d_3d, bearing_angle,
                     down_tilt_angle, h_angle, v_angle, ds_angle, fc, rx_speed, speed_tx, rx_coord, tx_coord, t_now, t_old)

    channel_gain_lineal = np.zeros((num_rx_ax, num_tx_ax))
    channel_gain_db = np.zeros((num_rx_ax, num_tx_ax))
    h_los_u_s_tau = tdl.tdl_channel_sos()

    for u in range(num_rx_ax):
        for s in range(num_tx_ax):
            channel_gain_lineal[u][s] = np.sum(np.abs(h_los_u_s_tau[u][s]) ** 2)
            channel_gain_db[u][s] = 10 * np.log10(channel_gain_lineal[u][s])
    return -channel_gain_db


#
# s = 100
# s_array = np.linspace(1, s, s)  # An array of values from 0 to 20
# cg_all = np.zeros(len(s_array))
# channel_gain = 0
# for i in range(len(s_array)):
#     rms_delay = 363e-9
#     rng = None
#     ax_panel_polarization = "dual"  # "single", "dual";  single polarized (P =1) or dual polarized (P =2)
#     delta_h = 23.5
#     d_2d = 50
#     d_3d = 75
#
#     h_angle = 10
#     v_angle = 15
#
#     ds_angle = 30
#
#     bearing_angle = 0
#     down_tilt_angle = 15
#     num_rx_ax = 1
#     num_tx_ax = 1
#
#     fc = 28
#     rx_speed = 1
#     rx_coord = np.array([50, 100, 1.5])
#     tx_coord = np.array([50, 50, 25])
#
#     t_now = 1
#     t_old = 0
#
#     tx_antenna_mode = "three_sectors"  # "three_sectors"
#     rx_antenna_mode = "three_sectors"
#     channel_model = "UMi"  # UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH
#     tdl_type = "A"
#     los = False
#     o2i = False
#     atmospheric_absorption = True
#     desired_delay_spread = "Short"  # Nominal, Very long, Long delay, Short, Very short, None
#
#     cg = tdl_ff_channel_gain(channel_model, los, o2i, atmospheric_absorption,
#                              desired_delay_spread, tdl_type, num_rx_ax,
#                              num_tx_ax, d_2d, d_3d, bearing_angle,
#                              down_tilt_angle, h_angle, v_angle, ds_angle, fc, rx_speed, speed_tx, rx_coord,
#                              tx_coord, t_now, t_old)
#     channel_gain += cg[0][0]
#     cg_all[i] = cg[0][0]
#     # channel_gain += cg
#     # cg_all[i] = cg
#
# channel_gain = channel_gain / len(s_array)
#
# #
# print("Channel gain dB")
# print("channel_gain", channel_gain)
#
# import matplotlib.pyplot as plt
# # Plot the Bessel function
# # plt.ylim(-180,180)
# plt.figure(figsize=(10, 6))
# plt.plot(s_array, cg_all, label="cg_all")
# plt.title('CDL FF')
# plt.xlabel('Samples')
# plt.ylabel('Channel gain (dB)')
# plt.grid(True)
# plt.legend()
# plt.show()