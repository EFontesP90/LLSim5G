"""
File: tdl_models_old.py

Purpose:
TODO

Authors: Ernesto Fontes Pupo / Claudia Carballo González
Date: 2024-10-30
Version: 1.0.0
SPDX-License-Identifier: Apache-2.0

"""

import numpy as np
import scipy.signal
import random

import channel_models.ff_models_tr_38_901and811.cdl_matrixs as cdl_ma
import channel_models.ff_models_tr_38_901and811.tdl_matrixs as tdl_ma
import channel_models.ff_models_tr_38_901and811.polarized_field_component as p_f_c


class TDL_models(object):
    """
    25/04/2024


    Required attributes:
    ():

    """

    def __init__(self, channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type, rms_delay, rng,
                 num_rx_ax, num_tx_ax, tx_antenna_mode, rx_antenna_mode,
                 ax_panel_polarization, delta_h, d_2d, d_3d, bearing_angle,
                 down_tilt_angle, h_angle, v_angle, fc, rx_speed, rx_coord, tx_coord, t_now, t_old):

        self.channel_model = channel_model  # string with the selected link channel to be modeled from the tr_138_901: Just valid for UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH.
        self.los = los
        self.o2i = o2i
        self.atmospheric_absorption = atmospheric_absorption
        self.desired_delay_spread = desired_delay_spread

        self.tdl_type = tdl_type  # nlos: A, B, C; los: C, D
        self.rms_delay = rms_delay  #
        self._rng = rng

        self.cp = tdl_ma.tdl_cp[self.tdl_type]
        self.c_delay_normalized = self.cp[:, 0]
        print("self.c_delay_normalized", self.c_delay_normalized)
        self.c_power = 10 ** (self.cp[:, 1] / 10)
        print("self.cp[:, 1]", self.cp[:, 1])
        # self.c_aod = self.cp[:, 2]
        # self.c_aoa = self.cp[:, 3]
        # self.c_zod = self.cp[:, 4]
        # self.c_zoa = self.cp[:, 5]

        self.pcp = cdl_ma.cdl_pcp[self.tdl_type]
        # self.rms_asd_spreads = self.pcp[0]
        # self.rms_asa_spreads = self.pcp[1]
        # self.rms_zsd_spreads = self.pcp[2]
        # self.rms_zsa_spreads = self.pcp[3]
        self.xpr_db = self.pcp[4]
        # self.los_model = self.pcp[5]
        #
        # self.roa = cdl_ma.ray_offset_angles
        self.tdl_rice_factors = tdl_ma.tdl_rice_factors[self.tdl_type]

        self.n = self.cp.shape[0]  # number of clusters 𝑁
        print("self.n", self.n)
        # self.m = self.roa.size  # number of rays per cluster
        self.u = num_rx_ax  # Number of receiver antenna elements
        self.s = num_tx_ax  # Number of transmitter antenna elements

        self.tx_antenna_mode = "omni"
        self.ax_panel_polarization = ax_panel_polarization  # "single", "dual";  single polarized (P =1) or dual polarized (P =2)
        self.rx_antenna_mode = "omni"

        self.delta_h = delta_h  # difference between the h of the tx and he rx
        self.d_2d = d_2d
        self.d_3d = d_3d

        self.bearing_angle = bearing_angle
        self.down_tilt_angle = down_tilt_angle
        self.v_tilt = self.down_tilt_angle
        # self.slant_angle = np.degrees(np.arctan(self.delta_h / self.d_2d))

        self.v_angle = v_angle
        self.h_angle = h_angle

        self.fc = fc
        self.lambda_0 = 300000000 / (self.fc * 1e9)

        self.rx_speed = rx_speed
        self.rx_coord = rx_coord
        self.tx_coord = tx_coord

        self.t_now = t_now
        self.t_old = t_old

    #   Step 1: Generate departure and arrival angles
    # def step_1_get_d_a_angles(self):
    #     # Generate cluster delays
    #     c_delays = self.rms_delay * self.c_delay_normalized
    #
    #     aod_array = np.add.outer(self.c_aod, self.rms_asd_spreads * self.roa)
    #     aoa_array = np.add.outer(self.c_aoa, self.rms_asa_spreads * self.roa)
    #     zod_array = np.add.outer(self.c_zod, self.rms_zsd_spreads * self.roa)
    #     zoa_array = np.add.outer(self.c_zoa, self.rms_zsa_spreads * self.roa)
    #
    #     return aod_array, aoa_array, zod_array, zoa_array

    # def step_2_get_angle_coupling_indices(self, aod_array, aoa_array, zod_array, zoa_array):
    #
    #     angle_candidate_indices = np.arange(self.m)
    #     # Generate permuted indices
    #     angle_coupling_indices = np.array(
    #         [
    #             [
    #                 self._rng.permutation(angle_candidate_indices)
    #                 for _ in range(self.n)
    #             ]
    #             for _ in range(4)
    #         ]
    #     )
    #
    #     # Step 2: Coupling of rays within a cluster for both azimuth and zenith
    #     # Equation 7.7-0b in ETSI TR 138.901 v17.0.0
    #     shuffled_aod_array = np.take_along_axis(
    #         aod_array, angle_coupling_indices[0, :], axis=1
    #     )
    #     shuffled_aoa_array = np.take_along_axis(
    #         aoa_array, angle_coupling_indices[1, :], axis=1
    #     )
    #     shuffled_zod_array = np.take_along_axis(
    #         zod_array, angle_coupling_indices[2, :], axis=1
    #     )
    #     shuffled_zoa_array = np.take_along_axis(
    #         zoa_array, angle_coupling_indices[3, :], axis=1
    #     )
    #
    #     return shuffled_aod_array, shuffled_aoa_array, shuffled_zod_array, shuffled_zoa_array

    # Step 3: Generate the cross polarization power ratios
    # Equation 7.7-0b in ETSI TR 138.901 v17.0.0
    def step_3_get_cross_polarization_power_ratios(
            self):  # TODO, I have doubs with this step, look to simple, with the same valu for all n x m
        xpr_factor = 10 ** (self.xpr_db / 10)
        xpr_factor_n = np.full(self.n, xpr_factor)

        return xpr_factor_n

    # Step 4 Coefficient generation: Follow the same procedure as in Steps 10 and 11 in Clause 7.5
    def step_10_draw_initial_random_phases(self):
        phases = np.random.uniform(-np.pi, np.pi, (4, self.n))

        polarizations = {
            'θθ': 0,
            'θϕ': 1,
            'ϕθ': 2,
            'ϕϕ': 3
        }

        return phases, polarizations

    def step_11_1_spherical_unit_vector_arrival(self, theta, phi):
        """
        Compute the spherical unit vector given azimuth  angle (phi) and zenith angle (theta).

        Args:
            theta (float): Zenith angle.
            phi (float): Azimuth angle.

        Returns:
            np.ndarray: The spherical unit vector.
        """
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

    def step_11_2_get_polarized_field_component_los(self, antenna_mode, theta, phi):
        pfc_ = p_f_c.pfc_los(antenna_mode, phi, theta, self.ax_panel_polarization,
                         self.delta_h, self.d_2d, self.bearing_angle, self.down_tilt_angle)
        return pfc_

    def step_11_2_get_polarized_field_component_nlos(self, antenna_mode, theta, phi):
        pfc_ = p_f_c.pfc_nlos(antenna_mode, phi, theta, self.ax_panel_polarization,
                         self.delta_h, self.d_2d, self.bearing_angle, self.down_tilt_angle)
        return pfc_

    def step_11_3_get_cluster_delay_spread(
            self):  # valid for UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH
        c_ds = 3.91 * 1e-9
        if self.channel_model == "UMi" or self.channel_model == "A2G":
            if self.o2i:
                c_ds = 11 * 1e-9
            else:
                if self.los:
                    c_ds = 5 * 1e-9
                else:
                    c_ds = 11 * 1e-9

        elif self.channel_model == "UMa":
            if self.o2i:
                c_ds = 11 * 1e-9
            else:
                if self.los:
                    c_ds = max(0.25 * 1e-9, (6.5622 - 3.4084 * np.log10(self.fc)) * 1e-9)
                else:
                    c_ds = max(0.25 * 1e-9, (6.5622 - 3.4084 * np.log10(self.fc)) * 1e-9)

        return c_ds

    # Function to create a delta function using scipy.signal.unit_impulse
    def delta_function(self, tau, center, dt):
        index = int(center / dt)
        return scipy.signal.unit_impulse(len(tau), index)

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
            dds = 1

        return dds

    def c_mm(self, large_scale_correlations):
        c_cmm_matrix = np.empty((3, 7, 7), dtype=np.float_)
        for c, C in enumerate(large_scale_correlations):
            c_mm_squared = np.array(
                [
                    #    DS,    ASD,   ASA,   ZSA,   ZSD,   K,     SF
                    [1.0, C[0], C[1], C[15], C[14], C[8], C[4]],  # DS
                    [C[0], 1.0, C[5], C[17], C[16], C[6], C[3]],  # ASD
                    [C[1], C[5], 1.0, C[19], C[18], C[7], C[2]],  # ASA
                    [C[15], C[17], C[19], 1.0, C[20], C[13], C[11]],  # ZSA
                    [C[14], C[16], C[18], C[20], 1.0, C[12], C[10]],  # ZSD
                    [C[8], C[6], C[7], C[13], C[12], 1.0, C[9]],  # K
                    [C[4], C[3], C[2], C[11], C[10], C[9], 1.0],  # SF
                ],
                dtype=np.float_,
            )

            # Section 4 of ETSI TR 138.901 v17.0.0 hints at using the cholesky decomposition
            # to enforce the expected cross-correlations between the large-scale parameters
            c_cmm_matrix[c, ...] = np.linalg.cholesky(c_mm_squared)

        return c_cmm_matrix, c_mm_squared

    def lage_scale_parameters(self):

        ds_mean_std = cdl_ma.ds_mean_std(self.channel_model, self.fc, self.los, self.o2i)
        ds_mean = ds_mean_std[0]
        ds_std = ds_mean_std[1]

        if self.channel_model == "UMi" or self.channel_model == "A2G" or self.channel_model == "D2D":
            k_mean = cdl_ma.k_mean_std["UMi"][0]
            k_std = cdl_ma.k_mean_std["UMi"][1]
            large_scale_correlations = cdl_ma.umi_large_scale_correlations
        elif self.channel_model == "UMa":
            k_mean = cdl_ma.k_mean_std["UMa"][0]
            k_std = cdl_ma.k_mean_std["UMa"][1]
            large_scale_correlations = cdl_ma.uma_large_scale_correlations
        elif self.channel_model == "RMa":
            k_mean = cdl_ma.k_mean_std["RMa"][0]
            k_std = cdl_ma.k_mean_std["RMa"][1]
            large_scale_correlations = cdl_ma.rma_large_scale_correlations
        elif self.channel_model == "InH-Mixed" or self.channel_model == "InH-Open":
            k_mean = cdl_ma.k_mean_std["InH"][0]
            k_std = cdl_ma.k_mean_std["InH"][1]
            large_scale_correlations = cdl_ma.inh_large_scale_correlations
        else:
            k_mean = cdl_ma.k_mean_std["InF"][0]
            k_std = cdl_ma.k_mean_std["InF"][1]
            large_scale_correlations = cdl_ma.inf_large_scale_correlations

        c_cmm_matrix, c_mm_squared = self.c_mm(large_scale_correlations)

        num_locations = c_mm_squared.shape[0]
        # Generate uncorrelated random variables (Gaussian distribution)
        uncorrelated_random_vars = np.random.randn(num_locations)
        # Generate correlated random variables
        correlated_random_vars = np.dot(c_cmm_matrix,
                                        uncorrelated_random_vars)  # TODO, I have the doubt about if i must use this matrix for computing K intead of directly c_cmm_matrix

        k_r_linear = 10 ** (k_mean / 10) + 10 ** (k_std / 10) * correlated_random_vars[0][
            5]  # TODO, chek all this implementation: 0 means LOS, and 5 the K-factor from the array of 0 to 6 values, the maxiz 7x7
        if self.o2i:
            ds = ds_mean + ds_std * correlated_random_vars[3][
                0]  # TODO, chek all this implementation: 0 means LOS, and 0 the ds-factor from the array of 0 to 6 values, the maxiz 7x7
        else:
            if self.los:
                ds = ds_mean + ds_std * correlated_random_vars[1][0]
            else:
                ds = ds_mean + ds_std * correlated_random_vars[2][0]
        # Ensure K-factors are non-negative
        k_r_linear = np.maximum(k_r_linear, 0)
        # k_r_db = 10 * np.log10(k_r_linear)
        return ds, k_r_linear

    def calculate_oxygen_loss(self):

        ol_n_fc = np.zeros(self.n)
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
            ds, k_r_linear = self.lage_scale_parameters()

            for n in range(self.n):
                if self.los:
                    tau_delta = 0
                else:
                    x_n = np.random.uniform(0, 1)
                    tau_prime_n = -r_tau * ds * 1e-9 * np.log(x_n)
                    tau_delta = tau_prime_n
                ol_n_fc[n] = -(alpha_fc / 1000) * (self.d_3d + c * (self.c_delay_normalized[n] * 1e-9 + tau_delta))

        return ol_n_fc

    # In the NLOS case, determine the LOS channel coefficient by:
    def step_11_nlos_channel_coefficients(self, xpr_factor_n):

        phases, polarizations = self.step_10_draw_initial_random_phases()

        # h_nlos_u_s_n_ = np.zeros((self.u, self.s, self.n), dtype=complex)
        h_nlos_u_s_n = np.zeros((self.u, self.s, self.n), dtype=complex)

        for u in range(self.u):
            for s in range(self.s):
                ol_n_fc = self.calculate_oxygen_loss()

                for n in range(self.n):

                    theta_rx = 0
                    phi_rx = 0
                    pfc_rx = self.step_11_2_get_polarized_field_component_nlos(self.rx_antenna_mode, theta_rx, phi_rx)
                    print("pfc_rxtest", pfc_rx)
                    pfc_rx = np.array([0.70710678, 0.70710678])
                    print("pfc_rx", pfc_rx)

                    theta_tx = 0
                    phi_tx = 0
                    pfc_tx = self.step_11_2_get_polarized_field_component_nlos(self.tx_antenna_mode, theta_tx, phi_tx)
                    print("pfc_txtest", pfc_tx)
                    pfc_tx = np.array([0.70710678, 0.70710678])
                    print("pfc_tx", pfc_tx)


                    phase_exp_mat = np.array([[np.exp(1j * phases[polarizations['θθ']][n]),
                                               np.sqrt(xpr_factor_n[n] ** -1) * np.exp(
                                                   1j * phases[polarizations['θϕ']][n])],
                                              [np.sqrt(xpr_factor_n[n] ** -1) * np.exp(
                                                  1j * phases[polarizations['ϕθ']][n]),
                                               np.exp(1j * phases[polarizations['ϕϕ']][n])]])

                    # phase_exp_mat = np.array([[1, 0],
                    #                           [0, -1]])

                    d_rx_u = self.rx_coord
                    d_tx_s = self.tx_coord
                        #  v_vec_ = np.array([[np.sin(theta_rx) * np.cos(phi_rx)], [np.sin(theta_rx) * np.sin(phi_rx)], [np.cos(theta_rx)]])
                    v_vec_ = np.array(
                        [np.sin(theta_rx) * np.cos(phi_rx), np.sin(theta_rx) * np.sin(phi_rx), np.cos(theta_rx)])
                    v_vec_ = np.array(
                        [1, 1, 1])
                    v_vec = np.dot(self.rx_speed, v_vec_.T)


                    r_rx_n = self.step_11_1_spherical_unit_vector_arrival(theta_rx, phi_rx)
                    r_tx_n = self.step_11_1_spherical_unit_vector_arrival(theta_tx, phi_tx)

                    term1 = np.exp(1j * 2 * np.pi * np.dot(r_rx_n.T, d_rx_u) / self.lambda_0)
                    term2 = np.exp(1j * 2 * np.pi * np.dot(r_tx_n.T, d_tx_s) / self.lambda_0)
                    term3 = np.exp(1j * 2 * np.pi * np.dot(r_rx_n.T, v_vec) * self.t_now / self.lambda_0)
                    result = term1 * term2 * term3
                    print("term1", term1)
                    print("term2", term2)
                    print("term3", term3)
                    print("result", result)
                    print("phase_exp_mat", phase_exp_mat)

                        # h_nlos_u_s_n_m[u][s][n][m] = (np.sqrt(self.c_power[n] / self.m) * (
                        #         np.dot(pfc_rx.T, phase_exp_mat) * np.dot(pfc_tx, result)))[0]
                    h_nlos = (np.dot(pfc_rx.T, phase_exp_mat) * np.dot(pfc_tx, result))
                    oxygen_loss = 10 ** (ol_n_fc[n] / 10)
                    h_nlos_n = np.sqrt(self.c_power[n] * oxygen_loss) * h_nlos
                    print("h_nlos_n", h_nlos_n)
                    # h_nlos_n = np.sqrt(self.c_power[n] * oxygen_loss)
                    h_nlos_u_s_n[u][s][n] = h_nlos_n[0]
                    # h_nlos_u_s_n[u][s][n] = h_nlos_n

        return h_nlos_u_s_n

    # In the LOS case, determine the LOS channel coefficient by:
    def step_11_los_channel_coefficients(self):

        h_los_u_s_1 = np.zeros((self.u, self.s), dtype=complex)
        # h_nlos_u_s_n_m = np.zeros((self.u, self.s, self.n, self.m), dtype=complex)

        for u in range(self.u):
            for s in range(self.s):
                ol_n_fc = self.calculate_oxygen_loss()

                # theta_rx = shuffled_zoa_array[0][0]
                # phi_rx = shuffled_aoa_array[0][0]
                theta_rx = 0
                phi_rx = 0
                pfc_rx = self.step_11_2_get_polarized_field_component_los(self.rx_antenna_mode, theta_rx, phi_rx)
                pfc_rx = np.array([1, 1])

                # theta_tx = shuffled_zod_array[0][0]
                # phi_tx = shuffled_aod_array[0][0]
                theta_tx = 0
                phi_tx = 0
                pfc_tx = self.step_11_2_get_polarized_field_component_los(self.tx_antenna_mode, theta_tx, phi_tx)
                pfc_tx = np.array([1, 1])

                phase_exp_mat = np.array([[1, 0],
                                          [0, -1]])

                d_rx_u = self.rx_coord
                d_tx_s = self.tx_coord
                #  v_vec_ = np.array([[np.sin(theta_rx) * np.cos(phi_rx)], [np.sin(theta_rx) * np.sin(phi_rx)], [np.cos(theta_rx)]])
                v_vec_ = np.array(
                    [np.sin(theta_rx) * np.cos(phi_rx), np.sin(theta_rx) * np.sin(phi_rx), np.cos(theta_rx)])
                v_vec = np.dot(self.rx_speed, v_vec_.T)

                r_rx_n = self.step_11_1_spherical_unit_vector_arrival(theta_rx, phi_rx)
                r_tx_n = self.step_11_1_spherical_unit_vector_arrival(theta_tx, phi_tx)

                term1 = np.exp(1j * 2 * np.pi * np.dot(r_rx_n.T, d_rx_u) / self.lambda_0)
                term2 = np.exp(1j * 2 * np.pi * np.dot(r_tx_n.T, d_tx_s) / self.lambda_0)
                term3 = np.exp(1j * 2 * np.pi * np.dot(r_rx_n.T, v_vec) * self.t_now / self.lambda_0)
                result = term1 * term2 * term3

                oxygen_loss = 10 ** (ol_n_fc[0] / 10)
                h_los = np.sqrt(oxygen_loss) * ((np.dot(pfc_rx.T, phase_exp_mat) * np.dot(pfc_tx, result)))
                # h_los = np.sqrt(oxygen_loss)

                h_los_u_s_1[u][s] = h_los[0]
                # h_los_u_s_1[u][s] = h_los

        return h_los_u_s_1


    def step_11_nlos_channel_impulse_response(self, xpr_factor_n):

        dds = self.get_desired_delay_spread()

        h_nlos_u_s_n = self.step_11_nlos_channel_coefficients(xpr_factor_n)

        c_ds = self.step_11_3_get_cluster_delay_spread()
        # Define R_i based on the table provided
        R_i = [
            [1, 2, 3, 4, 5, 6, 7, 8, 19, 20],
            [9, 10, 11, 12, 17, 18],
            [13, 14, 15, 16]
        ]
        power_i = [10 / 20, 6 / 20, 4 / 20]
        delay_offset = [0, 1.28 * c_ds, 2.56 * c_ds]

        # tau_n = np.random.rand(self.n) * c_ds  # Example delay offsets for clusters

        # Define a time vector for tau
        dt = 1e-9  # Time resolution (1 ns)
        tau_length = 100  # Length of the time vector
        tau = np.arange(tau_length) * dt  # Example time vector from 0 to 1000 ns

        h_nlos_u_s_tau = np.zeros((self.u, self.s, len(tau)), dtype=complex)
        for u in range(self.u):
            for s in range(self.s):
                # for n in range(2):  # for n = 0 to 1 (intra-cluster delay spread)
                #     for i in range(3):  # for i = 1 to 3
                #         for m in R_i[i]:
                #
                #             delta_ni = self.delta_function(tau, self.c_delay_normalized[n]*1e-9 + delay_offset[i], dt)
                #
                #             h_component = h_nlos_u_s_n_m[u][s][n][m-1]
                #
                #             h_nlos_u_s_tau[u, s, :] += h_component * delta_ni

                for n in range(self.n):
                    h_component = h_nlos_u_s_n[u][s][n]
                    print("h_component", h_component)
                    delta_n = self.delta_function(tau, dds * self.c_delay_normalized[n] * 1e-9, dt)
                    h_nlos_u_s_tau[u, s, :] += h_component * delta_n

        return h_nlos_u_s_tau

    # In the LOS case, the channel impulse response is given by: :
    def step_11_los_channel_impulse_response(self, xpr_factor_n):

        dds = self.get_desired_delay_spread()


        # ds, k_r_linear = self.lage_scale_parameters()
        k_r_linear = 10**(self.tdl_rice_factors[0]/10)
        print("k_r_dB", self.tdl_rice_factors[0])
        print("k_r_linear", k_r_linear )

        h_nlos_u_s_tau = self.step_11_nlos_channel_impulse_response(xpr_factor_n)
        #
        # channel_gainkr = np.sum(np.abs(np.sqrt(1 / (k_r_linear + 1)) * h_nlos_u_s_tau[0][0]) ** 2)
        # channel_gain = np.sum(np.abs(h_nlos_u_s_tau[0][0]) ** 2)
        # print("nols channel gain contribution:", 10*np.log10(channel_gain))
        # print("nols channel gain contribution with Kr:", 10 * np.log10(channel_gainkr))

        # Compute H_LOS_u_s(t, tau)
        c_ds = self.step_11_3_get_cluster_delay_spread()

        # Define a time vector for tau
        dt = 1e-9  # Time resolution (1 ns)
        tau_length = 100  # Length of the time vector
        tau = np.arange(tau_length) * dt  # Example time vector from 0 to 1000 ns

        h_los_u_s_1 = self.step_11_los_channel_coefficients()

        h_los_u_s_tau = np.zeros((self.u, self.s, len(tau)), dtype=complex)

        for u in range(self.u):
            for s in range(self.s):
                # Contribution from NLOS component
                # h_los_u_s_tau[u, s, :] = np.sqrt(1 / (k_r_linear + 1)) * h_nlos_u_s_tau[u, s, :]

                # Contribution from LOS component
                delta_ni = self.delta_function(tau, dds * self.c_delay_normalized[0] * 1e-9, dt)
                print("h_nlos_u_s_tau", h_nlos_u_s_tau[u, s, :])
                print("h_los_u_s_1", h_los_u_s_1[u][s])

                h_los_u_s_tau[u, s, :] = np.sqrt(1 / (k_r_linear + 1)) * h_nlos_u_s_tau[u, s, :] + (
                        np.sqrt(k_r_linear / (k_r_linear + 1)) * h_los_u_s_1[u][s] * delta_ni)

        return h_los_u_s_tau


def tdl_ff_channel_coefficients(channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type,
                                rms_delay, rng, num_rx_ax, num_tx_ax, tx_antenna_mode, rx_antenna_mode,
                                ax_panel_polarization, delta_h, d_2d, d_3d, bearing_angle,
                                down_tilt_angle, h_angle, v_angle, fc, rx_speed, rx_coord, tx_coord, t_now, t_old):
    tdl = TDL_models(channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type, rms_delay, rng,
                     num_rx_ax, num_tx_ax, tx_antenna_mode, rx_antenna_mode,
                     ax_panel_polarization, delta_h, d_2d, d_3d, bearing_angle,
                     down_tilt_angle, h_angle, v_angle, fc, rx_speed, rx_coord, tx_coord, t_now, t_old)

    # aod_array, aoa_array, zod_array, zoa_array = cdl.step_1_get_d_a_angles()
    # shuffled_aod_array, shuffled_aoa_array, shuffled_zod_array, shuffled_zoa_array = cdl.step_2_get_angle_coupling_indices(
    #     aod_array, aoa_array, zod_array, zoa_array)
    xpr_factor = tdl.step_3_get_cross_polarization_power_ratios()
    if los:
        h_los_u_s_1 = tdl.step_11_los_channel_coefficients()
        channel_coefficients = h_los_u_s_1
    else:
        h_nlos_u_s_n, h_nlos_u_s_n_m = tdl.step_11_nlos_channel_coefficients(xpr_factor)
        channel_coefficients = h_nlos_u_s_n
    return channel_coefficients


def tdl_ff_channel_impulse_response(channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type,
                                    rms_delay, rng, num_rx_ax, num_tx_ax, tx_antenna_mode, rx_antenna_mode,
                                    ax_panel_polarization, delta_h, d_2d, d_3d, bearing_angle,
                                    down_tilt_angle, h_angle, v_angle, fc, rx_speed, rx_coord, tx_coord, t_now, t_old):
    tdl = TDL_models(channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type, rms_delay, rng,
                     num_rx_ax, num_tx_ax, tx_antenna_mode, rx_antenna_mode,
                     ax_panel_polarization, delta_h, d_2d, d_3d, bearing_angle,
                     down_tilt_angle, h_angle, v_angle, fc, rx_speed, rx_coord, tx_coord, t_now, t_old)

    xpr_factor = tdl.step_3_get_cross_polarization_power_ratios()

    if los:
        h_los_u_s_tau = tdl.step_11_los_channel_impulse_response(xpr_factor)
        channel_impulse_response = h_los_u_s_tau
    else:
        h_nlos_u_s_tau = tdl.step_11_nlos_channel_impulse_response(xpr_factor)
        channel_impulse_response = h_nlos_u_s_tau
    return channel_impulse_response


def tdl_ff_channel_gain(channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type, rms_delay, rng,
                        num_rx_ax, num_tx_ax, tx_antenna_mode, rx_antenna_mode,
                        ax_panel_polarization, delta_h, d_2d, d_3d, bearing_angle,
                        down_tilt_angle, h_angle, v_angle, fc, rx_speed, rx_coord, tx_coord, t_now, t_old):
    tdl = TDL_models(channel_model, los, o2i, atmospheric_absorption, desired_delay_spread, tdl_type, rms_delay, rng,
                     num_rx_ax, num_tx_ax, tx_antenna_mode, rx_antenna_mode,
                     ax_panel_polarization, delta_h, d_2d, d_3d, bearing_angle,
                     down_tilt_angle, h_angle, v_angle, fc, rx_speed, rx_coord, tx_coord, t_now, t_old)

    # aod_array, aoa_array, zod_array, zoa_array = tdl.step_1_get_d_a_angles()
    # shuffled_aod_array, shuffled_aoa_array, shuffled_zod_array, shuffled_zoa_array = cdl.step_2_get_angle_coupling_indices(
    #     aod_array, aoa_array, zod_array, zoa_array)
    xpr_factor = tdl.step_3_get_cross_polarization_power_ratios()

    channel_gain_lineal = np.zeros((num_rx_ax, num_tx_ax))
    channel_gain_db = np.zeros((num_rx_ax, num_tx_ax))

    if los:

        h_los_u_s_tau = tdl.step_11_los_channel_impulse_response(xpr_factor)
        for u in range(num_rx_ax):
            for s in range(num_tx_ax):
                channel_gain_lineal[u][s] = np.sum(np.abs(h_los_u_s_tau[u][s]) ** 2)

                if channel_gain_lineal[u][s] == 0:
                    channel_gain_lineal[u][s] = 0.01  # TODO, I need to fix this error.
                    print("Error: Zero founded")
                channel_gain_db[u][s] = 10 * np.log10(channel_gain_lineal[u][s])

    else:
        h_nlos_u_s_tau = tdl.step_11_nlos_channel_impulse_response(xpr_factor)

        for u in range(num_rx_ax):
            for s in range(num_tx_ax):
                channel_gain_lineal[u][s] = np.sum(np.abs(h_nlos_u_s_tau[u][s]) ** 2)
                channel_gain_db[u][s] = 10 * np.log10(channel_gain_lineal[u][s])
    return channel_gain_db


rms_delay = 363e-9
# rng = np.random.default_rng()




ax_panel_polarization = "dual"  # "single", "dual";  single polarized (P =1) or dual polarized (P =2)
delta_h = 23.5
d_2d = 50
d_3d = 70

h_angle = 10
v_angle = 15
v_tilt = 15

bearing_angle = 0
down_tilt_angle = 15
num_rx_ax = 1
num_tx_ax = 1

fc = 28
rx_speed = 1000
rx_coord = np.array([50, 100, 1.5])
tx_coord = np.array([50, 50, 25])

t_now = 1
t_old = 0

tx_antenna_mode = "three_sectors"  # "three_sectors"
rx_antenna_mode = "omni"
channel_model = "UMi"  # UMa, UMi, RMa, A2G, InH-Mixed, InH-Open, InF-HH, InF-SL, InF-DL, InF-SH, InF-DH
tdl_type = "A"
los = False
o2i = False
atmospheric_absorption = False
desired_delay_spread = "None"  # Nominal, Very long, Long delay, Short, Very short, None

s = 100
s_array = np.linspace(1, s, s)  # An array of values from 0 to 20
cg_all = np.zeros(len(s_array))
channel_gain = 0
for i in range(len(s_array)):
    cg = tdl_ff_channel_gain(channel_model, los, o2i, atmospheric_absorption,
                             desired_delay_spread, tdl_type, rms_delay, rng, num_rx_ax,
                             num_tx_ax, tx_antenna_mode, rx_antenna_mode,
                             ax_panel_polarization, delta_h, d_2d, d_3d, bearing_angle,
                             down_tilt_angle, h_angle, v_angle, fc, rx_speed, rx_coord,
                             tx_coord, t_now, t_old)
    channel_gain += cg[0][0]
    cg_all[i] = cg[0][0]

channel_gain = channel_gain / len(s_array)

#
print("Channel gain dB")
print("channel_gain", channel_gain)

import matplotlib.pyplot as plt
# Plot the Bessel function
# plt.ylim(-180,180)
plt.figure(figsize=(10, 6))
plt.plot(s_array, cg_all, label="cg_all")
plt.title('CDL FF')
plt.xlabel('Samples')
plt.ylabel('Channel gain (dB)')
plt.grid(True)
plt.legend()
plt.show()