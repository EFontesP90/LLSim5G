"""
File: scenario_definition.py

Purpose:
This file defines the inputs of the enabled links to compute between the BSs and EDs. It processes the outputs of the
Scenario class for executing the nested loops over the number of BSs, the simulation time steps and the number uf EDs.
Its outputs are the main outputs of the simulator (in a dictionary form): SINR, BLER, and CQI for each enabled link
computation.


Author: Ernesto Fontes Pupo / Claudia Carballo González
Date: 2024-10-30
Version: 1.0.0
SPDX-License-Identifier: Apache-2.0

"""

# Third-party imports
import numpy as np
import pandas as pd

# Local application/library-specific imports
import scenario.scenario_definition as sx
from channel_models.geometry import geometry as gm
from channel_models.geometry import geometry_ntn as gm_ntn
from channel_models import channel_model_tr_38_901 as ch_38_901
from channel_models import channel_model_tr_38_811 as ch_38_811
from link_to_system_adaptation import bler_curves as bl
from link_to_system_adaptation import link_to_system as l2s



def ColumnName(df):  # para un df sin titulos
    name = np.empty([df.shape[1]], dtype=int)  # array of the resulting CQI of the K users respect to the BS and each proximity user in D2D comm
    for i in range(df.shape[1]):
        # name[i] = f"{i}"
        if i == 0:
            name[i] = 0
        else:
            name[i] = i

    df.columns = name
    return df


class Link_computation(object):
    """
    25/04/2024
    The class Link_computation defines the inputs of the enabled links to compute between the BSs and EDs. It processes
    the outputs of the Scenario class for executing the nested loops over the number of BSs, the simulation time steps
    and the number uf EDs.
    Its outputs are the main outputs of the simulator (in a dictionary form): SINR, BLER, and CQI for each enabled link
    computation.

    Required attributes:
    (bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                 df_z, time_map, grid_lla, grid_xy):

    Returns (link_computation_bs2d_dl, link_computation_bs2d_ul, link_computation_d2d ):
        metrics_dic_bs_dl: Dictionary with the SINR, BLER, and CQI for the downlink of each BS (TN/NTN) regarding each ED.
        metrics_dic_d2d: Dictionary with the SINR, BLER, and CQI among the EDs that has enabled the D2D communication.
        metrics_dic_bs_ul: Dictionary with the SINR, BLER, and CQI for the uplink of each BS (TN/NTN) regarding each ED.

    """

    def __init__(self, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                 df_z, time_map, grid_lla, grid_xy):

        self.bs_parameters = bs_parameters  # [number of bs]x["x", "y", "z", "type", "scenario", "antenna_mode", "fc", "numerology", "n_rb", "p_tx", "ax_gain", "cable_loss", "noise_figure"]
        self.general_channel_modeling = general_channel_modeling  # 1x["dynamic_los", "dynamic_hb", "o2i", "inside_what_o2i", "penetration_loss_model", "shadowing", "fast_fading", "fast_fading_model"]
        self.sub_groups_parameters = sub_groups_parameters  # [number of defined user subgroups]x["type", "k_sub", "antenna_mode", "p_tx", "ax_gain", "cable_loss", "noise_figure", "d2d", "fixed_height", "grid_size_ratio", "reference_location", "min_max_velocity", "wait_time", "mobility_model", "aggregation", "number_mg_rpg_model", "min_max_height"]
        self.general_parameters = general_parameters  # 1x["thermal_noise", "h_ceiling", "block_density"]

        self.df_x = df_x  # array of [time_steps]x[number of users], with the X coordinates of each user over time
        self.df_y = df_y  # array of [time_steps]x[number of users], with the Y coordinates of each user over time
        self.df_z = df_z  # array of [time_steps]x[number of users], with the Z coordinates of each user over time
        self.time_map = time_map
        self.grid_lla = grid_lla
        self.grid_xy = grid_xy


    def link_computation_bs2d_dl(self):

        number_bs = self.bs_parameters.shape[0]

        k = self.sub_groups_parameters['k_sub'].sum()
        number_sub = self.sub_groups_parameters.shape[0]
        sub_members = self.sub_groups_parameters['k_sub'].values
        cumulative_sub_members = np.cumsum(sub_members)
        sub_types = self.sub_groups_parameters['type'].values
        time_steps = self.df_x.shape[0]
        overall_metrics = []

        for bs in range(number_bs):
            type = self.bs_parameters["type"][bs]
            tx_coord = [self.bs_parameters["x"][bs], self.bs_parameters["y"][bs], self.bs_parameters["z"][bs]]
            tx_coord_old = tx_coord  # TODO: To be taken into account when we add the abs movement capability
            bw_rb = 12*(15*(2**self.bs_parameters["numerology"][bs]))*1e3
            n_rb = self.bs_parameters["n_rb"][bs]
            cable_loss_tx = self.bs_parameters["cable_loss"][bs]
            desired_elevation_angle = self.bs_parameters["desired_elevation_angle"][bs]

            if self.bs_parameters["fc"][bs] <= 6: f_band_rx = "S-band"     # "S-band", "Ka-band"
            elif self.bs_parameters["fc"][bs] > 6: f_band_rx = "Ka-band"  # "S-band", "Ka-band"

            overall_d_correlation_map_rx = []
            overall_hb_map_rx = []
            overall_jakes_map = []

            cqi_txk = np.zeros([time_steps, k], dtype=float)
            sinr_txk = np.zeros([time_steps, k], dtype=float)
            bler_txk = np.zeros([time_steps, k], dtype=float)
            metrics_dic_bs = {"sinr": None, "cqi": None, "bler": None}

            for t in range(time_steps):
                t_now = self.df_x[0][t]
                if t == 0:  t_old = t_now
                else: t_old = self.df_x[0][t - 1]

                for i in range(k):
                    rx_coord = [self.df_x[i + 1][t], self.df_y[i + 1][t], self.df_z[i + 1][t]]
                    if t == 0: rx_coord_old = rx_coord
                    else: rx_coord_old = [self.df_x[i + 1][t - 1], self.df_y[i + 1][t - 1], self.df_z[i + 1][t - 1]]

                    for s in range(number_sub):
                        if i <= cumulative_sub_members[s]:
                            type_rx = self.sub_groups_parameters["type"][s]
                            noise_figure_rx = self.sub_groups_parameters["noise_figure"][s]
                            antenna_mode_rx = self.sub_groups_parameters["antenna_mode"][s]
                            p_tx_rx = self.sub_groups_parameters["p_tx"][s]
                            ax_gain_rx = self.sub_groups_parameters["ax_gain"][s]
                            cable_loss_rx = self.sub_groups_parameters["cable_loss"][s]
                            d2d_rx = self.sub_groups_parameters["d2d"][s]
                            rx_scenario = self.sub_groups_parameters["rx_scenario"][s]
                    if type_rx == "car_mounted": inside_what = "car"
                    else: inside_what = self.general_channel_modeling["inside_what_o2i"][0]

                    if t == 0:
                        d_correlation_map_rx = {"t": None, "x": None, "y": None, "shadowing": None, "o2i_loss": None,
                                                "los": None, "d_correlation_sf": None}
                        hb_map_rx = {"t": None, "x": None, "y": None, "h_blockage": None}
                        jakes_map = np.zeros([n_rb * 6, 3], dtype=float)

                    else:
                        d_correlation_map_rx = overall_d_correlation_map_rx[i]
                        hb_map_rx = overall_hb_map_rx[i]
                        jakes_map = overall_jakes_map[i]

                    # print("type", type)

                    if type == "tbs" or type == "abs":
                        d_2d, d_3d, speed_rx, speed_tx, h_angle, v_angle, ds_angle = gm.geometry(self.bs_parameters["antenna_mode"][bs],
                                                                                       rx_coord, tx_coord, rx_coord_old,
                                                                                       tx_coord_old, t_now, t_old)
                        # print("lc h_angle", h_angle)
                        # print("lc v_angle", v_angle)

                        ch_outcomes_rx, d_correlation_map_rx, hb_map_rx, jakes_map = ch_38_901.get_ch_tr_38_901(
                                                                        self.bs_parameters["scenario"][bs],
                                                                        self.bs_parameters["antenna_mode"][bs],
                                                                        self.general_channel_modeling["shadowing"][0],
                                                                        self.general_channel_modeling["dynamic_los"][0],
                                                                        self.general_channel_modeling["dynamic_hb"][0],
                                                                        self.general_channel_modeling["o2i"][0],
                                                                        inside_what,
                                                                        self.general_channel_modeling["penetration_loss_model"][0],
                                                                        d_2d, d_3d, rx_coord[2], tx_coord[2],
                                                                        self.general_parameters["h_ceiling"][0],
                                                                        self.general_parameters["block_density"][0],
                                                                        self.bs_parameters["fc"][bs],
                                                                        d_correlation_map_rx, t_now, t_old, speed_rx, speed_tx, rx_coord, tx_coord,
                                                                        h_angle, v_angle, ds_angle, self.bs_parameters["v_tilt"][bs],
                                                                        n_rb, jakes_map,
                                                                        self.general_channel_modeling["fast_fading_model"][0],
                                                                        hb_map_rx, cable_loss_tx,
                                                                        self.general_parameters["thermal_noise"][0],
                                                                        bw_rb, noise_figure_rx, self.general_channel_modeling["fast_fading"][0],
                                                                        self.bs_parameters["p_tx"][bs], self.bs_parameters["ax_gain"][bs], ax_gain_rx,
                                                                        self.general_channel_modeling["atmospheric_absorption"][0],
                                                                        self.general_channel_modeling["desired_delay_spread"][0],
                                                                        self.bs_parameters["fast_fading_los_type"][bs],
                                                                        self.bs_parameters["fast_fading_nlos_type"][bs],
                                                                        1, 1,
                                                                        np.random.default_rng(),
                                                                        antenna_mode_rx, self.bs_parameters["ax_panel_polarization"][bs]
                                                                        )

                        # print(ch_outcomes_rx)


                    elif type == "sat":
                        hb_map_rx = {"t": None, "x": None, "y": None, "h_blockage": None}
                        d_3d, speed_rx, speed_tx, elevation_angle, ds_angle = gm_ntn.geometry(self.grid_lla, self.grid_xy, rx_coord, tx_coord, rx_coord_old, tx_coord_old, desired_elevation_angle, t_now, t_old)

                        ch_outcomes_rx, d_correlation_map_rx, jakes_map = ch_38_811.get_ch_tr_38_811(
                                                                        t_now, t_old, speed_rx, speed_tx, ds_angle, rx_coord, tx_coord,
                                                                        self.bs_parameters["scenario"][bs],
                                                                        rx_scenario,
                                                                        self.bs_parameters["antenna_mode"][bs],
                                                                        self.general_channel_modeling["dynamic_los"][0],
                                                                        elevation_angle, d_3d, tx_coord[2],
                                                                        self.bs_parameters["fc"][bs], f_band_rx,
                                                                        self.general_channel_modeling["o2i"][0],
                                                                        inside_what,
                                                                        self.general_channel_modeling["penetration_loss_model"][0],
                                                                        d_correlation_map_rx,
                                                                        self.general_channel_modeling["shadowing"][0],
                                                                        n_rb, jakes_map,
                                                                        self.general_channel_modeling["fast_fading_model"][0], cable_loss_tx,
                                                                        self.general_parameters["thermal_noise"][0],
                                                                        bw_rb, noise_figure_rx, self.general_channel_modeling["fast_fading"][0],
                                                                        self.bs_parameters["p_tx"][bs], self.bs_parameters["ax_gain"][bs], ax_gain_rx,
                                                                        self.general_channel_modeling["atmospheric_absorption"][0],
                                                                        self.general_channel_modeling["desired_delay_spread"][0],
                                                                        self.bs_parameters["fast_fading_los_type"][bs],
                                                                        self.bs_parameters["fast_fading_nlos_type"][bs],
                                                                        1, 1,
                                                                        np.random.default_rng(),
                                                                        antenna_mode_rx, self.bs_parameters["ax_panel_polarization"][bs]
                                                                        )

                        # print(ch_outcomes_rx)



                    if t == 0:
                        overall_d_correlation_map_rx.append(d_correlation_map_rx)
                        overall_hb_map_rx.append(hb_map_rx)
                        overall_jakes_map.append(jakes_map)
                    else:
                        overall_d_correlation_map_rx[i] = d_correlation_map_rx
                        overall_hb_map_rx[i] = hb_map_rx
                        overall_jakes_map[i] = jakes_map



                    bler_array, sinr_array = bl.get_channel_curves(self.general_parameters["channel_type"][0])
                    cqi, bler = l2s.get_cqi_bler(sinr_array, bler_array, self.general_parameters["target_bler"][0], ch_outcomes_rx["sinr"])

                    sinr_txk[t][i] = ch_outcomes_rx["sinr"]
                    cqi_txk[t][i] = cqi
                    bler_txk[t][i] = bler
                metrics_dic_bs["sinr"] = sinr_txk
                metrics_dic_bs["cqi"] = cqi_txk
                metrics_dic_bs["bler"] = bler_txk
            overall_metrics.append(metrics_dic_bs)

        return overall_metrics

    def link_computation_bs2d_ul(self):

        number_bs = self.bs_parameters.shape[0]

        k = self.sub_groups_parameters['k_sub'].sum()
        number_sub = self.sub_groups_parameters.shape[0]
        sub_members = self.sub_groups_parameters['k_sub'].values
        cumulative_sub_members = np.cumsum(sub_members)
        # print("cumulative_sub_members" ,cumulative_sub_members)
        sub_types = self.sub_groups_parameters['type'].values
        time_steps = self.df_x.shape[0]
        overall_metrics = []

        for bs in range(number_bs):
            type = self.bs_parameters["type"][bs]
            rx_coord = [self.bs_parameters["x"][bs], self.bs_parameters["y"][bs], self.bs_parameters["z"][bs]]
            rx_coord_old = rx_coord  # TODO: To be taken into account when we add the abs movement capability
            bw_rb = 12*(15*(2**self.bs_parameters["numerology"][bs]))*1e3
            n_rb = self.bs_parameters["n_rb"][bs]
            cable_loss_rx = self.bs_parameters["cable_loss"][bs]
            noise_figure_rx = self.bs_parameters["noise_figure"][bs]
            ax_gain_rx = self.bs_parameters["ax_gain"][bs]
            antenna_mode_rx = self.bs_parameters["antenna_mode"][bs]

            overall_d_correlation_map_rx = []
            overall_hb_map_rx = []
            overall_jakes_map = []

            cqi_txk = np.zeros([time_steps, k], dtype=float)
            sinr_txk = np.zeros([time_steps, k], dtype=float)
            bler_txk = np.zeros([time_steps, k], dtype=float)
            metrics_dic_bs = {"sinr": None, "cqi": None, "bler": None}

            for t in range(time_steps):
                t_now = self.df_x[0][t]
                if t == 0:  t_old = t_now
                else: t_old = self.df_x[0][t - 1]

                for i in range(k):
                    tx_coord = [self.df_x[i + 1][t], self.df_y[i + 1][t], self.df_z[i + 1][t]]
                    if t == 0: tx_coord_old = tx_coord
                    else: rx_coord_old = [self.df_x[i + 1][t - 1], self.df_y[i + 1][t - 1], self.df_z[i + 1][t - 1]]

                    for s in range(number_sub):
                        if i <= cumulative_sub_members[s]:
                            type_tx = self.sub_groups_parameters["type"][s]
                            noise_figure_tx = self.sub_groups_parameters["noise_figure"][s]
                            antenna_mode_tx = self.sub_groups_parameters["antenna_mode"][s]
                            p_tx_tx = self.sub_groups_parameters["p_tx"][s]
                            ax_gain_tx = self.sub_groups_parameters["ax_gain"][s]
                            d2d_tx = self.sub_groups_parameters["d2d"][s]
                    if type_tx == "car_mounted": inside_what = "car"
                    else: inside_what = self.general_channel_modeling["inside_what_o2i"][0],

                    if t == 0:
                        d_correlation_map_rx = {"t": None, "x": None, "y": None, "shadowing": None, "o2i_loss": None,
                                                "los": None, "d_correlation_sf": None}
                        hb_map_rx = {"t": None, "x": None, "y": None, "h_blockage": None}
                        jakes_map = np.zeros([n_rb * 6, 3], dtype=float)

                    else:
                        d_correlation_map_rx = overall_d_correlation_map_rx[i]
                        hb_map_rx = overall_hb_map_rx[i]
                        jakes_map = overall_jakes_map[i]



                    d_2d, d_3d, speed_rx, speed_tx, h_angle, v_angle, ds_angle = gm.geometry(
                                                                                    self.bs_parameters["antenna_mode"][bs],
                                                                                    rx_coord, tx_coord, rx_coord_old,
                                                                                    tx_coord_old, t_now, t_old)

                    ch_outcomes_rx, d_correlation_map_rx, hb_map_rx, jakes_map = ch_38_901.get_ch_tr_38_901(
                                                                    self.bs_parameters["scenario"][bs],
                                                                    self.bs_parameters["antenna_mode"][bs],
                                                                    self.general_channel_modeling["shadowing"][0],
                                                                    self.general_channel_modeling["dynamic_los"][0],
                                                                    self.general_channel_modeling["dynamic_hb"][0],
                                                                    self.general_channel_modeling["o2i"][0],
                                                                    inside_what,
                                                                    self.general_channel_modeling["penetration_loss_model"][0],
                                                                    d_2d, d_3d, rx_coord[2], tx_coord[2],
                                                                    self.general_parameters["h_ceiling"][0],
                                                                    self.general_parameters["block_density"][0],
                                                                    self.bs_parameters["fc"][bs],
                                                                    d_correlation_map_rx, t_now, t_old, speed_rx, speed_tx, rx_coord, tx_coord,
                                                                    h_angle, v_angle, ds_angle, self.bs_parameters["v_tilt"][bs],
                                                                    n_rb, jakes_map,
                                                                    self.general_channel_modeling["fast_fading_model"][0],
                                                                    hb_map_rx, cable_loss_rx,
                                                                    self.general_parameters["thermal_noise"][0],
                                                                    bw_rb, noise_figure_rx, self.general_channel_modeling["fast_fading"][0],
                                                                    p_tx_tx, ax_gain_tx, ax_gain_rx,
                                                                    self.general_channel_modeling["atmospheric_absorption"][0],
                                                                    self.general_channel_modeling["desired_delay_spread"][0],
                                                                    self.bs_parameters["fast_fading_los_type"][bs],
                                                                    self.bs_parameters["fast_fading_nlos_type"][bs],
                                                                    1, 1,
                                                                    np.random.default_rng(),
                                                                    antenna_mode_rx, self.bs_parameters["ax_panel_polarization"][bs]
                                                                    )

                    if t == 0:
                        overall_d_correlation_map_rx.append(d_correlation_map_rx)
                        overall_hb_map_rx.append(hb_map_rx)
                        overall_jakes_map.append(jakes_map)
                    else:
                        overall_d_correlation_map_rx[i] = d_correlation_map_rx
                        overall_hb_map_rx[i] = hb_map_rx
                        overall_jakes_map[i] = jakes_map


                    bler_array, sinr_array = bl.get_channel_curves(self.general_parameters["channel_type"][0])
                    cqi, bler = l2s.get_cqi_bler(sinr_array, bler_array, self.general_parameters["target_bler"][0], ch_outcomes_rx["sinr"])

                    sinr_txk[t][i] = ch_outcomes_rx["sinr"]
                    cqi_txk[t][i] = cqi
                    bler_txk[t][i] = bler
                metrics_dic_bs["sinr"] = sinr_txk
                metrics_dic_bs["cqi"] = cqi_txk
                metrics_dic_bs["bler"] = bler_txk

            df_sinr_txk = pd.DataFrame(sinr_txk)
            df_sinr_txk.insert(0, 't', self.time_map)

            df_cqi_txk = pd.DataFrame(cqi_txk)
            df_cqi_txk.insert(0, 't', self.time_map)

            df_bler_txk = pd.DataFrame(bler_txk)
            df_bler_txk.insert(0, 't', self.time_map)

            overall_metrics.append(metrics_dic_bs)
        return overall_metrics

    def link_computation_d2d(self):

        number_bs = self.bs_parameters.shape[0]

        k = self.sub_groups_parameters['k_sub'].sum()
        number_sub = self.sub_groups_parameters.shape[0]
        sub_members = self.sub_groups_parameters['k_sub'].values
        cumulative_sub_members = np.cumsum(sub_members)
        sub_types = self.sub_groups_parameters['type'].values
        time_steps = self.df_x.shape[0]

        overall_metrics = []
        shape = (time_steps, k, k)
        cqi_txkxk = np.full(shape, None, dtype=object)
        sinr_txkxk = np.full(shape, None, dtype=object)
        bler_txkxk = np.full(shape, None, dtype=object)
        metrics_dic = {"sinr": None, "cqi": None, "bler": None}

        for ii in range(k):  # loop for the end devises as forwarding devises (fd) regarding the remainder k-1 devices
            for s in range(number_sub):
                if ii <= cumulative_sub_members[s]:
                    type_fd = self.sub_groups_parameters["type"][s]
                    noise_figure_fd = self.sub_groups_parameters["noise_figure"][s]
                    antenna_mode_fd = self.sub_groups_parameters["antenna_mode"][s]
                    p_tx_fd = self.sub_groups_parameters["p_tx"][s]
                    ax_gain_fd = self.sub_groups_parameters["ax_gain"][s]
                    cable_loss_fd = self.sub_groups_parameters["cable_loss"][s]
                    d2d_fd = self.sub_groups_parameters["d2d"][s]
            if d2d_fd:
                bw_rb = 12*(15*(2**self.bs_parameters["numerology"][0]))*1e3  # TODO check, I am assuming the same numerology as the first bs
                n_rb = self.bs_parameters["n_rb"][0]  # TODO check, I am assuming the same n_rb as the first bs
                v_tilt = 0
                scenario = "D2D"
                o2i = False
                fc = self.bs_parameters["fc"][0]


                overall_d_correlation_map_rx = []
                overall_hb_map_rx = []
                overall_jakes_map = []

                for t in range(time_steps):
                    t_now = self.df_x[0][t]
                    fd_coord = [self.df_x[ii + 1][t], self.df_y[ii + 1][t], self.df_z[ii + 1][t]]
                    if t == 0:
                        t_old = t_now
                        fd_coord_old = fd_coord
                    else:
                        t_old = self.df_x[0][t - 1]
                        fd_coord_old = [self.df_x[ii + 1][t - 1], self.df_y[ii + 1][t - 1], self.df_z[ii + 1][t - 1]]

                    for i in range(k):
                        if i == ii:
                            if t == 0:
                                overall_d_correlation_map_rx.append({"t": None, "x": None, "y": None, "shadowing": None, "o2i_loss": None,
                                                    "los": None, "d_correlation_sf": None})
                                overall_hb_map_rx.append({"t": None, "x": None, "y": None, "h_blockage": None})
                                overall_jakes_map.append(np.zeros([n_rb * 6, 3], dtype=float))
                            else:
                                overall_d_correlation_map_rx[i] = {"t": None, "x": None, "y": None, "shadowing": None, "o2i_loss": None,
                                                        "los": None, "d_correlation_sf": None}
                                overall_hb_map_rx[i] = {"t": None, "x": None, "y": None, "h_blockage": None}
                                overall_jakes_map[i] = np.zeros([n_rb * 6, 3], dtype=float)
                        else:

                            rx_coord = [self.df_x[i + 1][t], self.df_y[i + 1][t], self.df_z[i + 1][t]]
                            if t == 0: rx_coord_old = rx_coord
                            else: rx_coord_old = [self.df_x[i + 1][t - 1], self.df_y[i + 1][t - 1], self.df_z[i + 1][t - 1]]

                            for s in range(number_sub):
                                if i <= cumulative_sub_members[s]:
                                    type_rx = self.sub_groups_parameters["type"][s]
                                    noise_figure_rx = self.sub_groups_parameters["noise_figure"][s]
                                    antenna_mode_rx = self.sub_groups_parameters["antenna_mode"][s]
                                    p_tx_rx = self.sub_groups_parameters["p_tx"][s]
                                    ax_gain_rx = self.sub_groups_parameters["ax_gain"][s]
                                    cable_loss_rx = self.sub_groups_parameters["cable_loss"][s]
                                    d2d_rx = self.sub_groups_parameters["d2d"][s]

                            if not d2d_rx:
                                if t == 0:
                                    overall_d_correlation_map_rx.append(
                                        {"t": None, "x": None, "y": None, "shadowing": None, "o2i_loss": None,
                                         "los": None, "d_correlation_sf": None})
                                    overall_hb_map_rx.append({"t": None, "x": None, "y": None, "h_blockage": None})
                                    overall_jakes_map.append(np.zeros([n_rb * 6, 3], dtype=float))
                                else:
                                    overall_d_correlation_map_rx[i] = {"t": None, "x": None, "y": None,
                                                                       "shadowing": None, "o2i_loss": None,
                                                                       "los": None, "d_correlation_sf": None}
                                    overall_hb_map_rx[i] = {"t": None, "x": None, "y": None, "h_blockage": None}
                                    overall_jakes_map[i] = np.zeros([n_rb * 6, 3], dtype=float)

                            else:

                                if type_rx == "car_mounted": inside_what = "car"
                                else: inside_what = self.general_channel_modeling["inside_what_o2i"][0],

                                if t == 0:
                                    d_correlation_map_rx = {"t": None, "x": None, "y": None, "shadowing": None, "o2i_loss": None,
                                                            "los": None, "d_correlation_sf": None}
                                    hb_map_rx = {"t": None, "x": None, "y": None, "h_blockage": None}
                                    jakes_map = np.zeros([n_rb * 6, 3], dtype=float)

                                else:
                                    d_correlation_map_rx = overall_d_correlation_map_rx[i]
                                    hb_map_rx = overall_hb_map_rx[i]
                                    jakes_map = overall_jakes_map[i]


                                d_2d, d_3d, speed_rx, speed_tx, h_angle, v_angle, ds_angle = gm.geometry(
                                                                                                antenna_mode_fd,
                                                                                                rx_coord, fd_coord, rx_coord_old,
                                                                                                fd_coord_old, t_now, t_old)

                                ch_outcomes_rx, d_correlation_map_rx, hb_map_rx, jakes_map = ch_38_901.get_ch_tr_38_901(
                                                                                scenario,
                                                                                antenna_mode_fd,
                                                                                self.general_channel_modeling["shadowing"][0],
                                                                                self.general_channel_modeling["dynamic_los"][0],
                                                                                self.general_channel_modeling["dynamic_hb"][0],
                                                                                o2i,
                                                                                inside_what,
                                                                                self.general_channel_modeling["penetration_loss_model"][0],
                                                                                d_2d, d_3d, rx_coord[2], fd_coord[2],
                                                                                self.general_parameters["h_ceiling"][0],
                                                                                self.general_parameters["block_density"][0],
                                                                                fc,
                                                                                d_correlation_map_rx, t_now, t_old, speed_rx, speed_tx, rx_coord, fd_coord,
                                                                                h_angle, v_angle, ds_angle, v_tilt,
                                                                                n_rb, jakes_map,
                                                                                "jakes",
                                                                                hb_map_rx, cable_loss_rx,
                                                                                self.general_parameters["thermal_noise"][0],
                                                                                bw_rb, noise_figure_rx, self.general_channel_modeling["fast_fading"][0],
                                                                                p_tx_fd, ax_gain_fd, ax_gain_rx,
                                                                                self.general_channel_modeling["atmospheric_absorption"][0],
                                                                                self.general_channel_modeling["desired_delay_spread"][0],
                                                                                None,
                                                                                None,
                                                                                1, 1,
                                                                                np.random.default_rng(),
                                                                                antenna_mode_rx, None
                                                                                )
                                if t == 0:
                                    overall_d_correlation_map_rx.append(d_correlation_map_rx)
                                    overall_hb_map_rx.append(hb_map_rx)
                                    overall_jakes_map.append(jakes_map)
                                else:
                                    overall_d_correlation_map_rx[i] = d_correlation_map_rx
                                    overall_hb_map_rx[i] = hb_map_rx
                                    overall_jakes_map[i] = jakes_map


                                bler_array, sinr_array = bl.get_channel_curves(self.general_parameters["channel_type"][0])
                                cqi, bler = l2s.get_cqi_bler(sinr_array, bler_array, self.general_parameters["target_bler"][0], ch_outcomes_rx["sinr"])

                                sinr_txkxk[t][ii][i] = ch_outcomes_rx["sinr"]
                                cqi_txkxk[t][ii][i] = cqi
                                bler_txkxk[t][ii][i] = bler

                metrics_dic["sinr"] = sinr_txkxk
                metrics_dic["cqi"] = cqi_txkxk
                metrics_dic["bler"] = bler_txkxk

        return metrics_dic





def link_computation_bs2d_dl(bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                        df_z, time_map, grid_lla, grid_xy):

    lc = Link_computation(bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                        df_z, time_map, grid_lla, grid_xy)
    metrics_dic_bs_dl = lc.link_computation_bs2d_dl()
    return metrics_dic_bs_dl

def link_computation_d2d(bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                        df_z, time_map, grid_lla, grid_xy):

    lc = Link_computation(bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                        df_z, time_map, grid_lla, grid_xy)
    metrics_dic_d2d = lc.link_computation_d2d()
    return metrics_dic_d2d

def link_computation_bs2d_ul(bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                        df_z, time_map, grid_lla, grid_xy):

    lc = Link_computation(bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                        df_z, time_map, grid_lla, grid_xy)
    metrics_dic_bs_ul = lc.link_computation_bs2d_ul()
    return metrics_dic_bs_ul
