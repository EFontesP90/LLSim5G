import numpy as np
import pandas as pd
# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from datetime import datetime
import pickle
import joblib


class Outputs_config(object):
    """
    25/04/2024


    Required attributes:
    ():

    """

    def __init__(self, general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters,
                 general_parameters, df_x, df_y,
                 df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, metrics_dic_bs_dl = None, metrics_dic_bs_ul = None, metrics_dic_d2d = None):

        self.general_simulation_parameters = general_simulation_parameters  # ["grid_xy", "simulation_time", "simulation_resolution", "downlink", "uplink", "d2d_link", "save_scenario_xlsx", "save_metrics_xlsx", "show_video", "save_video", "video_format", "video_velocity", "print_scenario_outputs", "print_metrics_outputs"]
        self.bs_parameters = bs_parameters  # [number of bs]x["x", "y", "z", "type", "scenario", "antenna_mode", "fc", "numerology", "n_rb", "p_tx", "ax_gain", "cable_loss", "noise_figure"]
        self.general_channel_modeling = general_channel_modeling  # 1x["dynamic_los", "dynamic_hb", "o2i", "inside_what_o2i", "penetration_loss_model", "shadowing", "fast_fading", "fast_fading_model"]
        self.sub_groups_parameters = sub_groups_parameters  # [number of defined user subgroups]x["type", "k_sub", "antenna_mode", "p_tx", "ax_gain", "cable_loss", "noise_figure", "d2d", "fixed_height", "grid_size_ratio", "reference_location", "min_max_velocity", "wait_time", "mobility_model", "aggregation", "number_mg_rpg_model", "min_max_height"]
        self.general_parameters = general_parameters  # 1x["thermal_noise", "h_ceiling", "block_density"]

        self.show_video = general_simulation_parameters["show_video"][0]
        self.save_video = general_simulation_parameters["save_video"][0]
        self.save_scenario_xlsx = general_simulation_parameters["save_scenario_xlsx"][0]
        self.video_format = general_simulation_parameters["video_format"][0]  # mp4, gif, Both
        self.save_metrics_xlsx = general_simulation_parameters["save_metrics_xlsx"][0]
        self.print_scenario_outputs = general_simulation_parameters["print_scenario_outputs"][0]
        self.print_metrics_outputs = general_simulation_parameters["print_metrics_outputs"][0]

        self.enabling_dl = general_simulation_parameters["downlink"][0]
        self.enabling_ul = general_simulation_parameters["uplink"][0]
        self.enabling_d2d = general_simulation_parameters["d2d_link"][0]

        self.df_x = df_x  # array of [time_steps]x[number of users], with the X coordinates of each user over time
        self.df_y = df_y  # array of [time_steps]x[number of users], with the Y coordinates of each user over time
        self.df_z = df_z  # array of [time_steps]x[number of users], with the Z coordinates of each user over time
        self.df_tbs_xyz = df_tbs_xyz
        self.df_abs_xyz = df_abs_xyz
        self.df_sat_lla = df_sat_lla
        self.time_map = time_map

        self.metrics_dic_bs_dl = metrics_dic_bs_dl  # List of Dictionaries: (number of bs)x{"sinr": (txk), "cqi": (txk), "bler": (txk)} with  the outuput od the Downlink-Link computation module
        self.metrics_dic_bs_ul = metrics_dic_bs_ul  # List of Dictionaries: (number of bs)x{"sinr": (txk), "cqi": (txk), "bler": (txk)} with  the outuput od the Uplink-Link computation module
        self.metrics_dic_d2d = metrics_dic_d2d  # Dictionary: {"sinr": (txkxk), "cqi": (txkxk), "bler": (txkxk)} with  the outuput od the Downlink-Link computation module

        self.current_date = datetime.now().strftime("%Y%m%d_%H%M")
        self.number_bs = self.bs_parameters.shape[0]
        self.k = self.sub_groups_parameters['k_sub'].sum()
        self.number_sub = self.sub_groups_parameters.shape[0]
        self.sub_members = self.sub_groups_parameters['k_sub'].values
        self.cumulative_sub_members = np.cumsum(self.sub_members)
        self.sub_types = self.sub_groups_parameters['type'].values
        self.time_steps = self.df_x.shape[0]

    def scenario_outputs(self):
        if self.print_scenario_outputs:
            print("data frame with the x coordinates of the simulated end-devices")
            print(self.df_x)
            print("data frame with the y coordinates of the simulated end-devices")
            print(self.df_y)
            print("data frame with the z coordinates of the simulated end-devices")
            print(self.df_z)
            print("data frame with the x,y,z coordinates of the simulated terrestrial base-stations")
            print(self.df_tbs_xyz)
            print("data frame with the x,y,z coordinates of the simulated aerial base-stations (UAV)")
            print(self.df_abs_xyz)
            print("data frame with the lla: latitude, longitude, and altitude of the simulated satellites (Sat)")
            print(self.df_sat_lla)

        if self.save_scenario_xlsx:
            self.df_x.to_excel(f"output/scenario/df_ed_x_{self.current_date}.xlsx")
            self.df_y.to_excel(f"output/scenario/df_ed_y_{self.current_date}.xlsx")
            self.df_z.to_excel(f"output/scenario/df_ed_z_{self.current_date}.xlsx")
            self.df_tbs_xyz.to_excel(f"output/scenario/df_tbs_xyz_{self.current_date}.xlsx")
            self.df_abs_xyz.to_excel(f"output/scenario/df_abs_xyz_{self.current_date}.xlsx")
            self.df_sat_lla.to_excel(f"output/scenario/df_sat_lla_{self.current_date}.xlsx")
        return True

    def link_computation_bs2d_dl_outputs(self):
        for bs in range(self.number_bs):

            type = self.bs_parameters["type"][bs]
            sinr_txk = self.metrics_dic_bs_dl[bs]["sinr"]
            cqi_txk = self.metrics_dic_bs_dl[bs]["cqi"]
            bler_txk = self.metrics_dic_bs_dl[bs]["bler"]

            df_sinr_txk = pd.DataFrame(sinr_txk)
            df_sinr_txk.insert(0, 't', self.time_map)
            df_cqi_txk = pd.DataFrame(cqi_txk)
            df_cqi_txk.insert(0, 't', self.time_map)
            df_bler_txk = pd.DataFrame(bler_txk)
            df_bler_txk.insert(0, 't', self.time_map)

            if self.print_metrics_outputs:
                print(
                    f"data frame with the downlink sinr of the end-devices regarding the base-station: {bs + 1} ({type})")
                print(df_sinr_txk)
                print(
                    f"data frame with the downlink CQI of the end-devices regarding the base-station: {bs + 1} ({type})")
                print(df_cqi_txk)
                print(
                    f"data frame with the downlink bler of the end-devices regarding the base-station: {bs + 1} ({type})")
                print(df_bler_txk)

            if self.save_metrics_xlsx:
                df_sinr_txk.to_excel(f"output/metrics/bs_{bs}_{type}_sinr_{self.current_date}.xlsx")
                df_cqi_txk.to_excel(f"output/metrics/bs_{bs}_{type}_cqi_{self.current_date}.xlsx")
                df_bler_txk.to_excel(f"output/metrics/bs_{bs}_{type}_bler_{self.current_date}.xlsx")
                joblib.dump(self.metrics_dic_bs_dl, f"output/metrics/metrics_{self.current_date}.pkl")
        return True

    def link_computation_bs2d_ul_outputs(self):
        for bs in range(self.number_bs):
            type = self.bs_parameters["type"][bs]

            sinr_txk = self.metrics_dic_bs_ul[bs]["sinr"]
            cqi_txk = self.metrics_dic_bs_ul[bs]["cqi"]
            bler_txk = self.metrics_dic_bs_ul[bs]["bler"]

            df_sinr_txk = pd.DataFrame(sinr_txk)
            df_sinr_txk.insert(0, 't', self.time_map)
            df_cqi_txk = pd.DataFrame(cqi_txk)
            df_cqi_txk.insert(0, 't', self.time_map)
            df_bler_txk = pd.DataFrame(bler_txk)
            df_bler_txk.insert(0, 't', self.time_map)

            if self.print_metrics_outputs:
                print(
                    f"data frame with the uplink sinr of the end-devices regarding the base-station: {bs + 1} ({type})")
                print(df_sinr_txk)
                print(
                    f"data frame with the uplink CQI of the end-devices regarding the base-station: {bs + 1} ({type})")
                print(df_cqi_txk)
                print(
                    f"data frame with the uplink bler of the end-devices regarding the base-station: {bs + 1} ({type})")
                print(df_bler_txk)

            if self.save_metrics_xlsx:
                df_sinr_txk.to_excel(f"output/metrics/bs_{bs}_{type}_sinr_{self.current_date}.xlsx")
                df_cqi_txk.to_excel(f"output/metrics/bs_{bs}_{type}_cqi_{self.current_date}.xlsx")
                df_bler_txk.to_excel(f"output/metrics/bs_{bs}_{type}_bler_{self.current_date}.xlsx")
                joblib.dump(self.metrics_dic_bs_ul, f"output/metrics/metrics_{self.current_date}.pkl")
        return True

    def link_computation_d2d_outputs(self):

        sinr_txkxk = self.metrics_dic_d2d["sinr"]
        cqi_txkxk = self.metrics_dic_d2d["cqi"]
        bler_txkxk = self.metrics_dic_d2d["bler"]

        if self.print_metrics_outputs:
            print(f"Matrix of KxK with the d2d link sinr among the K end-devices along the {self.time_steps} simulation time_steps")
            print(sinr_txkxk)
            print(f"Matrix of KxK with the d2d link cqi among the K end-devices along the {self.time_steps} simulation time_steps")
            print(cqi_txkxk)
            print(f"Matrix of KxK with the d2d link bler among the K end-devices along the {self.time_steps} simulation time_steps")
            print(bler_txkxk)

        if self.save_metrics_xlsx:
            joblib.dump(self.metrics_dic_d2d, f"output/metrics/d2d_metrics_{self.current_date}.pkl")

        return True


def scenario_outputs(general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                    df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, metrics_dic_bs_dl, metrics_dic_bs_ul, metrics_dic_d2d):
    outputs = Outputs_config(general_simulation_parameters, bs_parameters, general_channel_modeling,
                             sub_groups_parameters, general_parameters, df_x, df_y,
                             df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, metrics_dic_bs_dl, metrics_dic_bs_ul,
                             metrics_dic_d2d)
    outputs.scenario_outputs()
    return True


def bs2d_dl_outputs(general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                    df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, metrics_dic_bs_dl, metrics_dic_bs_ul, metrics_dic_d2d):
    outputs = Outputs_config(general_simulation_parameters, bs_parameters, general_channel_modeling,
                             sub_groups_parameters, general_parameters, df_x, df_y,
                             df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, metrics_dic_bs_dl, metrics_dic_bs_ul,
                             metrics_dic_d2d)
    outputs.link_computation_bs2d_dl_outputs()
    return True


def bs2d_ul_outputs(general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                 df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, metrics_dic_bs_dl, metrics_dic_bs_ul, metrics_dic_d2d):
    outputs = Outputs_config(general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                 df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, metrics_dic_bs_dl, metrics_dic_bs_ul, metrics_dic_d2d)
    outputs.link_computation_bs2d_ul_outputs()
    return True


def d2d_outputs(general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                 df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla,time_map, metrics_dic_bs_dl, metrics_dic_bs_ul, metrics_dic_d2d):
    outputs = Outputs_config(general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                 df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla,time_map, metrics_dic_bs_dl, metrics_dic_bs_ul, metrics_dic_d2d)
    outputs.link_computation_d2d_outputs()
    return True