import numpy as np
import pandas as pd
import ast
import random

import scenario.scenario_definition as sx
import link_computation.link_computation as lc
import output.outputs_config as out
import general.general as ge




pd.set_option('display.max_rows', None)  # This specific setting ensures that all rows in a DataFrame will be displayed when it's printed.
pd.set_option('display.max_columns', None)  # This specific setting ensures that all columns in a DataFrame will be displayed when it's printed.



if __name__ == "__main__":


    ########## Uncomment to fix the random outputs during simulations ##########
    # np.random.seed(42)  # Use any integer seed value
    # random.seed(42)  # Use any integer seed value
    ######################################################################

    simulation_settings = "from_main_script"  # String: ("from_main_script", "from_excel"). To set the simulation settings directly from the script or from external excels placed in the simulation_settings folder
    save_simulation_settings = True  # Boolean. To save the simulation configuration as excels files into the folder simulation_settings

    if simulation_settings == "from_main_script":

        general_simulation_parameters = pd.DataFrame(
            [[
                "[100, 100]",  # "grid_xy". Definition of the x and y in meters (m) of the simulation grid. It must be definded as a string for ensuring saving and uploading in and for the configuration excels.
                39.2137738,  # "grid_center_latitude". Latitude in degrees of the grid center, default value: grid_center_lat = 39.2137738 degrees, used for NTN simulations.
                9.1153844,  # "grid_center_longitude". Longitude in degrees of the grid centerin degrees, default value: grid_center_lat = 9.1153844 degrees, used for NTN simulations.
                1,  # "simulation_time". Simulation time in seconds (s).
                0.1,  # "simulation_resolution". Simulation resolution, e.g., 1 means a one second resolution (one sample per second), 0.1 means a 0.1 seconds resolution (100 ms) or 10 samples for every second.
                True,  # "downlink". To enable the downlink computation from the base stations (BSs) to the end-devices (EDs).
                False,  # "uplink". To enable the uplink computation from the EDs to the BSs. TODO, it's not enabled between the EDs and the non-terrestrial networks (NTNs).
                False,  # "d2d_link". To enable the device-to-device (D2D) link computation among EDs.
                False,  # "ntn_link". To enable the link computation between the available non-terrestrial networks (NTN) (e.g., LEO, MEO, HAPS) and the EDs.
                False,  # "save_scenario_xlsx". To save (./output/scenario/) as .xlsx the simulated scenario. It means the coordinate x, y and z of each EDs or BSs, or the latitude, longitude, altitude (LLA) of the NTNs.
                False,  # "save_metrics_xlsx". To save (./output/metrics/) as .xlsx the resulting LLS outputs: SINR, CQI, BLER, among each ED and each BS (TN/NTN), or EDs for D2D communications.
                True,  # "show_video". Boolean to enable or disable the simulation video display regarding the grid and the configured BSs and EDs with their mobility behaviour. (The link computation is executed after we closed the video).
                False,  # "save_video". Boolean to enable saving the video file.
                "gif",  # "video_format"
                0.1,  # "video_velocity". Float variable for modifying the video velocity. Default value: 0.1.
                True,  # "print_scenario_outputs". Boolean for enable printing the scenario output. It means the coordinate x, y and z of each EDs or BSs, or the LLA of the NTNs.
                True  # "print_metrics_outputs". Boolean for enable printing the resulting simulation metrics: SINR, CQI, BLER, among each ED and each BS (TN/NTN), or EDs for D2D communications.
            ]])
        general_simulation_parameters.columns = ["grid_xy", "grid_center_latitude", "grid_center_longitude", "simulation_time", "simulation_resolution", "downlink",
                                                 "uplink", "d2d_link", "ntn_link", "save_scenario_xlsx",
                                                 "save_metrics_xlsx", "show_video", "save_video", "video_format",
                                                 "video_velocity", "print_scenario_outputs", "print_metrics_outputs"]

        general_channel_modeling = pd.DataFrame(
            [[
                False,           # "dynamic_los": Boolean where True means a dynamic Line of Sight (LOS), a user could be in LOS or non LOS (NLOS) rearding each BS. False means only LOS.
                False,          # "dynamic_hb": Boolean where True means a dynamic human blockage (hb) in the link beteen the BS an the ED (mainly unsd for mmWave simulations). False means not hb considerations.
                False,          # "o2i": Boolean where True means a dynamic outdoor-to-indoor (o2i), a user could be simulated in o2i or not conditions regarding the BS (It means NLOS). False means only LOS.
                "dynamic",     # "inside_what_o2i": ("dynamic", "building", "car"). "dynamic" means that will be choused randomly if the user is inside a building or a car, it will modify the penetration losses considered. The other options are to fix, or inside a building or inside a car.
                "low-loss",     # "penetration_loss_model": ("low-loss", "high-loss"). Accirding to 3GPP TR 38.811: 6.6.3 O2I penetration loss for NTN simulations only. Low-loss means a traditional building type, and high-loss means a thermally efficient building type.
                True,          # "shadowing": Boolean for enabling the shadowing fading (slow-fading) attenuation in the link.
                True,           # "fast_fading": Boolean for enabling the fast fading attenuation in the link according to jakes, tdl or cdl (TODO CDL is not enabled for NTN links).
                "jakes",          # "fast_fading_model": ("jakes", "tdl", "cdl") TDL and CDL are implemented according 3GPP TR 38.901  (TODO CDL is not enabled for NTN links).
                False,            # "atmospheric_absorption". Boolean for enabling the atmospheric absorption attenuation in the link.
                "Very short"    # "desired_delay_spread": ("Nominal", "Very long", "Long delay", "Short", "Very short", None). 3GPP TR 38.901, Table 7.7.3-1. Example scaling parameters for CDL and TDL models.

            ]])
        general_channel_modeling.columns = ["dynamic_los", "dynamic_hb", "o2i", "inside_what_o2i", "penetration_loss_model",
                                            "shadowing", "fast_fading", "fast_fading_model", "atmospheric_absorption", "desired_delay_spread"]

        general_parameters = pd.DataFrame(
            [[
                -174,  # "thermal_noise". The value -174 dBm/Hz is commonly used to represent the thermal noise power spectral density at room temperature (approximately 290K).
                10,  # "h_ceiling". Considered height of the buildings (in m) for O2I penetration losses computation. Default value: 10 m. TODO make an internal variable
                0.2,  # "block_density". Human block density, valid when "dynamic_hb" is True. block_density = 1, means that if "dynamic_hb" is True always will be considered hb prenetration attenuation. block_density = 0, means =mpropbability of HB.
                "real",  # "channel_type": ("real", "awgn"). Used for the link to sstem adaptation proccess and computing the users CQI feedback or using the real BLER (Block Error Rate) curves or the AWGN curves. When FF and Shadowing is considered the chanel type must be always real.
                0.1  # "target_bler": typical values (0.1, 0.01). Is the target bler for selecting the CQI of the user.
            ]])
        general_parameters.columns = ["thermal_noise", "h_ceiling", "block_density", "channel_type", "target_bler"]

        # bs_parameters = pd.DataFrame([[50, 50, 25, "tbs", "UMa", "three_sectors", 2.4, 2, 50, 20, 10, 2, 7, 15, None],
        #                               [25, 25, 10, "tbs", "UMi", "three_sectors", 28, 2, 50, 10, 10, 2, 7, 15, None],
        #                               [75, 75, 10, "tbs", "UMi", "three_sectors", 28, 2, 50, 10, 10, 2, 7, 15, None],
        #                               [75, 25, 100, "abs", "A2G", "three_sectors", 28, 2, 50, 10, 10, 2, 7, 15, None],
        #                               [25, 75, 100, "abs", "A2G", "three_sectors", 28, 2, 50, 10, 10, 2, 7, 15, None],
        #                               [None, None, 8000, "sat", "HAPS", "Sat_ax", 28, 2, 50, 10, 10, 2, 7, None, 50]])
        bs_parameters = pd.DataFrame([
                                      [50, 50, 25, "tbs", "UMa", "three_sectors", "dual", "E", "B", 28, 2, 1, 20, 20, 2, 7, 15, None],
                                      [75, 75, 10, "tbs", "UMi", "three_sectors", "dual", "D", "A", 28, 2, 1, 10, 10, 2, 7, 15, None],
                                      [25, 25, 10, "tbs", "UMi", "three_sectors", "dual", "D", "A", 28, 2, 1, 10, 10, 2, 7, 15, None],
                                      # [39.2337738, 9.12153844, 50000, "sat", "HAPS", "Sat_ax", "dual", "C_ntn", "A_ntn", 2, 2, 50, 36, 30, 2, 7, None, 75],
                                      # [39.2137738, 9.1153844, 50000, "sat", "HAPS", "Sat_ax", "dual", "C_ntn", "A_ntn", 2, 2, 50, 36, 30, 2, 7, None, 90],
                                      ])
        # fast_fading_los_type: TN_los: "D", "E"; NTN: "C_ntn", "D_ntn"
        # fast_fading_nlos_type: TN_nlos: "A", "B", "C"; NTN: "A_ntn", "B_ntn"
        # NTN: "A100_ntn", "C5_ntn"
        bs_parameters.columns = ["x",  # For "tbs"/"abs": Int value.  x coordinate (in m) of the BS. For "sat": latitude of the satellite. It defines the elevation angle of the satellite regarding the center of the grid (in degrees).
                                 "y",  # For "tbs"/"abs": Int value. y coordinate (in m) of the BS. For "sat": longitude of the satellite. It defines the elevation angle of the satellite regarding the center of the grid (in degrees).
                                 "z",  # For "tbs"/"abs"/"sat": Int value. z coordinate (in m) of the BS (height of BS). It defines the elevation angle of the satellite regarding the center of the grid (in degrees).
                                 "type",  # ("tbs", "abs", "sat"). For defining if the BS is a terrestrial (TN) BS, or an Aerial BS (a BS on top of a UAV), or a satellite NTN.
                                 "scenario",  # for tbs:("UMi", "UMa", "RMa", "InH-Mixed", "InH-Open", "InF-HH", "InF-SL", "InF-DL", "InF-SH", "InF-DH", "D2D"), for abs ("A2G"), for sat ("HAPS", "LEO", "MEO"). TODO for the NTN (type = sat) this is not considered.
                                 "antenna_mode", # for tbs and abs: ("omni", "three_sectors", "four_sectors", "one_sectors_90_degrees"), for sat: ("omni", "Sat_ax").
                                 "ax_panel_polarization",  # ("single", "dual"). For considering an antenna with single or dual polarization elements.
                                 "fast_fading_los_type",  # ("D", "E"; NTN: "C_ntn", "D_ntn"), according to 3GPP TR 38.901/38.811.
                                 "fast_fading_nlos_type",  # ("A", "B", "C"; NTN: "A_ntn", "B_ntn"), according to 3GPP TR 38.901/38.81.
                                 "fc",  # float from 0.5 to 100 Gigahertz (frequency of the BS).
                                 "numerology",  # (0, 1, 2, 3, 4) numerology of the BS according to 5G NR, 3GPP TS 38.214.
                                 "n_rb",  # Int with the number of physical resource blocks (RBs).
                                 "p_tx",  # Transmission (tx) power of the BS in dB.
                                 "ax_gain",  # Antenna (ax) gain of the BS in dBi.
                                 "cable_loss",  # Cable loss, default value 2 dB.
                                 "noise_figure",  # Noise Figure, default value 7 dB.
                                 "v_tilt",  # vertical tilt of the BS antenna, default value 15 degrees.
                                 "desired_elevation_angle"  # desired elevation angle of the satellite regarding the center of the grid (in degrees). Just used for comparison with real elevation angle configure to the Sat from theri LLA coordinates and the grid coordinates.
                                 ]



        sub_groups_parameters = pd.DataFrame(
            [["pedestrian", 10, "omni", 0, 0, 0, 7, True, True, "[1, 1]", "[50, 50]", "[0.4, 1.2]", 1, "Random Waypoint", None, None, "[1.5, 1.5]", "urban"],
             # ["pedestrian", 10, "three_sectors", 0, 0, 0, 7, True, False, "[0.5, 0.5]", "[150, 150]", "[1, 5]", 1, "Random Direction model", None, None, "[1.5, 1.5]", "urban"],
             # ["car_mounted", 1, "three_sectors", 10, 10, 0, 7, True, False, "[1, 0.2]", "[50, 75]", "[10, 15]", 1, "Random Waypoint", None, None, "[1.5, 1.5]", "urban"],
             # ["iot", 1, "three_sectors", 10, 10, 0, 7, True, False, "[0.25, 0.25]", "[75, 75]", "[0.0001, 0.0002]", 1, "Random Waypoint", None, None, "[1.5, 1.5]", "urban"]
             ])

        sub_groups_parameters.columns = ["type",  # ("pedestrian", "car_mounted", "iot"). These classifications are only used to identify among three possible kinds of simulated EDs.
                                         "k_sub",  # Int, the number of EDs to simulate.
                                         "antenna_mode",  # ("omni", "three_sectors")
                                         "p_tx",  # Transmission (tx) power of the ED in dB.
                                         "ax_gain",  # Antenna (ax) gain of the BS in dBi. The ax_gain is considered a 0 if the antenna_mode = "omni".
                                         "cable_loss",  # Cable loss, default value 2 dB.
                                         "noise_figure",  # Noise Figure, default value 7 dB.
                                         "d2d",  # Boolean for enabling the D2D capability of this EDs
                                         "fixed_height",  # Boolean for enabling an scenario definition where all the ED have the same and fixed height. fixed_height = true min_max_height[mg][0] == min_max_height[mg][1].
                                         "grid_size_ratio",  # Array of two Double parameters (between 0 and 1) (grid_size_ratio[0], grid_size_ratio[1]). grid_size_ratio[0], grid_size_ratio[0] = 1 means that the users will be randomly distributed in 100% of the grid regarding the x and y coordinates. grid_size_ratio[0], grid_size_ratio[0] = 0.1 means that the users will be randomly distributed in 10% of the grid regarding the x and y coordinates. The x grid_size_ratio[0] and y grid_size_ratio[1] percents that can be configured independently.
                                         "reference_location",  # Array of two int values (reference_location[0], reference_location[1]), for defining the center reference location where the users will be simulated. x = reference_location[0], y = reference_location[1].
                                         "min_max_velocity",  # Array of two float values (min_max_velocity[0], min_max_velocity[1]), for defining the minimun and maximun velocity (m/s) that the users could experience.
                                         "wait_time",  # Float value for defining the wait time (s) that the users with a certain mobility could experience. It means that the users in their trajectory could be static for the defined "wait_time".
                                         "mobility_model",  # ("Random Static", "Random Walk", "Random Waypoint", "Truncated Levy Walk model", "Random Direction model", "Gauss-Markov model", "Reference Point Group model"), s.t., https://github.com/panisson/pymobility.
                                         "aggregation",  # Double, parameter (between 0 and 1). The parameter 'aggregation' controls how close the nodes are to the group center. It is only valid for "Reference Point Group model".
                                         "number_mg_rpg_model", # The parameter 'number_mg_rpg_model' is an Int value that define the number of sub-groups, k_sub/number_mg_rpg_model must be integer. It is only valid for "Reference Point Group model".
                                         "min_max_height",  # Array of two float values (min_max_height[0], min_max_height[1]), for defining the min and max height of the EDs. When fixed_height is equal true min_max_height[0] = min_max_height[1].
                                         "rx_scenario"  # ("dense urban", "urban", "suburban", "rural"). This is only used for type = sat.
                                         ]

        if save_simulation_settings:
            general_simulation_parameters.to_excel("simulation_settings/general_simulation_parameters.xlsx")
            general_channel_modeling.to_excel("simulation_settings/general_channel_modeling.xlsx")
            general_parameters.to_excel("simulation_settings/general_parameters.xlsx")
            bs_parameters.to_excel("simulation_settings/bs_parameters.xlsx")
            sub_groups_parameters.to_excel(f"simulation_settings/sub_groups_parameters.xlsx")

        general_simulation_parameters["grid_xy"] = general_simulation_parameters["grid_xy"].apply(ge.convert_grid_xy)
        sub_groups_parameters["grid_size_ratio"] = sub_groups_parameters["grid_size_ratio"].apply(ge.convert_grid_xy)
        sub_groups_parameters["reference_location"] = sub_groups_parameters["reference_location"].apply(ge.convert_grid_xy)
        sub_groups_parameters["min_max_velocity"] = sub_groups_parameters["min_max_velocity"].apply(ge.convert_grid_xy)
        sub_groups_parameters["min_max_height"] = sub_groups_parameters["min_max_height"].apply(ge.convert_grid_xy)

    elif simulation_settings == "from_excel":

        general_simulation_parameters = pd.read_excel("simulation_settings/general_simulation_parameters.xlsx")
        general_channel_modeling = pd.read_excel("simulation_settings/general_channel_modeling.xlsx")
        general_parameters = pd.read_excel("simulation_settings/general_parameters.xlsx")
        bs_parameters = pd.read_excel("simulation_settings/bs_parameters.xlsx")
        sub_groups_parameters = pd.read_excel("simulation_settings/sub_groups_parameters.xlsx")

        general_simulation_parameters["grid_xy"] = general_simulation_parameters["grid_xy"].apply(ge.convert_grid_xy)
        sub_groups_parameters["grid_size_ratio"] = sub_groups_parameters["grid_size_ratio"].apply(ge.convert_grid_xy)
        sub_groups_parameters["reference_location"] = sub_groups_parameters["reference_location"].apply(ge.convert_grid_xy)
        sub_groups_parameters["min_max_velocity"] = sub_groups_parameters["min_max_velocity"].apply(ge.convert_grid_xy)
        sub_groups_parameters["min_max_height"] = sub_groups_parameters["min_max_height"].apply(ge.convert_grid_xy)

        print(general_simulation_parameters)
        print(general_channel_modeling)
        print(general_parameters)
        print(bs_parameters)
        print(sub_groups_parameters)

    df_x, df_y, df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, grid_lla, grid_xy = sx.scenario_definition(general_simulation_parameters,
                                                                                bs_parameters, general_channel_modeling,
                                                                                sub_groups_parameters,
                                                                                general_parameters)

    metrics_dic_bs_dl = 0
    metrics_dic_bs_ul = 0
    metrics_dic_d2d = 0

    out.scenario_outputs(general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                    df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, metrics_dic_bs_dl, metrics_dic_bs_ul, metrics_dic_d2d)

    if general_simulation_parameters["downlink"][0]:
        metrics_dic_bs_dl = lc.link_computation_bs2d_dl(bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                        df_z, time_map, grid_lla, grid_xy)
        out.bs2d_dl_outputs(general_simulation_parameters, bs_parameters, general_channel_modeling,
                            sub_groups_parameters, general_parameters, df_x, df_y,
                            df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, metrics_dic_bs_dl, metrics_dic_bs_ul,
                            metrics_dic_d2d)

    if general_simulation_parameters["uplink"][0]:
        metrics_dic_bs_ul = lc.link_computation_bs2d_ul(bs_parameters, general_channel_modeling, sub_groups_parameters,
                                                        general_parameters, df_x, df_y,
                                                        df_z, time_map, grid_lla, grid_xy)
        out.bs2d_ul_outputs(general_simulation_parameters, bs_parameters, general_channel_modeling,
                            sub_groups_parameters, general_parameters, df_x, df_y,
                            df_z, df_tbs_xyz, df_abs_xyz, time_map, metrics_dic_bs_dl, metrics_dic_bs_ul,
                            metrics_dic_d2d)

    if general_simulation_parameters["d2d_link"][0]:
        metrics_dic_d2d = lc.link_computation_d2d(bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                            df_z, time_map, grid_lla, grid_xy)
        out.d2d_outputs(general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters, df_x, df_y,
                    df_z, df_tbs_xyz, df_abs_xyz, time_map, metrics_dic_bs_dl, metrics_dic_bs_ul, metrics_dic_d2d)
