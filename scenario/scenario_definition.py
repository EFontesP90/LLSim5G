"""
File: scenario_definition.py

Purpose:
This file allows to define the whole grid and mobility behaviours. Its outputs are the coordinates
of each user in the grid as well as the coordinates of the base stations.

Author: Ernesto Fontes Pupo / Claudia Carballo González
Date: 2024-10-30
Version: 1.0.0
SPDX-License-Identifier: Apache-2.0

"""

# Standard library imports
import logging

# Third-party imports
import numpy as np
import pandas as pd
from numpy import random
from datetime import datetime

# Local application/library-specific imports
from scenario.mobility import models as mob
import scenario.show as sw
import scenario.sattelites_lla_info as sat_pos


def ColumnName(df):  
    name = np.empty([df.shape[1]],
                    dtype=int)  # array of the resulting CQI of the K users regarding the BS and each proximity user in D2D communication.
    for i in range(df.shape[1]):
        # name[i] = f"{i}"
        if i == 0:
            name[i] = 0
        else:
            name[i] = i

    df.columns = name
    return df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Scenario(object):

    """
    25/04/2024
    The class Scenario allows to define the whole grid and mobility behaviours. Its outputs are the coordinates
    of each user in the grid as well as the coordinates of the base stations. If enabled, this class
    generate (in the output file) the .xlsx files with coordinates of all the elements in the grid, and the video (mp4, gif, or both)
    for representing the defined grid and mobility patterns. TODO, to enable the mp4 option for the GitHub version.

    Required attributes:
    (general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters):

    Returns (scenario_definition):
        df_x: dataframe (simulation_steps x number_ue +1) with the x coordinates of the defined EDs (number_ue)
        df_y: dataframe (simulation_steps y number_ue +1) with the x coordinates of the defined EDs (number_ue)
        df_z: dataframe (simulation_steps z number_ue +1) with the x coordinates of the defined EDs (number_ue)
        df_tbs_xyz: dataframe  with the (x, y, z) coordinates of the defined tbs
        df_abs_xyz: dataframe  with the (x, y, z) coordinates of the defined abs
        df_sat_lla: dataframe  with the (l, l, a) coordinates of the defined sat
        time_map: dataframe  with the time steps of the simulation
        grid_lla: dataframe  with the lla of the center of the grid, jus used for satellite simulations
        grid_xy: array with the x and Y onf the simulated grid (in m)

    """

    def __init__(self, general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters):

        grid_xy = general_simulation_parameters["grid_xy"][0]
        grid_center_latitude = general_simulation_parameters["grid_center_latitude"][0]
        grid_center_longitude = general_simulation_parameters["grid_center_longitude"][0]

        self.grid_xy = grid_xy  # Tuple with the size x and y of the grid
        self.grid_center_latitude = grid_center_latitude
        self.grid_center_longitude = grid_center_longitude

        show_video = general_simulation_parameters["show_video"][0]
        save_video = general_simulation_parameters["save_video"][0]
        save_scenario_xlsx = general_simulation_parameters["save_scenario_xlsx"][0]
        video_format = general_simulation_parameters["video_format"][0]  # mp4, gif, Both
        assert video_format == "gif", f"At the moment the video format (in the general_simulation_parameters input dataframe) can be just saved as .gif format!"
        video_velocity = general_simulation_parameters["video_velocity"][0]
        simulation_time = general_simulation_parameters["simulation_time"][0]
        simulation_resolution = general_simulation_parameters["simulation_resolution"][0]
        #################################

        number_bs = bs_parameters.shape[0]
        tbs_flag = False
        abs_flag = False
        sat_flag = False
        for bs in range(number_bs):
            type = bs_parameters["type"][bs]
            if type == 'tbs': tbs_flag = True
            elif type == 'abs': abs_flag = True
            elif type == 'sat': sat_flag = True


        if tbs_flag:
            number_tbs = bs_parameters['type'].value_counts()['tbs']
            tbs_coord_xyz = bs_parameters.loc[bs_parameters['type'] == 'tbs', ['x', 'y', 'z']].to_numpy().reshape(
                number_tbs, 3)
            self.tbs_coord_xyz = tbs_coord_xyz  # Array (number_tbs x 3) with the x,y,z coordinates of the terrestrial BSs (tbs), e.g., [[50, 50, 10], [25,25,10], ... [x,y,z]]

        else:
            number_tbs = 0
            self.tbs_coord_xyz = [None, None, None]

        if abs_flag:
            number_abs = bs_parameters['type'].value_counts()['abs']
            abs_coord_xyz = bs_parameters.loc[bs_parameters['type'] == 'abs', ['x', 'y', 'z']].to_numpy().reshape(
            number_abs, 3)
            self.abs_coord_xyz = abs_coord_xyz  # Array (number_abs x 3) with the initial x,y,z coordinates of the aerial BSs (abs), e.g., [[50, 50, 100], [25,25,100], ... [x,y,z]]
            abs_mobility_model = None
            self.abs_mobility_model = abs_mobility_model  # TODO Array (number_abs x 1) with the mobility model of the abs, e.g., [static, personalized_mobility, ... , etc]
        else:
            number_abs = 0
            self.abs_coord_xyz = [None, None, None]

        if sat_flag:
            number_sat = bs_parameters['type'].value_counts()['sat']
            sat_coord_lla = bs_parameters.loc[bs_parameters['type'] == 'sat', ['x', 'y', 'z']].to_numpy().reshape(
            number_sat, 3)
            desired_elevation_angle = bs_parameters.loc[bs_parameters['type'] == 'sat', 'desired_elevation_angle'].to_numpy()
            self.desired_elevation_angle = desired_elevation_angle
            elevation_angle_grid_center = np.zeros(np.shape(desired_elevation_angle))

            for s in range(number_sat):
                # Satellite position in LLA (latitude, longitude, altitude)
                satellite_lat = sat_coord_lla[s][0]  # degrees
                satellite_lon = sat_coord_lla[s][1]  # degrees
                satellite_alt = sat_coord_lla[s][2]  # altitude in meters

                # Grid center in LLA
                grid_center_lat = self.grid_center_latitude  # degrees
                grid_center_lon = self.grid_center_longitude  # degrees
                grid_center_alt = 0
                # grid_size = 1000.0  # grid size in meters

                elevation_angle_grid_center[s] = sat_pos.sat_elevation_angle_from_lla(satellite_lat, satellite_lon, satellite_alt,
                                                                                      grid_center_lat, grid_center_lon, grid_center_alt)
                if abs(elevation_angle_grid_center[s] - desired_elevation_angle[s]) > 5:
                    # print("testest", abs(elevation_angle_grid_center[s] - desired_elevation_angle[s]))
                    logger.info("The resulting elevation angle %s of the satellite number %s is more than 5 degrees different regarding the desired elevation angle (%s degrees)", elevation_angle_grid_center[s], s, desired_elevation_angle[s])
                    logger.info("Modify the LLA coordinates of the satellite to obtain the desired elevation angle")


            self.elevation_angle_grid_center = elevation_angle_grid_center

            # print("elevation_angle_grid_center")
            # print(elevation_angle_grid_center)

            self.sat_coord_lla = sat_coord_lla  # Array (number_sat x 3) with the initial x,y,z coordinates of the sat (sat), e.g, [[0, 0, 50000], [0, 0, 8000], ... [x,y,z]]
            sat_mobility_model = None
            self.sat_mobility_model = sat_mobility_model  # TODO Array (number_sat x 1) with the mobility model of the sat, e.g, [static, personalized_mobility, ... , etc]
        else:
            number_sat = 0
            self.sat_coord_lla = [None, None, None]
            self.desired_elevation_angle = None
            self.elevation_angle_grid_center = None

        abs_mobility_model = None

        number_ue = sub_groups_parameters['k_sub'].sum()
        # fixed_height_ue = 1.5
        number_mg = sub_groups_parameters.shape[0]
        mg_members = sub_groups_parameters['k_sub'].values
        sub_types = sub_groups_parameters['type'].values

        mg_grid_size_ratio = np.vstack(sub_groups_parameters['grid_size_ratio'])
        mg_reference_location = np.vstack(sub_groups_parameters['reference_location'])
        mg_min_max_velocity = np.vstack(sub_groups_parameters['min_max_velocity'])
        mg_wait_time = np.vstack(sub_groups_parameters['wait_time'])
        mg_mobility_model = sub_groups_parameters[
            'mobility_model'].values  # Random Static, Random Waypoint, Random Walk, Truncated Levy Walk model, Random Direction model, Gauss-Markov model, Reference Point Group model, Time-variant Community Mobility Model
        mg_aggregation = sub_groups_parameters['aggregation'].values
        number_mg_rpg_model = sub_groups_parameters['number_mg_rpg_model'].values
        fixed_height_ue = sub_groups_parameters['fixed_height'].values
        min_max_height = np.vstack(sub_groups_parameters['min_max_height'])

        if number_tbs > 0:
            assert number_tbs == np.array(tbs_coord_xyz).shape[0], f"The number of TBS {number_tbs} is not equal to the entered TBS coordinates ({np.array(tbs_coord_xyz).shape[0]})!"
        if number_abs > 0:
            assert number_abs == np.array(abs_coord_xyz).shape[0], f"The number of ABS {number_abs} is not equal to the entered ABS coordinates ({np.array(abs_coord_xyz).shape[0]})!"
        if number_sat > 0:
            assert number_sat == np.array(sat_coord_lla).shape[0], f"The number of ABS {number_sat} is not equal to the entered ABS coordinates ({np.array(sat_coord_lla).shape[0]})!"

        assert number_ue == sum(mg_members), f"The number of multicast group (MG) members: {sum(mg_members)} must be equal to the total number of users (UEs): {number_ue}!"
        assert np.shape(np.array(mg_members))[0] == number_mg, f"The MG members array (mg_members) must have the same dimensions that the number of MGs: {number_mg}!"
        assert np.shape(np.array(mg_reference_location))[0] == number_mg, f"The MGs reference location array (mg_reference_location) must have the same dimensions that the number of MGs: {number_mg}!"
        assert np.shape(np.array(mg_grid_size_ratio))[0] == number_mg, f"The MGs grid size ratio array (mg_grid_size_ratio) must have the same dimensions that the number of MGs: {number_mg}!"
        assert np.shape(np.array(mg_min_max_velocity))[0] == number_mg, f"The MGs min-max velocity array (mg_min_max_velocity) must have the same dimensions that the number of MGs: {number_mg}!"
        assert np.shape(np.array(mg_wait_time))[0] == number_mg, f"The MGs wait time array (mg_wait_time) must have the same dimensions that the number of MGs: {number_mg}!"
        assert np.shape(np.array(mg_mobility_model))[0] == number_mg, f"The MGs mobility models array (mg_mobility_model) must have the same dimensions that the number of MGs: {number_mg}!"

        for mg in range(number_mg):

            assert mg_reference_location[mg][0] <= grid_xy[0]*(1-mg_grid_size_ratio[mg][0]/2), f"The selected reference location for the MG = {mg + 1} must be lower than or equal to: X = {grid_xy[0]*(1-mg_grid_size_ratio[mg][0]/2)}, according to the configured values of grid size and MG grid size ratio!"
            assert mg_reference_location[mg][0] >= grid_xy[0]*mg_grid_size_ratio[mg][0]/2, f"The selected reference location for the MG = {mg + 1} must be higher than or equal to: X = {grid_xy[0]*mg_grid_size_ratio[mg][0]/2}, according to the configured values of grid size and MG grid size ratio!"

            assert mg_reference_location[mg][1] <= grid_xy[1]*(1 - mg_grid_size_ratio[mg][1]/2), f"The selected reference location for the MG = {mg + 1} must be lower than or equal to: Y = {grid_xy[1]*(1-mg_grid_size_ratio[mg][1]/2)}, according to the configured values of grid size and MG grid size ratio!"
            assert mg_reference_location[mg][1] >= grid_xy[1]*mg_grid_size_ratio[mg][1]/2, f"The selected reference location for the MG = {mg + 1} must be higher than or equal to: X = {grid_xy[1]*mg_grid_size_ratio[mg][1]/2}, according to the configured values of grid size and MG grid size ratio!"

            if mg_mobility_model[mg] == "Reference Point Group model":

                # assert type(mg_members[mg] / number_mg_rpg_model[mg]) == int, f"Hola"
                assert 0 <= mg_aggregation[mg] <= 1, \
                    f"The defined MG aggregation level {mg_aggregation[mg]} must be a number between 0 (minimun agreggation) and 1 (maximun aggregation)! (this parameter is only valid for: Reference Point Group model)"

                # assert type(mg_members[mg]/number_mg_rpg_model[mg]) == int, f"The defined MG aggregation level ({mg_aggregation[mg]}) must be a number between 0 (minimun agreggation) and 1 (maximun aggregation)! (this parameter is only valid for: Reference Point Group model)"
            assert mg_mobility_model[mg] != "Time-variant Community Mobility Model", f"The mobility model: {mg_mobility_model[mg]}, is no correctly implemented, please select another option!"

            if fixed_height_ue[mg]:
                assert min_max_height[mg][0] == min_max_height[mg][1], f"If fixed_height_ue is: {fixed_height_ue[mg]}, for the subgroup:{mg+1} the two min_max_height values ({min_max_height[mg]}) must be equal"

        self.general_simulation_parameters = general_simulation_parameters  # ["grid_xy", "grid_center_latitude", "grid_center_longitude", "simulation_time", "simulation_resolution", "downlink", "uplink", "d2d_link", "save_scenario_xlsx", "save_metrics_xlsx", "show_video", "save_video", "video_format", "video_velocity", "print_scenario_outputs", "print_metrics_outputs"]
        self.bs_parameters = bs_parameters  # [number of bs]x["x", "y", "z", "type", "scenario", "antenna_mode", "fc", "numerology", "n_rb", "p_tx", "ax_gain", "cable_loss", "noise_figure"]
        self.general_channel_modeling = general_channel_modeling  # 1x["dynamic_los", "dynamic_hb", "o2i", "inside_what_o2i", "penetration_loss_model", "shadowing", "fast_fading", "fast_fading_model"]
        self.sub_groups_parameters = sub_groups_parameters  # [number of defined user subgroups]x["type", "k_sub", "antenna_mode", "p_tx", "ax_gain", "cable_loss", "noise_figure", "d2d", "fixed_height", "grid_size_ratio", "reference_location", "min_max_velocity", "wait_time", "mobility_model", "aggregation", "number_mg_rpg_model", "min_max_height"]
        self.general_parameters = general_parameters  # 1x["thermal_noise", "h_ceiling", "block_density"]



        self.save_scenario_xlsx = save_scenario_xlsx
        self.show_video = show_video  # Boolean variable to enable or not the scenario video display.
        self.save_video = save_video  # Boolean variable to enable or not saving the scenario video.
        self.video_format = video_format # String variable to define the video format for saving the scenario video. The options are: mp4, gif, both (TODO to enable mp4 for GitHub version).
        self.video_velocity = video_velocity  # To adjust the velocity of the video.
        self.simulation_time = simulation_time  # To adjust the simulation time, e.g., 60 seconds.
        self.simulation_resolution = simulation_resolution  # To adjust the simulation resolution along the time, e.g., 0.1 seconds.

        self.number_tbs = number_tbs  # Number of Terrestrial BSs
        self.number_abs = number_abs  # Number of Aerial BS (UAV)
        self.number_sat = number_sat  # Number of Sat BS (Sat)
        # self.tbs_coord_xyz = tbs_coord_xyz  # Array (number_tbs x 3) with the x,y,z coordinates of the terrestrial BSs (tbs), e.g, [[50, 50, 10],[25,25,10], ... [x,y,z]]
        # self.abs_coord_xyz = abs_coord_xyz  # Array (number_abs x 3) with the initial x,y,z coordinates of the aerial BSs (abs), e.g, [[50, 50, 100],[25,25,100], ... [x,y,z]]
        # self.abs_mobility_model = abs_mobility_model  # TODO Array (number_abs x 1) with the mobility model of the abs, e.g, [static, personalized_mobility, ... , etc]

        self.number_ue = number_ue  # Number of user equipments (UE)
        self.fixed_height_ue = fixed_height_ue  # A fixed value of height for all the UEs, (to be considered if it is enabled a posible variable or different users height)
        self.min_max_height = min_max_height
        self.number_mg = number_mg  # Number of multicast groups (MG)
        self.mg_members = mg_members  # Array (number_mg x 1) with the number of MG members (MG members x number_mg = number_ue)
        self.sub_types = sub_types  # Array with the definition of the subgroup of users type
        self.mg_grid_size_ratio = mg_grid_size_ratio  # Array (number_mg x 2) with the grid size ratio (e.g. 20 % of the total x, and 20 % of the total y) covered by each MG regarding the full grid size, e.g., [[0.2, 0.2], [0.2,1], ... [%x,%y]]
        self.mg_aggregation = mg_aggregation  # Array (number_mg x 1) that defines the MGs aggregation, how close are the MG members. It is a value between 0 and 1, where 1 means not aggregation at all.


        self.mg_reference_location = mg_reference_location  # Array (number_mg x 2) that defines the initial reference location of each MG inside the grid, e.g, [[50, 50], [10,20], ... [x,y]]
        self.mg_min_max_velocity = mg_min_max_velocity  # Array (number_mg x 2) that defines the minimum and maximum velocity of the members of each MG in meters per second, e.g, [[0.4, 1], [5,10], ... [x,y]]
        self.mg_wait_time = mg_wait_time  # Array (number_mg x 1) that defines the random possible wait time (in seconds) of each member of the MG
        self.mg_mobility_model = mg_mobility_model  # Array (number_mg x 1) that defines the mobility model that follow the members of ech MG, e.g, [["static"], ["random_walk"], ... [...]]
        self.number_mg_rpg_model = number_mg_rpg_model # Array (number_mg x 1) that is used for the mobility model Reference Point Group model, and it defines in how many subgroups are splitted the MGs with this mobility model

        self.simulation_steps = int(self.simulation_time / self.simulation_resolution) # This value defines the total steps the simulation must follow, e.g., if simulation_time=10 and simulation_resolution=0.1, then simulation_steps = 100

    def scenario(self):
        time_Map = np.zeros([self.simulation_steps], dtype=float)  # Array of the resulting ShadowingMap of the K users respect to the BS and each proximity user in D2D communication, [time stamp, shadowing]

        # rw = {}
        overall_mob_map = []
        # overall_time_map = {}
        for g in range(self.number_mg):
            resolution = self.simulation_resolution
            K_nodes = self.mg_members[g]
            MIN_V = self.mg_min_max_velocity[g][0]
            MAX_V = self.mg_min_max_velocity[g][1]
            MAX_WT = self.mg_wait_time[g]
            MAX_X = self.grid_xy[0] * self.mg_grid_size_ratio[g][0]
            MAX_Y = self.grid_xy[1] * self.mg_grid_size_ratio[g][1]
            sim_step = self.simulation_steps
            SimTime = self.simulation_time
            video_velocity = self.video_velocity
            aggregation = self.mg_aggregation[g]

            mobility_map = np.zeros([self.simulation_steps, K_nodes, 3])  # array of the resulting ShadowingMap of the K users respect to the BS and each proximity user in D2D communication, [time stamp, shadowing]

            if self.mg_mobility_model[g] == "Random Static": rw = mob.random_walk(K_nodes, dimensions=(MAX_X, MAX_Y), velocity=0, distance=0)
            elif self.mg_mobility_model[g] == "Random Walk": rw = mob.random_walk(K_nodes, dimensions=(MAX_X, MAX_Y), velocity=MAX_V * resolution, distance=MAX_V * resolution)
            elif self.mg_mobility_model[g] == "Random Waypoint": rw = mob.random_waypoint(K_nodes, dimensions=(MAX_X, MAX_Y), velocity=(MIN_V * resolution, MAX_V * resolution), wt_max=MAX_WT)
            elif self.mg_mobility_model[g] == "Truncated Levy Walk model": rw = mob.truncated_levy_walk(K_nodes, dimensions=(MAX_X, MAX_Y), FL_EXP=-2.6, FL_MAX=50., WT_EXP=-1.8, WT_MAX=MAX_WT)
            elif self.mg_mobility_model[g] == "Random Direction model": rw = mob.random_direction(K_nodes, dimensions=(MAX_X, MAX_Y), velocity=(MIN_V * resolution, MAX_V * resolution), wt_max=MAX_WT)
            elif self.mg_mobility_model[g] == "Gauss-Markov model": rw = mob.gauss_markov(K_nodes, dimensions=(MIN_V * resolution, MAX_V * resolution), alpha=0.99)
            elif self.mg_mobility_model[g] == "Reference Point Group model":
                groups = [int(K_nodes / self.number_mg_rpg_model[g]) for _ in range(self.number_mg_rpg_model[g])]
                nr_nodes = sum(groups)
                rw = mob.reference_point_group(groups, dimensions=(MAX_X, MAX_Y), aggregation=aggregation,
                                                           velocity=(MIN_V * resolution, MAX_V * resolution))
                step = 0
                STEPS_TO_IGNORE = 10000
                a = False
                for ii in range(STEPS_TO_IGNORE):
                    discard = next(rw)

            elif self.mg_mobility_model[g] == "Time-variant Community Mobility Model": # It is not correctly implementted
                raise Exception(f"The mobility model: Time-variant Community Mobility Model, is no correctly implemented, please select another option!")
                # groups = [4 for _ in range(10)]
                # nr_nodes = sum(groups)
                # rw = mob.tvc(groups, dimensions=(MAX_X, MAX_Y), aggregation=[0.5,0.], epoch=[100,100])
            heigth = random.uniform(self.min_max_height[g][0], self.min_max_height[g][1]) #TODO: for constant heigth over time
            for t in range(sim_step):
                positions = next(rw)
                mobility_map[t][:, 0] = positions[:, 0] + (self.mg_reference_location[g][0] - MAX_X / 2)
                mobility_map[t][:, 1] = positions[:, 1] + (self.mg_reference_location[g][1] - MAX_Y / 2)
                mobility_map[t][:, 2] = heigth   #TODO: If I want to enable that the heigth change over time

                time_Map[t] = t * SimTime / sim_step

            if g == 0: overall_mob_map = mobility_map
            else: overall_mob_map = np.hstack((overall_mob_map, mobility_map))

        if self.show_video:
            #sw.show_scenario(self.grid_xy, self.number_tbs, self.number_abs, sim_step, overall_mob_map, self.tbs_coord_xyz, self.abs_coord_xyz, video_velocity, self.save_video, self.video_format)
            sw.show_scenario(self.grid_xy, self.grid_center_latitude, self.grid_center_longitude, self.number_tbs, self.number_abs, self.number_sat, self.number_mg, self.mg_members, self.sub_types, sim_step, overall_mob_map,  self.tbs_coord_xyz,  self.abs_coord_xyz, self.sat_coord_lla, self.elevation_angle_grid_center, self.desired_elevation_angle, video_velocity,  self.save_video,  self.video_format)

        df_xpos = pd.DataFrame(overall_mob_map.T[0]).T
        df_xpos.insert(0, 't', time_Map)
        df_xpos = ColumnName(df_xpos)
        # print(dfXpos)
        df_ypos = pd.DataFrame(overall_mob_map.T[1]).T
        df_ypos.insert(0, 't', time_Map)
        df_ypos = ColumnName(df_ypos)

        df_zpos = pd.DataFrame(overall_mob_map.T[2]).T
        df_zpos.insert(0, 't', time_Map)
        df_zpos = ColumnName(df_zpos)

        if self.number_tbs > 0:
            tbs_identifier = np.arange(1, self.number_tbs + 1)
            df_tbs_xyz = pd.DataFrame(self.tbs_coord_xyz)
            df_tbs_xyz.insert(0, 'TBS', tbs_identifier)
            df_tbs_xyz.columns = ['TBS',"X", "Y", "Z"]
        else:
            df_tbs_xyz = pd.DataFrame([[0,0,0]])
            df_tbs_xyz.insert(0, 'TBS', 0)
            df_tbs_xyz.columns = ['TBS',"X", "Y", "Z"]

        if self.number_abs > 0:
            abs_identifier = np.arange(1, self.number_abs + 1)
            df_abs_xyz = pd.DataFrame(self.abs_coord_xyz)
            df_abs_xyz.insert(0, 'ABS', abs_identifier)
            df_abs_xyz.columns = ['ABS', "X", "Y", "Z"]
        else:
            df_abs_xyz = pd.DataFrame([[0,0,0]])
            df_abs_xyz.insert(0, 'ABS', 0)
            df_abs_xyz.columns = ['ABS', "X", "Y", "Z"]

        if self.number_sat > 0:
            sat_identifier = np.arange(1, self.number_sat + 1)
            df_sat_lla = pd.DataFrame(self.sat_coord_lla)
            df_sat_lla.insert(0, 'SAT', sat_identifier)
            df_sat_lla.columns = ['SAT', "Latitude", "longitude", "Altitude"]
            df_sat_lla["Elevation angle"] = self.elevation_angle_grid_center
        else:
            df_sat_lla = pd.DataFrame([[0,0,0]])
            df_sat_lla.insert(0, 'SAT', 0)
            df_sat_lla.columns = ['SAT',"Latitude", "longitude", "Altitude"]
            df_sat_lla["Elevation angle"] = 0

        # if self.save_scenario_xlsx:
        #     current_date = datetime.now().strftime("%Y%m%d_%H%M")
        #     df_xpos.to_excel(f"output/scenario/dfXpos_{current_date}.xlsx")
        #     df_ypos.to_excel(f"output/scenario/dfYpos_{current_date}.xlsx")
        #     df_zpos.to_excel(f"output/scenario/dfZpos_{current_date}.xlsx")
        #     df_tbs_xyz.to_excel(f"output/scenario/df_tbs_xyz_{current_date}.xlsx")
        #     df_abs_xyz.to_excel(f"output/scenario/df_abs_xyz_{current_date}.xlsx")  #TODO: Important if we enable the abs movement capability
        grid_lla = [self.grid_center_latitude, self.grid_center_longitude, 0]

        return df_xpos, df_ypos, df_zpos, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_Map, grid_lla, self.grid_xy

def scenario_definition(general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters):

    sx = Scenario(general_simulation_parameters, bs_parameters, general_channel_modeling, sub_groups_parameters, general_parameters)

    df_x, df_y, df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, grid_lla, grid_xy = sx.scenario()
    return df_x, df_y, df_z, df_tbs_xyz, df_abs_xyz, df_sat_lla, time_map, grid_lla, grid_xy

