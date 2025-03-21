"""
File: show.py

Purpose:
This file defines the methods for showing and saving in mp4 and gif the video of the defined grid components
and mobility patterns.


Author: Ernesto Fontes Pupo / Claudia Carballo González
        University of Cagliari
Date: 2024-10-30
Version: 1.0.0
                   GNU LESSER GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

    LLSim5G is a link-level simulator for HetNet 5G use cases.
    Copyright (C) 2024  Ernesto Fontes, Claudia Carballo

"""

# Third-party imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter



def show_scenario(grid_xy, grid_center_latitude, grid_center_longitude, number_tbs, number_abs, number_sat, number_mg, sub_members, sub_types, sim_step, overall_mob_map, tbs_coord_xyz, abs_coord_xyz, sat_coord_lla, elevation_angle_grid_center, desired_elevation_angle, video_velocity, save_video, video_format):

    """
    25/04/2024
    Method for showing and saving in mp4 and gif the video for representing the defined grid and mobility patterns.

    Required arguments:

      *grid_xy*:
        Tuple with X and Y size of the recreated grid e.g., (100, 100).

      *number_tbs*:
        Int with the number of terrestrial-base-stations (tbs).

      *number_abs*:
        Int with the number of aerial-base-stations (abs).

      *sim_step*:
        Int with the simulation steps equal to_: int(self.simulation_time / self.simulation_resolution) .

      *overall_mob_map*:
        A matrix with the x and y coordinates of each user in the grid

      *tbs_coord_xyz*
        A matrix with the x and y coordinates of each tbs in the grid

      *abs_coord_xyz*
        A matrix with the x and y coordinates of each abs in the grid

      *video_velocity*
        A float used to adjust the velocity of the video, by default = 0.1
    """
    matplotlib.rcParams['animation.ffmpeg_path'] = "D:\\PostDocTrabajo\\LLS 5G-MBS-BF\\lls_mbs\\scenario\\ffmpeg\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe"
    # matplotlib.rcParams['animation.ffmpeg_path'] = "D:\\PostDocTrabajo\\LLS 5G-MBS-BF\\lls_mbs\\scenario\\ffmpeg.exe"
    plt.figure(figsize=(10, 7))
    plt.xlim(0, grid_xy[0])
    plt.ylim(0, grid_xy[1])
    max_xy = max(grid_xy[0], grid_xy[1])
    ax = plt.subplot(111)
    p = 0
    c = 0
    i = 0
    for g in range(number_mg):
        if sub_types[g] == "pedestrian":
            if p == 0: line_p, = ax.plot(range(max_xy), range(max_xy), label="pedestrian_ue", linestyle='', marker='.', markersize=8)
            p = p + sub_members[g]
        elif sub_types[g] == "car_mounted":
            if c == 0: line_c, = ax.plot(range(max_xy), range(max_xy), label="car_mounted_ue", linestyle='', marker='s', markersize=4)
            c = c + sub_members[g]
        elif sub_types[g] == "iot":
            if i == 0: line_i, = ax.plot(range(max_xy), range(max_xy), label="iot_devices", linestyle='', marker='*', markersize=6)
            i = i + sub_members[g]

    #else: line, = ax.plot(range(max_xy), range(max_xy), label="UE", linestyle='', marker='.', markersize=8)
    if number_tbs > 0: lineTBSs, = ax.plot(range(max_xy), range(max_xy), label="TBS", linestyle='', marker="^", markerfacecolor='black', markersize=12)
    if number_abs > 0: lineABSs, = ax.plot(range(max_xy), range(max_xy), label="ABS", linestyle='', marker=(5, 0), markerfacecolor='black', markersize=10)


    if number_sat > 0:
        if number_sat == 1:
            textstr = f"Sat: Height = {round(sat_coord_lla[0][2], 2)} m, elevation angle = {round(elevation_angle_grid_center[0], 2)}º "
        if number_sat == 2:
            textstr = '\n'.join((
                # f"Sat1: {round(sat_coord_lla[0][0], 2)}º, {round(sat_coord_lla[0][1], 2)}º, {round(sat_coord_lla[0][2], 2)} m, elevation angle =  ",
                # f"Sat2: {round(sat_coord_lla[1][0], 2)}º, {round(sat_coord_lla[1][1], 2)}º, {round(sat_coord_lla[1][2], 2)} m "))
                f"Sat1: Height = {round(sat_coord_lla[0][2], 2)} m, elevation angle = {elevation_angle_grid_center[0]}º ",
                f"Sat2: Height = {round(sat_coord_lla[1][2], 2)} m, elevation angle = {elevation_angle_grid_center[1]}º "))
        if number_sat == 3:
            textstr = '\n'.join((

                f"Sat1: Height = {round(sat_coord_lla[0][2], 2)} m, elevation angle = {elevation_angle_grid_center[0]}º ",
                f"Sat2: Height = {round(sat_coord_lla[1][2], 2)} m, elevation angle = {elevation_angle_grid_center[1]}º ",
                f"Sat3: Height = {round(sat_coord_lla[2][2], 2)} m, elevation angle = {elevation_angle_grid_center[2]}º "))
        if number_sat > 3:
            textstr = '\n'.join((

                f"Sat1: Height = {round(sat_coord_lla[0][2], 2)} m, elevation angle = {elevation_angle_grid_center[0]}º ",
                f"Sat2: Height = {round(sat_coord_lla[1][2], 2)} m, elevation angle = {elevation_angle_grid_center[1]}º ",
                f"Sat3: Height = {round(sat_coord_lla[2][2], 2)} m, elevation angle = {elevation_angle_grid_center[2]}º ",
                f"..."))


        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

    plt.xlabel(f"{grid_xy[0]} meters \n Grid center: lat = {round(grid_center_latitude, 2)}º, log = {round(grid_center_longitude, 2)}º", fontsize=20)
    plt.ylabel(f"{grid_xy[0]} meters", fontsize=20)
    plt.title("Simulation Grid", fontsize=16)
    plt.legend(fontsize=14, title='Legend', title_fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, frameon=False, fancybox=True)

    plt.tight_layout(rect=[0, 0, 0.99, 1])  # Reserve space on the right for the legend

    def update(frame):

        if p > 0: line_p.set_data(overall_mob_map[frame][:p, 0], overall_mob_map[frame][:p, 1])
        if c > 0: line_c.set_data(overall_mob_map[frame][p:p+c, 0], overall_mob_map[frame][p:p+c, 1])
        if i > 0: line_i.set_data(overall_mob_map[frame][p+c:, 0], overall_mob_map[frame][p+c:, 1])

        #line.set_data(overall_mob_map[frame][:, 0], overall_mob_map[frame][:, 1])
        if number_tbs > 0: lineTBSs.set_data(np.array(tbs_coord_xyz)[:, 0], np.array(tbs_coord_xyz)[:, 1])
        if number_abs > 0: lineABSs.set_data(np.array(abs_coord_xyz)[:, 0], np.array(abs_coord_xyz)[:, 1])

    # Change sim_step to the number of frames in your animation
    ani = FuncAnimation(plt.gcf(), update, frames=sim_step, interval=video_velocity * 1000, repeat=True)

    if save_video == True and video_format == "gif":
        # ani.save("recreated_scenario.gif")
        ani.save("output/scenario/recreated_scenario.gif", writer=PillowWriter(fps=10))  # Adjust fps as needed

    elif save_video == True and video_format == "mp4":
        ani.save("output/scenario/recreated_scenario.mp4")

    elif save_video == True and video_format == "both":
        ani.save("output/scenario/recreated_scenario.gif")
        ani.save("output/scenario/recreated_scenario.mp4")

    plt.show()

# Old version

