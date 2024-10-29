# import numpy as np
#
# # Constants
# R_EARTH = 6378137.0  # Earth's radius in meters (WGS 84)
# ECCENTRICITY = 0.08181919  # Earth's eccentricity (WGS 84)
#
# # Desired elevation angle in degrees
# desired_elevation_angle = 80.0  # degrees
#
# # Grid center in LLA (latitude, longitude)
# grid_center_lat = 45.0  # degrees
# grid_center_lon = 45.0  # degrees
#
#
# # Convert latitude, longitude to ECEF coordinates
# def lla_to_ecef(lat, lon, alt):
#     lat_rad = np.radians(lat)
#     lon_rad = np.radians(lon)
#     N = R_EARTH / np.sqrt(1 - ECCENTRICITY ** 2 * np.sin(lat_rad) ** 2)
#
#     X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
#     Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
#     Z = (N * (1 - ECCENTRICITY ** 2) + alt) * np.sin(lat_rad)
#
#     return X, Y, Z
#
#
# # Convert ECEF coordinates to LLA (latitude, longitude, altitude)
# def ecef_to_lla(X, Y, Z):
#     lon = np.arctan2(Y, X)
#     p = np.sqrt(X ** 2 + Y ** 2)
#     theta = np.arctan2(Z * R_EARTH, p * (R_EARTH * (1 - ECCENTRICITY ** 2)))
#     lat = np.arctan2(Z + ECCENTRICITY ** 2 * R_EARTH * np.sin(theta) ** 3,
#                      p - ECCENTRICITY ** 2 * R_EARTH * np.cos(theta) ** 3)
#     N = R_EARTH / np.sqrt(1 - ECCENTRICITY ** 2 * np.sin(lat) ** 2)
#     alt = p / np.cos(lat) - N
#
#     lat = np.degrees(lat)
#     lon = np.degrees(lon)
#
#     return lat, lon, alt
#
#
# # Compute the satellite position in ECEF for a given elevation angle
# def satellite_position_for_elevation(lat_g, lon_g, alt_g, elevation_angle):
#     # Convert grid center to ECEF
#     X_g, Y_g, Z_g = lla_to_ecef(lat_g, lon_g, alt_g)
#
#     # Elevation angle in radians
#     elevation_rad = np.radians(elevation_angle)
#
#     # Distance from Earth's center to the satellite (r_sat)
#     r_sat = (R_EARTH + alt_g) / np.sin(elevation_rad)
#
#     # Satellite position in ECEF
#     X_s = X_g + r_sat * np.cos(elevation_rad)
#     Y_s = Y_g
#     Z_s = Z_g + r_sat * np.sin(elevation_rad)
#
#     return X_s, Y_s, Z_s
#
#
# # Grid center altitude (0 for simplicity)
# alt_g = 0
#
# # Calculate satellite ECEF coordinates for the desired elevation angle
# X_s, Y_s, Z_s = satellite_position_for_elevation(grid_center_lat, grid_center_lon, alt_g, desired_elevation_angle)
#
# # Convert satellite ECEF coordinates back to LLA
# satellite_lat, satellite_lon, satellite_alt = ecef_to_lla(X_s, Y_s, Z_s)
#
# # Display the satellite LLA coordinates
# print(f"Satellite Position for {desired_elevation_angle}° Elevation Angle:")
# print(f"Latitude: {satellite_lat:.6f}°")
# print(f"Longitude: {satellite_lon:.6f}°")
# print(f"Altitude: {satellite_alt:.2f} meters")
import math as ma

import numpy as np
from numpy import random
from numpy.random import normal
# np.random.default_rng()  # Set the seed for reproducibility
arr = np.array([2, -1, -3, 0, -1, 15, 2, -9])

def clip_outliers(data, lower_percentile=5, upper_percentile=95):
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return np.clip(data, lower_bound, upper_bound)

# Example usage:
clipped_arr = clip_outliers(arr)
print(clipped_arr)