# import math as ma
# import numpy as np
#
# d2D = 50
# d3D_ref = 100
# h_elevation = 85
# new_h = 180-h_elevation
#
#
# d3D_new = np.sqrt(d3D_ref**2 + d2D**2 - 2*d3D_ref*d2D*np.cos(np.radians(new_h)))
# h_elevation_new = np.arccos((d3D_new**2 + d2D**2 - d3D_ref**2)/(2*d3D_new*d2D))
# print(d3D_new)
# print(np.degrees(h_elevation_new))


import numpy as np

# Constants
R_EARTH = 6378137.0  # Earth's radius in meters (WGS 84)
ECCENTRICITY = 0.08181919  # Earth's eccentricity (WGS 84)

# Satellite position in LLA (latitude, longitude, altitude)
satellite_lat = 39.3137738  # degrees
satellite_lon = 9.1153844  # degrees
satellite_alt = 50000.0  # altitude in meters

# Grid center in LLA
grid_center_lat = 39.2137738  # degrees
grid_center_lon = 9.1153844  # degrees
grid_size = 1000.0  # grid size in meters

# Define the grid
num_points = 2  # Number of points along one axis (10x10 grid)
latitudes = np.linspace(grid_center_lat - 0.005, grid_center_lat + 0.005, num_points)
longitudes = np.linspace(grid_center_lon - 0.005, grid_center_lon + 0.005, num_points)


# Convert latitude, longitude, altitude to ECEF coordinates
def lla_to_ecef(lat, lon, alt):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    N = R_EARTH / np.sqrt(1 - ECCENTRICITY ** 2 * np.sin(lat_rad) ** 2)

    X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - ECCENTRICITY ** 2) + alt) * np.sin(lat_rad)

    return X, Y, Z


# Satellite ECEF coordinates
X_s, Y_s, Z_s = lla_to_ecef(satellite_lat, satellite_lon, satellite_alt)
print("Satellite")
print(X_s, Y_s, Z_s)

# Compute the elevation angle for each grid point
elevation_angles = np.zeros((num_points, num_points))

for i, lat in enumerate(latitudes):
    for j, lon in enumerate(longitudes):
        # Grid point ECEF coordinates
        X_g, Y_g, Z_g = lla_to_ecef(lat, lon, 0)
        print("Grid")
        print(X_g, Y_g, Z_g)

        # Line-of-sight vector from grid point to satellite
        v_x = X_s - X_g
        v_y = Y_s - Y_g
        v_z = Z_s - Z_g

        # Compute the magnitudes of the vectors
        norm_v = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
        norm_g = np.sqrt(X_g ** 2 + Y_g ** 2 + Z_g ** 2)

        # Compute the dot product of the line-of-sight vector with the grid point's position vector
        dot_product = v_x * X_g + v_y * Y_g + v_z * Z_g

        # Cosine of the elevation angle
        cos_elevation_angle = dot_product / (norm_v * norm_g)

        # Elevation angle
        elevation_angle = np.arcsin(cos_elevation_angle)

        # Convert to degrees and store
        elevation_angles[i, j] = np.degrees(elevation_angle)

# Display results
print("Elevation Angles (in degrees):")
print(elevation_angles)
