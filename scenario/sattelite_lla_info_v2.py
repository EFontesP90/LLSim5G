import numpy as np

# Constants
R_EARTH = 6378137  # Earth's radius in meters
ECCENTRICITY = 0.08181919  # Earth's eccentricity (WGS 84)

# Conversion from LLA (lat, lon, alt) to ECEF (X, Y, Z)
# def lla_to_ecef(lat, lon, alt):
#     lat, lon = np.radians(lat), np.radians(lon)
#     X = (R_EARTH + alt) * np.cos(lat) * np.cos(lon)
#     Y = (R_EARTH + alt) * np.cos(lat) * np.sin(lon)
#     Z = (R_EARTH + alt) * np.sin(lat)
#     return X, Y, Z
#
# # Conversion from ECEF (X, Y, Z) to LLA (lat, lon, alt)
# def ecef_to_lla(X, Y, Z):
#     lon = np.degrees(np.arctan2(Y, X))
#     p = np.sqrt(X**2 + Y**2)
#     lat = np.degrees(np.arctan2(Z, p))
#     alt = np.sqrt(X**2 + Y**2 + Z**2) - R_EARTH
#     return lat, lon, alt

def lla_to_ecef(lat, lon, alt):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    N = R_EARTH / np.sqrt(1 - ECCENTRICITY ** 2 * np.sin(lat_rad) ** 2)

    X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - ECCENTRICITY ** 2) + alt) * np.sin(lat_rad)

    return X, Y, Z

# Convert ECEF coordinates to LLA (latitude, longitude, altitude)
def ecef_to_lla(X, Y, Z):
    lon = np.arctan2(Y, X)
    p = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Z * R_EARTH, p * (R_EARTH * (1 - ECCENTRICITY ** 2)))
    lat = np.arctan2(Z + ECCENTRICITY ** 2 * R_EARTH * np.sin(theta) ** 3,
                     p - ECCENTRICITY ** 2 * R_EARTH * np.cos(theta) ** 3)
    N = R_EARTH / np.sqrt(1 - ECCENTRICITY ** 2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt

# Function to compute satellite position for a given elevation angle
def satellite_position_for_elevation(lat_g, lon_g, alt_g, elevation_angle, sat_altitude):
    # Convert grid center to ECEF
    X_g, Y_g, Z_g = lla_to_ecef(lat_g, lon_g, alt_g)
    lat_gg, lon_gg, alt_gg = ecef_to_lla(X_g, Y_g, Z_g)
    print("lat_g, lon_g, alt_g", lat_g, lon_g, alt_g)
    print("lat_gg, lon_gg, alt_gg", lat_gg, lon_gg, alt_gg)

    # Elevation angle in radians
    elevation_rad = np.radians(elevation_angle)

    # Distance from the grid center to the satellite
    r_gs = (sat_altitude + R_EARTH - alt_g) / np.tan(elevation_rad)

    # Satellite position relative to the grid center
    X_s = X_g + r_gs * np.cos(np.radians(lat_g)) * np.cos(np.radians(lon_g))
    Y_s = Y_g + r_gs * np.cos(np.radians(lat_g)) * np.sin(np.radians(lon_g))
    Z_s = Z_g + r_gs * np.sin(np.radians(lat_g))

    # Convert satellite ECEF back to LLA
    lat_s, lon_s, alt_s = ecef_to_lla(X_s, Y_s, Z_s)
    return lat_s, lon_s, alt_s

# Example Grid LLA
grid_lat = 39.2337738
grid_lon = 9.1153844
grid_alt = 100  # altitude in meters

# Satellite parameters
sat_altitude = 50000  # in meters
desired_elevation_angle = 70  # in degrees

# Compute satellite LLA
satellite_lla = satellite_position_for_elevation(grid_lat, grid_lon, grid_alt, desired_elevation_angle, sat_altitude)

print("satellite_lla", satellite_lla)


def calculate_elevation_angle(lat_g, lon_g, alt_g, lat_s, lon_s, alt_s):
    # Convert grid and satellite positions to ECEF
    X_g, Y_g, Z_g = lla_to_ecef(lat_g, lon_g, alt_g)
    X_s, Y_s, Z_s = lla_to_ecef(lat_s, lon_s, alt_s)

    # Vector from grid center to satellite
    vec_gs = np.array([X_s - X_g, Y_s - Y_g, Z_s - Z_g])

    # Distance between grid center and satellite
    distance_gs = np.linalg.norm(vec_gs)

    # Vertical distance (Z-axis difference)
    vertical_distance = Z_s - Z_g

    # Elevation angle in radians
    elevation_rad = np.arcsin(vertical_distance / distance_gs)

    # Convert to degrees
    elevation_deg = np.degrees(elevation_rad)
    return elevation_deg


# Check the computed elevation angle
computed_elevation_angle = calculate_elevation_angle(grid_lat, grid_lon, grid_alt, satellite_lla[0], satellite_lla[1],
                                                     satellite_lla[2])

print("computed_elevation_angle", computed_elevation_angle)

##################################################################################################################
##################################################################################################################

# import numpy as np
#
# # Constants
# R_EARTH = 6378137  # Earth's radius in meters
# sat_altitude = 50000  # Satellite altitude in meters
# desired_elevation_angle = 50  # Desired elevation angle in degrees
#
# # Grid center coordinates (LLA)
# lat_g = 39.2337738  # Grid center latitude in degrees
# lon_g = 9.1153844  # Grid center longitude in degrees
# alt_g = 0  # Grid center altitude in meters (at sea level)
#
#
# # Convert LLA to ECEF (Earth-Centered, Earth-Fixed)
# def lla_to_ecef(lat, lon, alt):
#     lat_rad = np.radians(lat)
#     lon_rad = np.radians(lon)
#
#     N = R_EARTH / np.sqrt(1 - (2 / 298.257223563) * np.sin(lat_rad) ** 2)
#
#     X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
#     Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
#     Z = ((1 - 1 / 298.257223563) ** 2 * N + alt) * np.sin(lat_rad)
#
#     return X, Y, Z
#
#
# # Convert ECEF to LLA
# def ecef_to_lla(X, Y, Z):
#     # Iterative method to convert ECEF to LLA
#     b = R_EARTH * (1 - 1 / 298.257223563)  # Semi-minor axis
#     e = np.sqrt(R_EARTH ** 2 - b ** 2) / R_EARTH  # Eccentricity
#
#     lon = np.arctan2(Y, X)
#     p = np.sqrt(X ** 2 + Y ** 2)
#     lat = np.arctan2(Z, p * (1 - e ** 2))
#     alt = p / np.cos(lat) - R_EARTH
#
#     lat = np.degrees(lat)
#     lon = np.degrees(lon)
#
#     return lat, lon, alt
#
#
# # Calculate satellite position based on desired elevation angle
# def satellite_position_for_elevation(lat_g, lon_g, alt_g, elevation_angle, sat_altitude):
#     # Convert grid center to ECEF
#     X_g, Y_g, Z_g = lla_to_ecef(lat_g, lon_g, alt_g)
#
#     # Elevation angle in radians
#     elevation_rad = np.radians(elevation_angle)
#
#     # Distance from the grid center to the satellite
#     r_gs = (sat_altitude + R_EARTH - alt_g) / np.tan(elevation_rad)
#
#     # Satellite position relative to the grid center
#     X_s = X_g + r_gs * np.cos(np.radians(lat_g)) * np.cos(np.radians(lon_g))
#     Y_s = Y_g + r_gs * np.cos(np.radians(lat_g)) * np.sin(np.radians(lon_g))
#     Z_s = Z_g + r_gs * np.sin(np.radians(lat_g))
#
#     # Return satellite position in ECEF
#     return X_s, Y_s, Z_s
#
#
# # Compute satellite position
# X_s, Y_s, Z_s = satellite_position_for_elevation(lat_g, lon_g, alt_g, desired_elevation_angle, sat_altitude)
#
# # Convert satellite ECEF to LLA
# sat_lat, sat_lon, sat_alt = ecef_to_lla(X_s, Y_s, Z_s)
#
#
# # Now compute the resulting elevation angle for verification
# def calculate_elevation_angle(X_g, Y_g, Z_g, X_s, Y_s, Z_s):
#     # Distance between grid center and satellite
#     r_gs = np.sqrt((X_s - X_g) ** 2 + (Y_s - Y_g) ** 2 + (Z_s - Z_g) ** 2)
#
#     # Line of sight vector (from grid to satellite)
#     los_vector = np.array([X_s - X_g, Y_s - Y_g, Z_s - Z_g])
#
#     # Ground normal vector (from Earth's center to grid center)
#     ground_vector = np.array([X_g, Y_g, Z_g])
#
#     # Calculate the elevation angle using the dot product
#     dot_product = np.dot(los_vector, ground_vector)
#     norm_los = np.linalg.norm(los_vector)
#     norm_ground = np.linalg.norm(ground_vector)
#
#     cos_elevation_angle = dot_product / (norm_los * norm_ground)
#     elevation_angle_rad = np.arccos(cos_elevation_angle)
#
#     # Convert to degrees
#     elevation_angle_deg = np.degrees(elevation_angle_rad)
#
#     return elevation_angle_deg
#
#
# # Recalculate elevation angle for verification
# X_g, Y_g, Z_g = lla_to_ecef(lat_g, lon_g, alt_g)
# resulting_elevation_angle = calculate_elevation_angle(X_g, Y_g, Z_g, X_s, Y_s, Z_s)
#
# print("sat_lat, sat_lon, sat_alt, resulting_elevation_angle", sat_lat, sat_lon, sat_alt, resulting_elevation_angle)
# # sat_lat, sat_lon, sat_alt, resulting_elevation_angle
