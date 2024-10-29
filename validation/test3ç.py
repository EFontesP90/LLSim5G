import numpy as np

# Constants
R_EARTH = 6378137.0  # Earth's radius in meters (WGS 84)
ECCENTRICITY = 0.08181919  # Earth's eccentricity (WGS 84)

# Desired elevation angle and satellite altitude
desired_elevation_angle = 80.0  # degrees
satellite_altitude = 50000.0  # altitude in meters

# Grid center in LLA (latitude, longitude)
# Via dei Pisani 4, 09124 Cagliari CA, Italia
grid_center_lat = 39.2137738  # degrees
grid_center_lon = 9.1153844  # degrees


# Convert latitude, longitude, altitude to ECEF coordinates
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


# Calculate the satellite position given a specific elevation angle and altitude
def satellite_position_for_elevation(lat_g, lon_g, alt_g, elevation_angle, sat_altitude):
    # Convert grid center to ECEF
    X_g, Y_g, Z_g = lla_to_ecef(lat_g, lon_g, alt_g)

    # Elevation angle in radians
    elevation_rad = np.radians(elevation_angle)

    # Distance from the grid center to the satellite
    r_gs = (sat_altitude + R_EARTH - alt_g) / np.tan(elevation_rad)

    # Satellite position relative to the grid center
    X_s = X_g + r_gs * np.cos(np.radians(lat_g)) * np.cos(np.radians(lon_g))
    Y_s = Y_g + r_gs * np.cos(np.radians(lat_g)) * np.sin(np.radians(lon_g))
    Z_s = Z_g + r_gs * np.sin(np.radians(lat_g))
    print(X_s, Y_s, Z_s)

    # Ensure the satellite's altitude is correct
    # X_s, Y_s, Z_s = lla_to_ecef(grid_center_lat, grid_center_lon, sat_altitude)
    print(X_s, Y_s, Z_s)

    return X_s, Y_s, Z_s


# Calculate the satellite's ECEF position
X_s, Y_s, Z_s = satellite_position_for_elevation(
    grid_center_lat, grid_center_lon, 0, desired_elevation_angle, satellite_altitude)

# Convert the satellite's ECEF coordinates back to LLA
satellite_lat, satellite_lon, satellite_alt = ecef_to_lla(X_s, Y_s, Z_s)

# Display the satellite LLA coordinates
print(f"Satellite Position for {desired_elevation_angle}° Elevation Angle and {satellite_altitude} m Altitude:")
print(f"Latitude: {satellite_lat:.6f}°")
print(f"Longitude: {satellite_lon:.6f}°")
print(f"Altitude: {satellite_alt:.2f} meters")
