import pandas as pd
import numpy as np

# Function to integrate the gyroscope readings (angular velocity) over time to estimate the heading change (turning angle)
# def estimate_turning_angle(gyro_data: pd.DataFrame, frequency):
#     """
#     Estimates the turning angle using gyroscope data.
    
#     Parameters:
#     gyro_data (pd.DataFrame): DataFrame with columns 'x', 'y', 'z' representing gyroscope readings in radians per second
#     frequency (float): Sampling frequency (Hz)
    
#     Returns:
#     float: Total turning angle (in radians) from the start
#     """
#     total_angle = 0.0  # Initialize total turning angle
#     dt = 1.0 / frequency

#     # Iterate over rows in the DataFrame
#     for _, row in gyro_data.iterrows():
#         gx, gy, gz = row['x'], row['y'], row['z']
#         total_angle += gz * dt  # Integrate Z-axis angular velocity (yaw)

#     return total_angle


def estimate_turning_angle(gyro_data: np.ndarray, frequency):
    """
    Estimates the turning angle using gyroscope data.
    """
    total_angle = 0.0  # Initialize total turning angle
    dt = 1.0 / frequency

    total_angle += sum(gyro_data[:,2]) * dt  # Integrate Z-axis angular velocity (yaw)

    return total_angle

