import json
import numpy as np

def parse_json(json_file_path: str) -> dict:
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return {}


def calculate_and_print_errors(t, x, y, z, t_gps, x_gps, y_gps, z_gps):
    """
    Calculate and print error statistics between interpolated predictions and GPS data.
    
    Args:
        t: Array of all timestamps (GPS + interpolated)
        x: Array of all x positions (GPS + interpolated)
        y: Array of all y positions (GPS + interpolated)
        z: Array of all z positions (GPS + interpolated)
        t_gps: Array of GPS timestamps
        x_gps: Array of GPS x positions
        y_gps: Array of GPS y positions
        z_gps: Array of GPS z positions
    """
    print("\n======================== ERROR STATISTICS ========================")
    
    # Calculate errors at GPS points
    errors = []
    
    for i in range(len(t_gps)):
        gps_time = t_gps[i]
        
        # Find the closest interpolated point to this GPS time
        closest_idx = np.argmin(np.abs(np.array(t) - gps_time))
        
        # Calculate Euclidean distance error
        error = np.sqrt(
            (x[closest_idx] - x_gps[i])**2 + 
            (y[closest_idx] - y_gps[i])**2 + 
            (z[closest_idx] - z_gps[i])**2
        )
        errors.append(error)
    
    # Calculate statistics
    average_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    print(f"\n3D Position Error:")
    print(f"  Average: {average_error:.2f} meters")
    print(f"  Maximum: {max_error:.2f} meters")
    print(f"  Minimum: {min_error:.2f} meters")
    
    # Calculate errors per axis
    error_x = np.mean([abs(x[np.argmin(np.abs(np.array(t) - t_gps[i]))] - x_gps[i]) for i in range(len(t_gps))])
    error_y = np.mean([abs(y[np.argmin(np.abs(np.array(t) - t_gps[i]))] - y_gps[i]) for i in range(len(t_gps))])
    error_z = np.mean([abs(z[np.argmin(np.abs(np.array(t) - t_gps[i]))] - z_gps[i]) for i in range(len(t_gps))])
    
    print(f"\nPer-Axis Average Absolute Error:")
    print(f"  X (Easting):  {error_x:.2f} meters")
    print(f"  Y (Northing): {error_y:.2f} meters")
    print(f"  Z (Altitude): {error_z:.2f} meters")
    
    print(f"\nTotal GPS samples: {len(x_gps)}")
    print(f"Total predictions (including GPS + interpolated): {len(x)}")
    print("==================================================================\n")