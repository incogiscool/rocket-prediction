import json
import numpy as np
import pandas

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

def generate_srad_data(folder_path):
    """
    Generate SRAD data from CSV files in 1-second intervals with 10 samples per interval.
    Each sample is averaged from all data points within its 0.1s sub-interval.
    Saves the result to srad_data.json.
    
    Args:
        folder_path: Path to folder containing SRAD CSV files
    """
    # Util function to filter out gnss altitude
    def has_min_2_decimals(value):
        str_val = str(value)
        if '.' in str_val:
            decimal_part = str_val.split('.')[1]
            return len(decimal_part) >= 2
        return False

    alt = pandas.read_csv(folder_path + "/altitude_sea_level.csv")
    vel = pandas.read_csv(folder_path + "/angular_velocity.csv")
    raw_acc = pandas.read_csv(folder_path + "/linear_acceleration.csv")
    acc = raw_acc[raw_acc["magnitude"].apply(has_min_2_decimals)]
    pos = pandas.read_csv(folder_path + "/gnss.csv")

    # Get time range
    start_time = pos['mission_time'].min()
    end_time = pos['mission_time'].max()
    
    # Generate packets with 1-second intervals
    packets = []
    current_time = start_time
    
    while current_time < end_time:
        interval_end = current_time + 1.0
        
        # Get data for this 1-second interval
        pos_in_interval = pos[(pos['mission_time'] >= current_time) & 
                              (pos['mission_time'] < interval_end)]
        alt_in_interval = alt[(alt['mission_time'] >= current_time) & 
                              (alt['mission_time'] < interval_end)]
        acc_in_interval = acc[(acc['mission_time'] >= current_time) & 
                              (acc['mission_time'] < interval_end)]
        vel_in_interval = vel[(vel['mission_time'] >= current_time) & 
                              (vel['mission_time'] < interval_end)]
        
        if len(pos_in_interval) == 0:
            current_time += 1.0
            continue
        
        # Create 10 samples for this interval by averaging data in each 0.1s sub-interval
        samples = []
        for i in range(10):
            sub_interval_start = current_time + (i * 0.1)
            sub_interval_end = sub_interval_start + 0.1
            
            # Get data within this 0.1s sub-interval
            pos_sub = pos_in_interval[(pos_in_interval['mission_time'] >= sub_interval_start) & 
                                       (pos_in_interval['mission_time'] < sub_interval_end)]
            alt_sub = alt_in_interval[(alt_in_interval['mission_time'] >= sub_interval_start) & 
                                       (alt_in_interval['mission_time'] < sub_interval_end)]
            acc_sub = acc_in_interval[(acc_in_interval['mission_time'] >= sub_interval_start) & 
                                       (acc_in_interval['mission_time'] < sub_interval_end)]
            vel_sub = vel_in_interval[(vel_in_interval['mission_time'] >= sub_interval_start) & 
                                       (vel_in_interval['mission_time'] < sub_interval_end)]
            
            # Average all values in this sub-interval
            sample = {
                'time': float(sub_interval_start),
                'lat': float(pos_sub['latitude'].mean()) if len(pos_sub) > 0 else 0.0,
                'lon': float(pos_sub['longitude'].mean()) if len(pos_sub) > 0 else 0.0,
                'alt': float(alt_sub['metres'].mean()) if len(alt_sub) > 0 else 0.0,
                'acc_x': float(acc_sub['x'].mean()) if len(acc_sub) > 0 else 0.0,
                'acc_y': float(acc_sub['y'].mean()) if len(acc_sub) > 0 else 0.0,
                'acc_z': float(acc_sub['z'].mean()) if len(acc_sub) > 0 else 0.0,
                'acc_mag': float(acc_sub['magnitude'].mean()) if len(acc_sub) > 0 else 0.0,
                'vel_x': float(vel_sub['x'].mean()) if len(vel_sub) > 0 else 0.0,
                'vel_y': float(vel_sub['y'].mean()) if len(vel_sub) > 0 else 0.0,
                'vel_z': float(vel_sub['z'].mean()) if len(vel_sub) > 0 else 0.0,
                'vel_mag': float(vel_sub['magnitude'].mean()) if len(vel_sub) > 0 else 0.0
            }
            samples.append(sample)
        
        packet = {
            'timestamp': float(current_time),
            'samples': samples
        }
        packets.append(packet)
        
        current_time += 1.0
    
    # Create the final data structure
    srad_data = {
        'packets': packets,
        'metadata': {
            'source': 'generated_from_csv',
            'start_time': float(start_time),
            'end_time': float(end_time),
            'total_packets': len(packets),
            'samples_per_packet': 10
        }
    }
    
    # Save to srad_data.json
    with open('srad_data.json', 'w') as f:
        json.dump(srad_data, f, indent=2)
    
    print(f"Generated {len(packets)} packets with 10 samples each")
    print(f"Saved to srad_data.json")
    
    return srad_data
