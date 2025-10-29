import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import utm
import util
import formulas
import filters


def load_raw_srad_data():
    """
    Load raw SRAD data from CSV files on the SD card.
    
    Returns:
        Dictionary with raw data: time, x, y, z arrays
    """
    print("Loading raw SRAD CSV data...")
    
    # Load GNSS data
    gnss_df = pd.read_csv('./srad/gnss.csv')
    
    # Load altitude data
    alt_df = pd.read_csv('./srad/altitude_sea_level.csv')
    
    # Filter out invalid altitude readings (365.0 appears to be error value)
    alt_df = alt_df[alt_df['metres'] != 365.0]
    
    # Convert lat/lon to UTM
    raw_data = {
        'time': [],
        'x': [],
        'y': [],
        'z': []
    }
    
    # Merge GNSS with altitude based on mission_time
    merged_df = pd.merge_asof(
        gnss_df.sort_values('mission_time'),
        alt_df.sort_values('mission_time'),
        on='mission_time',
        direction='nearest',
        tolerance=0.1  # 100ms tolerance
    )
    
    # Convert to UTM coordinates
    for _, row in merged_df.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']) and pd.notna(row['metres']):
            easting, northing, _, _ = utm.from_latlon(row['latitude'], row['longitude'])
            # Normalize time to start at 0
            time_normalized = row['mission_time'] - merged_df['mission_time'].min()
            
            raw_data['time'].append(time_normalized)
            raw_data['x'].append(easting)
            raw_data['y'].append(northing)
            raw_data['z'].append(row['metres'])
    
    print(f"Loaded {len(raw_data['time'])} raw data points")
    return raw_data


def simulate_predictions(srad_data: dict, prediction_interval: float = 0.1):
    """
    Simulate the prediction algorithm and collect data for plotting.
    
    Args:
        srad_data: Parsed SRAD data dictionary
        prediction_interval: Time interval for prediction updates (default: 0.1s)
        
    Returns:
        Tuple of (predicted_data, real_data)
    """
    packets = srad_data.get('packets', [])
    
    # Storage for plotting
    predicted_data = {
        'time': [],
        'x': [],
        'y': [],
        'z': []
    }
    
    real_data = {
        'time': [],
        'x': [],
        'y': [],
        'z': []
    }
    
    # Initial position state
    pos = {
        'x': 0,
        'y': 0,
        'z': 0,
        'velx': 0,
        'vely': 0,
        'velz': 0,
        'accx': 0,
        'accy': 0,
        'accz': 0
    }
    
    # Process each packet
    for packet_idx, packet in enumerate(packets):
        if not isinstance(packet, dict) or 'samples' not in packet:
            continue
            
        samples = packet.get('samples', [])
        if not samples:
            continue
        
        latest_sample = samples[-1]
        packet_time = packet.get('timestamp', 0)
        
        # Convert lat/lon to UTM for real data
        easting, northing, _, _ = utm.from_latlon(
            latest_sample.get('lat', 0), 
            latest_sample.get('lon', 0)
        )
        
        # Store real data (offset by 2 seconds for LoRa delay)
        real_data['time'].append(packet_time + 2)
        real_data['x'].append(easting)
        real_data['y'].append(northing)
        real_data['z'].append(latest_sample.get('alt', 0))
        
        # Update position state from SRAD packet
        pos['x'] = easting
        pos['y'] = northing
        pos['z'] = latest_sample.get('alt', 0)
        
        # Calculate velocity from last 2 samples
        if len(samples) >= 2:
            prev_sample = samples[-2]
            time_delta = latest_sample.get('time', 0) - prev_sample.get('time', 0)
            
            prev_easting, prev_northing, _, _ = utm.from_latlon(
                prev_sample.get('lat', 0), 
                prev_sample.get('lon', 0)
            )
            
            if time_delta > 0:
                pos['velx'] = (easting - prev_easting) / time_delta
                pos['vely'] = (northing - prev_northing) / time_delta
                pos['velz'] = (latest_sample.get('alt', 0) - prev_sample.get('alt', 0)) / time_delta
        
        # Update acceleration
        pos['accx'] = latest_sample.get('acc_x', 0) - 9.81
        pos['accy'] = latest_sample.get('acc_y', 0)
        pos['accz'] = latest_sample.get('acc_z', 0)
        
        # Predict positions for the next second (until next packet)
        num_steps = int(1.0 / prediction_interval)
        for step in range(num_steps):
            time_elapsed = step * prediction_interval + 2  # Add 2 second LoRa delay
            current_time = packet_time + step * prediction_interval
            
            # Use constant acceleration equation for prediction (matching main.py logic)
            # Note: x and z are flipped to match the main.py implementation
            # pred_x = formulas.const_acc_eq(pos['velz'], time_elapsed, pos['accz'], pos['x'])
            # pred_y = formulas.const_acc_eq(pos['vely'], time_elapsed, pos['accy'], pos['y'])
            # pred_z = formulas.const_acc_eq(pos['velx'], time_elapsed, pos['accx'], pos['z'])

            ALPHA = 0.35
            
            # Update position using constant acceleration equation for each dimension. Flipped x and z because accelerometer seems to be mounted differently
            pred_x = filters.lowpass(formulas.const_acc_eq(pos["velz"], time_elapsed + 2, pos["accz"], pos['x']), pos['x'], ALPHA)
            pred_y = filters.lowpass(formulas.const_acc_eq(pos["vely"], time_elapsed + 2, pos["accy"], pos['y']), pos['y'], ALPHA)
            pred_z = filters.lowpass(formulas.const_acc_eq(pos["velx"], time_elapsed + 2, pos["accx"], pos['z']), pos['z'], ALPHA)
            
            predicted_data['time'].append(current_time)
            predicted_data['x'].append(pred_x)
            predicted_data['y'].append(pred_y)
            predicted_data['z'].append(pred_z)
    
    return predicted_data, real_data


def plot_comparison(predicted_data: dict, real_data: dict, raw_data: dict = None):
    """
    Plot predicted vs real data comparison.
    
    Args:
        predicted_data: Dictionary with predicted position data
        real_data: Dictionary with real SRAD position data (offset by 2 seconds)
        raw_data: Dictionary with raw high-frequency SRAD data (optional)
    """
    # Create figure with 2D plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Prediction Algorithm vs Raw SRAD Data', fontsize=16)
    
    # X position (Easting)
    if raw_data:
        axes[0, 0].plot(raw_data['time'], raw_data['x'], 
                        label='Raw SD Card Data', alpha=0.8, linewidth=2, color='darkgreen')
    axes[0, 0].scatter(predicted_data['time'], predicted_data['x'], 
                    label='Predicted (1Hz)', alpha=0.7, s=2, color='blue')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Position (Easting, m)')
    axes[0, 0].set_title('X Position Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Y position (Northing)
    if raw_data:
        axes[0, 1].plot(raw_data['time'], raw_data['y'], 
                        label='Raw SD Card Data', alpha=0.8, linewidth=2, color='darkgreen')
    axes[0, 1].scatter(predicted_data['time'], predicted_data['y'], 
                    label='Predicted (1Hz)', alpha=0.7, s=2, color='blue')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Y Position (Northing, m)')
    axes[0, 1].set_title('Y Position Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Z position (Altitude)
    if raw_data:
        axes[1, 0].plot(raw_data['time'], raw_data['z'], 
                        label='Raw SD Card Data', alpha=0.8, linewidth=2, color='darkgreen')
    axes[1, 0].scatter(predicted_data['time'], predicted_data['z'], 
                    label='Predicted (1Hz)', alpha=0.7, s=2, color='blue')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Z Position (Altitude, m)')
    axes[1, 0].set_title('Z Position (Altitude) Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 2D trajectory (X-Y plane)
    if raw_data:
        axes[1, 1].plot(raw_data['x'], raw_data['y'], 
                        label='Raw SD Card Data', alpha=0.8, linewidth=2, color='darkgreen')
    axes[1, 1].scatter(predicted_data['x'], predicted_data['y'], 
                    label='Predicted Path (1Hz)', alpha=0.7, s=2, color='blue')
    axes[1, 1].set_xlabel('X Position (Easting, m)')
    axes[1, 1].set_ylabel('Y Position (Northing, m)')
    axes[1, 1].set_title('2D Trajectory (Top View)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('prediction_vs_real_2d.png', dpi=300, bbox_inches='tight')
    print("2D plot saved as 'prediction_vs_real_2d.png'")
    
    # Create 3D plot
    fig_3d = plt.figure(figsize=(14, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Plot raw data if available
    if raw_data:
        ax_3d.plot(raw_data['x'], raw_data['y'], raw_data['z'],
                   label='Raw SD Card Data', alpha=0.8, linewidth=2.5, color='darkgreen')
    
    # Plot predicted trajectory as dots
    ax_3d.scatter(predicted_data['x'], predicted_data['y'], predicted_data['z'],
               label='Predicted Path (1Hz)', alpha=0.7, s=30, color='blue')
    
    # Mark start and end points
    if raw_data:
        ax_3d.scatter([raw_data['x'][0]], [raw_data['y'][0]], [raw_data['z'][0]],
                      color='green', s=150, marker='o', label='Start', edgecolors='black', linewidths=2)
        ax_3d.scatter([raw_data['x'][-1]], [raw_data['y'][-1]], [raw_data['z'][-1]],
                      color='purple', s=150, marker='s', label='End', edgecolors='black', linewidths=2)
    
    ax_3d.set_xlabel('X Position (Easting, m)', fontsize=11)
    ax_3d.set_ylabel('Y Position (Northing, m)', fontsize=11)
    ax_3d.set_zlabel('Z Position (Altitude, m)', fontsize=11)
    
    title = '3D Trajectory Comparison\nPrediction Algorithm vs Raw SRAD Data'
    if raw_data:
        title += '\n(Dark Green: Raw SD Card Data, Blue Dashed: 1Hz Prediction with 2s Delay)'
    ax_3d.set_title(title, fontsize=13, pad=20)
    ax_3d.legend(loc='upper left', fontsize=10)
    ax_3d.grid(True, alpha=0.3)
    
    # Set viewing angle for better perspective
    ax_3d.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('prediction_vs_real_3d.png', dpi=300, bbox_inches='tight')
    print("3D plot saved as 'prediction_vs_real_3d.png'")
    
    plt.show()


def calculate_errors(predicted_data: dict, real_data: dict):
    """
    Calculate error metrics between predicted and real data.
    
    Args:
        predicted_data: Dictionary with predicted position data
        real_data: Dictionary with real SRAD position data
    """
    # Interpolate predicted data at real data timestamps
    from scipy import interpolate
    
    # Create interpolation functions for predicted data
    pred_x_interp = interpolate.interp1d(predicted_data['time'], predicted_data['x'], 
                                          kind='linear', fill_value='extrapolate')
    pred_y_interp = interpolate.interp1d(predicted_data['time'], predicted_data['y'], 
                                          kind='linear', fill_value='extrapolate')
    pred_z_interp = interpolate.interp1d(predicted_data['time'], predicted_data['z'], 
                                          kind='linear', fill_value='extrapolate')
    
    # Get predicted values at real timestamps
    pred_x_at_real = pred_x_interp(real_data['time'])
    pred_y_at_real = pred_y_interp(real_data['time'])
    pred_z_at_real = pred_z_interp(real_data['time'])
    
    # Calculate errors
    error_x = np.array(pred_x_at_real) - np.array(real_data['x'])
    error_y = np.array(pred_y_at_real) - np.array(real_data['y'])
    error_z = np.array(pred_z_at_real) - np.array(real_data['z'])
    
    # Calculate 3D Euclidean distance error
    error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)
    
    print("\n" + "="*60)
    print("ERROR METRICS")
    print("="*60)
    print(f"X Position Error:")
    print(f"  Mean: {np.mean(error_x):.3f} m")
    print(f"  RMS:  {np.sqrt(np.mean(error_x**2)):.3f} m")
    print(f"  Max:  {np.max(np.abs(error_x)):.3f} m")
    
    print(f"\nY Position Error:")
    print(f"  Mean: {np.mean(error_y):.3f} m")
    print(f"  RMS:  {np.sqrt(np.mean(error_y**2)):.3f} m")
    print(f"  Max:  {np.max(np.abs(error_y)):.3f} m")
    
    print(f"\nZ Position Error:")
    print(f"  Mean: {np.mean(error_z):.3f} m")
    print(f"  RMS:  {np.sqrt(np.mean(error_z**2)):.3f} m")
    print(f"  Max:  {np.max(np.abs(error_z)):.3f} m")
    
    print(f"\n3D Distance Error:")
    print(f"  Mean: {np.mean(error_3d):.3f} m")
    print(f"  RMS:  {np.sqrt(np.mean(error_3d**2)):.3f} m")
    print(f"  Max:  {np.max(error_3d):.3f} m")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Load SRAD data
    print("Loading SRAD data...")
    srad_data = util.parse_json("./srad_data.json")
    
    # Load raw SD card data
    raw_data = None
    try:
        raw_data = load_raw_srad_data()
    except Exception as e:
        print(f"Warning: Could not load raw SD card data: {e}")
        print("Continuing without raw data overlay...")
    
    # Run simulation and collect data
    print("Running prediction simulation...")
    predicted_data, real_data = simulate_predictions(srad_data, prediction_interval=0.1)
    
    print(f"Collected {len(predicted_data['time'])} predicted points")
    print(f"Collected {len(real_data['time'])} real data points")
    if raw_data:
        print(f"Collected {len(raw_data['time'])} raw SD card data points")
    
    # Calculate and display errors
    try:
        calculate_errors(predicted_data, real_data)
    except ImportError:
        print("Install scipy for error metrics: pip install scipy")
    except Exception as e:
        print(f"Could not calculate errors: {e}")
    
    # Generate plots
    print("Generating plots...")
    plot_comparison(predicted_data, real_data, raw_data)
