import matplotlib.pyplot as plt
import numpy as np

def plot_position_data(t, x, y, z, t_gps=None, x_gps=None, y_gps=None, z_gps=None):
    """
    Plot position data over time in 3 subplots.
    
    Args:
        t, x, y, z: Time and position arrays (combined/interpolated data)
        t_gps, x_gps, y_gps, z_gps: Optional GPS data to overlay
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Position vs Time', fontsize=16, fontweight='bold')
    
    # Configuration for each axis
    configs = [
        {'data': (x, x_gps), 'colors': ('lightblue', 'darkblue'), 
         'ylabel': 'X Position (m)', 'title': 'X Position (Easting) vs Time'},
        {'data': (y, y_gps), 'colors': ('lightgreen', 'darkgreen'), 
         'ylabel': 'Y Position (m)', 'title': 'Y Position (Northing) vs Time'},
        {'data': (z, z_gps), 'colors': ('lightcoral', 'darkred'), 
         'ylabel': 'Z Position (m)', 'title': 'Z Position (Altitude) vs Time'}
    ]
    
    for ax, config in zip(axes, configs):
        # Plot interpolated data
        ax.scatter(t, config['data'][0], c=config['colors'][0], 
                  s=2, alpha=0.4, label='Interpolated')
        
        # Plot GPS data if provided
        if config['data'][1] is not None and t_gps is not None:
            ax.scatter(t_gps, config['data'][1], c=config['colors'][1], 
                      s=2, alpha=0.8, marker='o', label='GPS Data')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(config['ylabel'], fontsize=12)
        ax.set_title(config['title'], fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_errors(t, x, y, z, t_gps, x_gps, y_gps, z_gps):
    """
    Plot error analysis between interpolated predictions and GPS data.
    
    Creates multiple visualizations:
    1. 3D Euclidean error over time
    2. Per-axis errors (X, Y, Z) over time
    3. Error distribution histogram
    4. Cumulative error plot
    
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
    # Calculate errors at each GPS point
    errors_3d = []
    errors_x = []
    errors_y = []
    errors_z = []
    error_times = []
    
    t_array = np.array(t)
    x_array = np.array(x)
    y_array = np.array(y)
    z_array = np.array(z)
    
    for i in range(len(t_gps)):
        gps_time = t_gps[i]
        
        # Find the closest interpolated point to this GPS time
        closest_idx = np.argmin(np.abs(t_array - gps_time))
        
        # Calculate per-axis errors
        err_x = x_array[closest_idx] - x_gps[i]
        err_y = y_array[closest_idx] - y_gps[i]
        err_z = z_array[closest_idx] - z_gps[i]
        
        # Calculate 3D Euclidean error
        error_3d = np.sqrt(err_x**2 + err_y**2 + err_z**2)
        
        errors_3d.append(error_3d)
        errors_x.append(err_x)
        errors_y.append(err_y)
        errors_z.append(err_z)
        error_times.append(gps_time)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D Euclidean Error over Time
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(error_times, errors_3d, 'o-', color='purple', linewidth=1.5, markersize=4, alpha=0.7)
    ax1.axhline(y=np.mean(errors_3d), color='red', linestyle='--', 
                label=f'Mean: {np.mean(errors_3d):.2f}m', linewidth=2)
    ax1.fill_between(error_times, 0, errors_3d, alpha=0.3, color='purple')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('3D Error (m)', fontsize=11)
    ax1.set_title('3D Euclidean Position Error Over Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Per-Axis Errors over Time
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(error_times, errors_x, 'o-', label='X Error (Easting)', 
             color='blue', linewidth=1.5, markersize=3, alpha=0.7)
    ax2.plot(error_times, errors_y, 's-', label='Y Error (Northing)', 
             color='green', linewidth=1.5, markersize=3, alpha=0.7)
    ax2.plot(error_times, errors_z, '^-', label='Z Error (Altitude)', 
             color='red', linewidth=1.5, markersize=3, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Error (m)', fontsize=11)
    ax2.set_title('Per-Axis Position Errors Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Error Statistics Table
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    
    stats_data = [
        ['Metric', 'X (m)', 'Y (m)', 'Z (m)', '3D (m)'],
        ['Mean', f'{np.mean(np.abs(errors_x)):.2f}', f'{np.mean(np.abs(errors_y)):.2f}', 
         f'{np.mean(np.abs(errors_z)):.2f}', f'{np.mean(errors_3d):.2f}'],
        ['Median', f'{np.median(np.abs(errors_x)):.2f}', f'{np.median(np.abs(errors_y)):.2f}', 
         f'{np.median(np.abs(errors_z)):.2f}', f'{np.median(errors_3d):.2f}'],
        ['Std Dev', f'{np.std(errors_x):.2f}', f'{np.std(errors_y):.2f}', 
         f'{np.std(errors_z):.2f}', f'{np.std(errors_3d):.2f}'],
        ['Max', f'{np.max(np.abs(errors_x)):.2f}', f'{np.max(np.abs(errors_y)):.2f}', 
         f'{np.max(np.abs(errors_z)):.2f}', f'{np.max(errors_3d):.2f}'],
        ['Min', f'{np.min(np.abs(errors_x)):.2f}', f'{np.min(np.abs(errors_y)):.2f}', 
         f'{np.min(np.abs(errors_z)):.2f}', f'{np.min(errors_3d):.2f}'],
        ['RMSE', f'{np.sqrt(np.mean(np.array(errors_x)**2)):.2f}', 
         f'{np.sqrt(np.mean(np.array(errors_y)**2)):.2f}', 
         f'{np.sqrt(np.mean(np.array(errors_z)**2)):.2f}', 
         f'{np.sqrt(np.mean(np.array(errors_3d)**2)):.2f}']
    ]
    
    table = ax3.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    ax3.set_title('Error Statistics Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('Prediction Error Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    # Print summary to console
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total GPS samples analyzed: {len(t_gps)}")
    print(f"\n3D Position Error:")
    print(f"  Mean:   {np.mean(errors_3d):.2f} m")
    print(f"  Median: {np.median(errors_3d):.2f} m")
    print(f"  Std:    {np.std(errors_3d):.2f} m")
    print(f"  RMSE:   {np.sqrt(np.mean(np.array(errors_3d)**2)):.2f} m")
    print(f"  Max:    {np.max(errors_3d):.2f} m")
    print(f"  Min:    {np.min(errors_3d):.2f} m")
    print("="*60 + "\n")


def plot_error_over_time(t, x, y, z, t_gps, x_gps, y_gps, z_gps):
    """
    Plot error over time for X, Y, and Z (altitude) in separate subplots.
    
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
    # Calculate errors at each GPS point
    errors_x = []
    errors_y = []
    errors_z = []
    error_times = []
    
    t_array = np.array(t)
    x_array = np.array(x)
    y_array = np.array(y)
    z_array = np.array(z)
    
    for i in range(len(t_gps)):
        gps_time = t_gps[i]
        
        # Find the closest interpolated point to this GPS time
        closest_idx = np.argmin(np.abs(t_array - gps_time))
        
        # Calculate per-axis errors (absolute)
        err_x = abs(x_array[closest_idx] - x_gps[i])
        err_y = abs(y_array[closest_idx] - y_gps[i])
        err_z = abs(z_array[closest_idx] - z_gps[i])
        
        errors_x.append(err_x)
        errors_y.append(err_y)
        errors_z.append(err_z)
        error_times.append(gps_time)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Absolute Error Over Time by Axis', fontsize=16, fontweight='bold')
    
    # X Error (Easting)
    axes[0].plot(error_times, errors_x, 'o-', color='blue', linewidth=2, markersize=4, alpha=0.7)
    axes[0].fill_between(error_times, 0, errors_x, alpha=0.3, color='blue')
    axes[0].axhline(y=np.mean(errors_x), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(errors_x):.2f}m', linewidth=2)
    axes[0].axhline(y=np.median(errors_x), color='orange', linestyle='--', 
                    label=f'Median: {np.median(errors_x):.2f}m', linewidth=2)
    axes[0].set_ylabel('X Error (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('X Position Error (Easting)', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Y Error (Northing)
    axes[1].plot(error_times, errors_y, 'o-', color='green', linewidth=2, markersize=4, alpha=0.7)
    axes[1].fill_between(error_times, 0, errors_y, alpha=0.3, color='green')
    axes[1].axhline(y=np.mean(errors_y), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(errors_y):.2f}m', linewidth=2)
    axes[1].axhline(y=np.median(errors_y), color='orange', linestyle='--', 
                    label=f'Median: {np.median(errors_y):.2f}m', linewidth=2)
    axes[1].set_ylabel('Y Error (m)', fontsize=12, fontweight='bold')
    axes[1].set_title('Y Position Error (Northing)', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Z Error (Altitude)
    axes[2].plot(error_times, errors_z, 'o-', color='red', linewidth=2, markersize=4, alpha=0.7)
    axes[2].fill_between(error_times, 0, errors_z, alpha=0.3, color='red')
    axes[2].axhline(y=np.mean(errors_z), color='darkred', linestyle='--', 
                    label=f'Mean: {np.mean(errors_z):.2f}m', linewidth=2)
    axes[2].axhline(y=np.median(errors_z), color='orange', linestyle='--', 
                    label=f'Median: {np.median(errors_z):.2f}m', linewidth=2)
    axes[2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Z Error (m)', fontsize=12, fontweight='bold')
    axes[2].set_title('Z Position Error (Altitude)', fontsize=13)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Print error statistics per axis
    print("\n" + "="*60)
    print("ERROR OVER TIME BY AXIS")
    print("="*60)
    print(f"\nX Error (Easting):")
    print(f"  Mean:   {np.mean(errors_x):.2f} m")
    print(f"  Median: {np.median(errors_x):.2f} m")
    print(f"  Std:    {np.std(errors_x):.2f} m")
    print(f"  Max:    {np.max(errors_x):.2f} m")
    print(f"  Min:    {np.min(errors_x):.2f} m")
    
    print(f"\nY Error (Northing):")
    print(f"  Mean:   {np.mean(errors_y):.2f} m")
    print(f"  Median: {np.median(errors_y):.2f} m")
    print(f"  Std:    {np.std(errors_y):.2f} m")
    print(f"  Max:    {np.max(errors_y):.2f} m")
    print(f"  Min:    {np.min(errors_y):.2f} m")
    
    print(f"\nZ Error (Altitude):")
    print(f"  Mean:   {np.mean(errors_z):.2f} m")
    print(f"  Median: {np.median(errors_z):.2f} m")
    print(f"  Std:    {np.std(errors_z):.2f} m")
    print(f"  Max:    {np.max(errors_z):.2f} m")
    print(f"  Min:    {np.min(errors_z):.2f} m")
    print("="*60 + "\n")
