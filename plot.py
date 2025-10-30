import matplotlib.pyplot as plt

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