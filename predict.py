import formulas
import util
import utm


def process_packet(packet, x, y, z, t, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, x_gps, y_gps, z_gps, t_gps):
    """
    Process a single packet and update position/velocity/acceleration arrays.
    
    Parameters:
    -----------
    packet : dict
        Data packet containing samples with GPS, acceleration data
    x, y, z, t : list
        Arrays of all positions and timestamps (GPS + interpolated)
    vel_x, vel_y, vel_z : list
        Arrays of velocities for each axis
    acc_x, acc_y, acc_z : list
        Arrays of accelerations for each axis
    x_gps, y_gps, z_gps, t_gps : list
        Arrays of GPS positions and timestamps
    
    Returns:
    --------
    None (modifies arrays in place)
    """
    samples = packet['samples']
    first_sample = samples[0]
    last_sample = samples[-1]
    
    # Handle invalid GPS/altitude data - use constant acceleration equation
    if (last_sample['lat'] == 0.0 or last_sample['lon'] == 0.0 or last_sample['alt'] == 0.0) and len(x) > 0:
        time_since_last = last_sample['time'] - t[-1]
        
        # Calculate positions using last known values
        new_x = formulas.const_acc_eq(vel_x[-1], time_since_last, acc_x[-1], x[-1])
        new_y = formulas.const_acc_eq(vel_y[-1], time_since_last, acc_y[-1], y[-1])
        new_z = formulas.const_acc_eq(vel_z[-1], time_since_last, acc_z[-1], z[-1])
        
        # Store calculated positions
        x.append(new_x)
        y.append(new_y)
        z.append(new_z)
        t.append(last_sample['time'])
        
        # Store as GPS data (use calculated instead of invalid 0)
        x_gps.append(new_x)
        y_gps.append(new_y)
        z_gps.append(new_z)
        t_gps.append(last_sample['time'])
        
        # Maintain last known velocities
        vel_x.append(vel_x[-1])
        vel_y.append(vel_y[-1])
        vel_z.append(vel_z[-1])
        
        # Update accelerations from sensor (accelerometer is mounted sideways: flip X and Z)
        acc_x.append(last_sample['acc_z'])
        acc_y.append(last_sample['acc_y'])
        acc_z.append(last_sample['acc_x'])
        
        return

    # Calculate velocity using last 3 samples for X and Y (more stable)
    num_samples = min(3, len(samples))
    samples_for_xy_velocity = samples[-num_samples:]
    
    first_xy_sample = samples_for_xy_velocity[0]
    last_xy_sample = samples_for_xy_velocity[-1]
    
    # Convert to UTM coordinates
    first_xy_x, first_xy_y, _, _ = utm.from_latlon(first_xy_sample['lat'], first_xy_sample['lon'])
    last_xy_x, last_xy_y, _, _ = utm.from_latlon(last_xy_sample['lat'], last_xy_sample['lon'])
    last_sample_x, last_sample_y, _, _ = utm.from_latlon(last_sample['lat'], last_sample['lon'])
    
    # Calculate time deltas
    delta_t_xy = (last_xy_sample['time'] - first_xy_sample['time']) or 0.01
    delta_t_z = (last_sample['time'] - first_sample['time']) or 0.01

    # Calculate velocities
    if first_xy_x == 0.0 and len(vel_x) > 0:
        new_vel_x = vel_x[-1]
    else:
        new_vel_x = (last_xy_x - first_xy_x) / delta_t_xy

    if first_xy_y == 0.0 and len(vel_y) > 0:
        new_vel_y = vel_y[-1]
    else:
        new_vel_y = (last_xy_y - first_xy_y) / delta_t_xy

    if first_sample['alt'] == 0.0 and len(vel_z) > 0:
        new_vel_z = vel_z[-1]
    else:
        new_vel_z = (last_sample['alt'] - first_sample['alt']) / delta_t_z

    # Check for velocity outliers and get corrected velocities
    new_vel_x, new_vel_y, new_vel_z = util.check_velocity_outliers(
        new_vel_x, new_vel_y, new_vel_z,
        vel_x, vel_y, vel_z,
        packet['timestamp']
    )

    # Append the (possibly corrected) velocities
    vel_x.append(new_vel_x)
    vel_y.append(new_vel_y)
    vel_z.append(new_vel_z)

    # Store accelerations (accelerometer is mounted sideways: flip X and Z)
    acc_x.append(last_sample['acc_z'])
    acc_y.append(last_sample['acc_y'])
    acc_z.append(last_sample['acc_x'])
    
    # Store GPS data
    x_gps.append(last_sample_x)
    y_gps.append(last_sample_y)
    z_gps.append(last_sample['alt'])
    t_gps.append(last_sample['time'])

    # Store current position
    x.append(last_sample_x)
    y.append(last_sample_y)
    z.append(last_sample['alt'])
    t.append(last_sample['time'])