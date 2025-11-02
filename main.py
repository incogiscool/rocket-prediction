import formulas
import util
import utm
import plot

# Load sensor data
srad_data = util.parse_json("./srad_data.json")

# GPS data arrays
x_gps = []
y_gps = []
z_gps = []
t_gps = []

# Combined data (GPS + interpolated)
x = []
y = []
z = []
t = []

# Velocity arrays
vel_x = []
vel_y = []
vel_z = []

# Acceleration arrays
acc_x = []
acc_y = []
acc_z = []

# Constants
ITERATIONS = 10
ITERATION_TIME = 1 / ITERATIONS  # 0.1 seconds per iteration


for packet in srad_data["packets"]:
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
        
        continue

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

    # Check for velocity outliers - if detected, use previous velocity
    if len(vel_x) >= 20:
        is_outlier_vx, (lower_vx, upper_vx) = util.is_outlier_iqr(new_vel_x, vel_x[-20:], 5000.0)
        if is_outlier_vx:
            print(f"VELOCITY X OUTLIER at timestamp {packet['timestamp']}: {new_vel_x:.2f} m/s -> using previous {vel_x[-1]:.2f} m/s")
            new_vel_x = vel_x[-1]

    if len(vel_y) >= 20:
        is_outlier_vy, (lower_vy, upper_vy) = util.is_outlier_iqr(new_vel_y, vel_y[-20:], 5000)
        if is_outlier_vy:
            print(f"VELOCITY Y OUTLIER at timestamp {packet['timestamp']}: {new_vel_y:.2f} m/s -> using previous {vel_y[-1]:.2f} m/s")
            new_vel_y = vel_y[-1]

    if len(vel_z) >= 20:
        is_outlier_vz, (lower_vz, upper_vz) = util.is_outlier_iqr(new_vel_z, vel_z[-20:], 5000.0)
        if is_outlier_vz:
            print(f"VELOCITY Z OUTLIER at timestamp {packet['timestamp']}: {new_vel_z:.2f} m/s -> using previous {vel_z[-1]:.2f} m/s")
            new_vel_z = vel_z[-1]

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

    # Interpolate positions between GPS samples
    for i in range(ITERATIONS):
        if packet['timestamp'] == 5317.25:
            print(x[-1], y[-1], z[-1], vel_x[-1], vel_y[-1], vel_z[-1])
        x.append(formulas.const_acc_eq(vel_x[-1], ITERATION_TIME, acc_x[-1], x[-1]))
        y.append(formulas.const_acc_eq(vel_y[-1], ITERATION_TIME, acc_y[-1], y[-1]))
        z.append(formulas.const_acc_eq(vel_z[-1], ITERATION_TIME, acc_z[-1], z[-1]))
        t.append(last_sample['time'] + ITERATION_TIME * (i + 1))

# Visualize results
plot.plot_position_data(t, x, y, z, t_gps, x_gps, y_gps, z_gps)

# Calculate and print error statistics
util.calculate_and_print_errors(t, x, y, z, t_gps, x_gps, y_gps, z_gps)

# Uncomment to see error over time plot
# plot.plot_error_over_time(t, x, y, z, t_gps, x_gps, y_gps, z_gps)
