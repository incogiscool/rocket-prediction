import formulas
import util
import utm
import plot
import predict

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


for packet in srad_data["packets"]:
    # Process packet and update all arrays
    predict.process_packet(
        packet,
        x, y, z, t,
        vel_x, vel_y, vel_z,
        acc_x, acc_y, acc_z,
        x_gps, y_gps, z_gps, t_gps
    )

    # Interpolate positions between GPS samples
    ITERATIONS = 10
    ITERATION_TIME = 1 / ITERATIONS  # 0.1 seconds per iteration
    
    for i in range(ITERATIONS):
        x.append(formulas.const_acc_eq(vel_x[-1], ITERATION_TIME, acc_x[-1], x[-1]))
        y.append(formulas.const_acc_eq(vel_y[-1], ITERATION_TIME, acc_y[-1], y[-1]))
        z.append(formulas.const_acc_eq(vel_z[-1], ITERATION_TIME, acc_z[-1], z[-1]))
        t.append(t[-1] + ITERATION_TIME)

# Visualize results
plot.plot_position_data(t, x, y, z, t_gps, x_gps, y_gps, z_gps)

# Calculate and print error statistics
util.calculate_and_print_errors(t, x, y, z, t_gps, x_gps, y_gps, z_gps)

# Uncomment to see error over time plot
# plot.plot_error_over_time(t, x, y, z, t_gps, x_gps, y_gps, z_gps)
