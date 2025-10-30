import formulas
import util
import utm
import filters
import plot

srad_data = util.parse_json("./srad_data.json")

# Original GPS data
x_gps = []
y_gps = []
z_gps = []
t_gps = []

# Combined data (GPS + interpolated)
x = []
y = []
z = []
t = []

for packet in srad_data["packets"]:
    print("------------------------ NEW PACKET ------------------------")

    first_sample = packet['samples'][0]
    last_sample = packet['samples'][-1]

    # Treating easting/northing as x and y
    first_sample_x, first_sample_y, _, _  = utm.from_latlon(first_sample['lat'], first_sample['lon'])
    last_sample_x, last_sample_y, _, _ = utm.from_latlon(last_sample['lat'], last_sample['lon'])

    first_sample_z = first_sample['alt']
    last_sample_z = last_sample['alt']

    delta_t = (last_sample['time'] - first_sample['time'])

    vel_x = (last_sample_x - first_sample_x) / delta_t
    vel_y = (last_sample_y - first_sample_y) / delta_t
    vel_z = (last_sample_z - first_sample_z) / delta_t

    # FLIP X AND Z since accelerometer mounted sideways. TODO: Change this to a positional based.
    acc_x = last_sample['acc_z']
    acc_y = last_sample['acc_y']
    acc_z = last_sample['acc_x']

    # Store original GPS data
    x_gps.append(last_sample_x)
    y_gps.append(last_sample_y)
    z_gps.append(last_sample_z)
    t_gps.append(last_sample['time'])

    # Also add to combined data
    x.append(last_sample_x)
    y.append(last_sample_y)
    z.append(last_sample_z)
    t.append(last_sample['time'])

    ITERATIONS = 10
    ITERATION_TIME = 1 / ITERATIONS # 1 second / # of iterations

    for i in range(ITERATIONS):
        x.append(formulas.const_acc_eq(vel_x, ITERATION_TIME, acc_x, x[-1]))
        y.append(formulas.const_acc_eq(vel_y, ITERATION_TIME, acc_y, y[-1]))
        z.append(formulas.const_acc_eq(vel_z, ITERATION_TIME, acc_z, z[-1]))
        t.append(last_sample['time'] + ITERATION_TIME * (i + 1))

plot.plot_position_data(t, x, y, z, t_gps, x_gps, y_gps, z_gps)

util.calculate_and_print_errors(t, x, y, z, t_gps, x_gps, y_gps, z_gps)