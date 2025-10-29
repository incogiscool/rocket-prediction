# Constant Acceleration Equation
def const_acc_eq(vel, time, accel, orig_pos):
    return orig_pos + vel * time + 0.5 * accel * (time ** 2)