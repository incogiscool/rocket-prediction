# Constant Acceleration Equation
def const_acc_eq(vel, time, acc, orig_pos):
    return orig_pos + vel * time + 0.5 * acc * (time ** 2)

