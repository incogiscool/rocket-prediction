def lowpass(pos, prev_pos, alpha):
    return alpha * pos + (1 - alpha) * prev_pos

# def highpass(pos, prev_pos, prev_filtered, alpha):
    """
    High-pass filter implementation.
    
    Args:
        pos: Current position/signal value
        prev_pos: Previous position/signal value
        prev_filtered: Previous filtered output
        alpha: Filter coefficient (0 to 1)
               Higher alpha = higher cutoff frequency (less filtering)
               Lower alpha = lower cutoff frequency (more filtering)
    
    Returns:
        Filtered signal value
    
    Formula: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
    where y is filtered output and x is input signal
    """
    return alpha * (prev_filtered + pos - prev_pos)


def highpass_simple(pos, prev_pos, alpha):
    """
    Simplified high-pass filter (first-order).
    
    Args:
        pos: Current position/signal value
        prev_pos: Previous filtered value
        alpha: Filter coefficient (0 to 1)
               Higher alpha = more aggressive filtering
    
    Returns:
        Filtered signal value
    
    Formula: y[n] = alpha * (x[n] - x[n-1])
    """
    return alpha * (pos - prev_pos)