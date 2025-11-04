import numpy as np


class KalmanFilter:
    """
    3D Kalman Filter for rocket trajectory prediction with delayed sensor measurements.
    
    State vector: [x, y, z, vx, vy, vz]
    - Position (x, y, z) in meters
    - Velocity (vx, vy, vz) in m/s
    """
    
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=10.0):
        """
        Initialize the Kalman Filter.
        
        Parameters:
        -----------
        dt : float
            Time step between measurements (default 1 second for GPS updates)
        process_noise : float
            Process noise covariance (uncertainty in model)
        measurement_noise : float
            Measurement noise covariance (uncertainty in GPS measurements)
        """
        self.dt = dt
        
        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # State covariance matrix (6x6)
        self.P = np.eye(6) * 1000  # Initial uncertainty
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])
        
        # Control input matrix (acceleration input)
        self.B = np.array([
            [0.5 * dt**2, 0, 0],
            [0, 0.5 * dt**2, 0],
            [0, 0, 0.5 * dt**2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt]
        ])
        
        # Measurement matrix (we measure position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(6) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise
        
        # Flag to track if filter has been initialized
        self.initialized = False
    
    def initialize(self, position, velocity):
        """
        Initialize the filter with first measurements.
        
        Parameters:
        -----------
        position : array-like [x, y, z]
            Initial position
        velocity : array-like [vx, vy, vz]
            Initial velocity
        """
        self.state = np.array([
            position[0], position[1], position[2],
            velocity[0], velocity[1], velocity[2]
        ])
        self.initialized = True
    
    def predict(self, acceleration):
        """
        Prediction step: predict state forward in time using acceleration.
        
        Parameters:
        -----------
        acceleration : array-like [ax, ay, az]
            Acceleration measurements
        
        Returns:
        --------
        predicted_state : ndarray
            Predicted state [x, y, z, vx, vy, vz]
        """
        # Control input (acceleration)
        u = np.array(acceleration)
        
        # Predict state: x_k = F * x_k-1 + B * u_k
        self.state = self.F @ self.state + self.B @ u
        
        # Predict covariance: P_k = F * P_k-1 * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state.copy()
    
    def update(self, measurement):
        """
        Update step: correct prediction with GPS measurement.
        
        Parameters:
        -----------
        measurement : array-like [x, y, z]
            GPS position measurement
        
        Returns:
        --------
        updated_state : ndarray
            Updated state [x, y, z, vx, vy, vz]
        """
        z = np.array(measurement)
        
        # Innovation (measurement residual): y = z - H * x
        y = z - self.H @ self.state
        
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.state = self.state + K @ y
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state.copy()
    
    def predict_future(self, time_ahead, acceleration):
        """
        Predict future state without updating internal state.
        Used for compensating 2-second sensor delay.
        
        Parameters:
        -----------
        time_ahead : float
            Time to predict ahead (in seconds)
        acceleration : array-like [ax, ay, az]
            Current acceleration
        
        Returns:
        --------
        future_state : ndarray
            Predicted future state [x, y, z, vx, vy, vz]
        """
        # Create temporary state transition matrix for future prediction
        F_future = np.array([
            [1, 0, 0, time_ahead, 0, 0],
            [0, 1, 0, 0, time_ahead, 0],
            [0, 0, 1, 0, 0, time_ahead],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Control input matrix for future prediction
        B_future = np.array([
            [0.5 * time_ahead**2, 0, 0],
            [0, 0.5 * time_ahead**2, 0],
            [0, 0, 0.5 * time_ahead**2],
            [time_ahead, 0, 0],
            [0, time_ahead, 0],
            [0, 0, time_ahead]
        ])
        
        u = np.array(acceleration)
        future_state = F_future @ self.state + B_future @ u
        
        return future_state
    
    def get_position(self):
        """Get current position estimate [x, y, z]"""
        return self.state[0:3]
    
    def get_velocity(self):
        """Get current velocity estimate [vx, vy, vz]"""
        return self.state[3:6]
