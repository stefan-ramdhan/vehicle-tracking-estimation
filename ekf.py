# from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
import pandas as pd
def init_ekf(initial_state, init_P, sigma_a, sigma_omega, sigma_x, sigma_y, sigma_theta, dt):

    ekf = ExtendedKalmanFilter(dim_x = 5, dim_z = 3)

    ekf.x = np.array(initial_state).reshape(ekf.dim_x, 1)
    pos_x, pos_y, v, theta, omega = ekf.x
    v = v[0]
    theta = theta[0]
    omega = omega[0]

    ekf.F = np.zeros((ekf.dim_x, ekf.dim_x))

    ekf.R = np.array([ #Measurement covariance R
                    [sigma_x**2, 0, 0],
                    [ 0, sigma_y**2, 0],
                    [ 0, 0, sigma_theta**2]
                    ])
    
    ekf.H = np.array([
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0]
                    ])

    if abs(omega) > 1e-5:
        G = np.array([
            [v * np.cos(theta) / omega, 0],
            [v * np.sin(theta) / omega, 0],
            [dt, 0],
            [0, 1/2 * dt**2],
            [0, dt]
        ])
    else:
        G = np.array([
            [0, 0],
            [0, 0],
            [dt, 0],
            [0, 1/2 * dt**2],
            [0, dt]
        ])

    # covariance of process noise. assumes independence
    ekf.cov_w = np.diag([sigma_a**2, sigma_omega**2])

    ekf.Q = G @ ekf.cov_w @ G.T

    ekf.P = init_P * np.eye(5)

    return ekf


def state_transition(x, dt):
    x_pos, y_pos, v, theta, omega = x
    epsilon = 1e-5
    if abs(omega) > epsilon: #nonlinear change in x and y
        dx = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        dy = (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
    else: #linearized version when omega is near 0, to prevent issues with singularity.
        dx = v * dt * np.cos(theta)
        dy = v * dt * np.sin(theta)

    dv = 0  # Constant velocity assumption
    dtheta = omega * dt
    domega = 0  # Constant turn rate assumption

    new_theta = theta + dtheta
    # new_theta = normalize_angle(new_theta)

    return np.array([x_pos + dx, #doesn't include process noise because we incorporate that in the Q matrix.
                     y_pos + dy,
                     v + dv,
                     new_theta,
                     omega + domega])

# Jacobian of the state transition
def state_transition_jacobian(x, dt):
    x_pos, y_pos, v, theta, omega = x
    epsilon = 1e-5
    F = np.eye(5)

    if abs(omega) > epsilon:
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_theta_wdt = np.sin(theta + omega * dt)
        cos_theta_wdt = np.cos(theta + omega * dt)

        # Partial derivatives for x
        F[0, 0] = 1
        F[0, 1] = 0
        F[0, 2] = (sin_theta_wdt - sin_theta) / omega
        F[0, 3] = (v / omega) * (cos_theta_wdt - cos_theta)
        F[0, 4] = (-v / (omega ** 2)) * (sin_theta_wdt - sin_theta) + (v * dt / omega) * cos_theta_wdt

        # Partial derivatives for y
        F[1, 0] = 0
        F[1, 1] = 1
        F[1, 2] = (-cos_theta_wdt + np.cos(theta)) / omega
        F[1, 3] = (v / omega) * (sin_theta_wdt - np.sin(theta))
        F[1, 4] = (-v / (omega ** 2)) * (-cos_theta_wdt + np.cos(theta)) + (v * dt / omega) * sin_theta_wdt

    else:
        # Partial derivatives for straight motion
        F[0, 2] = dt * np.cos(theta)
        F[0, 3] = -v * dt * np.sin(theta)
        F[1, 2] = dt * np.sin(theta)
        F[1, 3] = v * dt * np.cos(theta)
        # F[0,4] and F[1,4] remain 0

    # Partial derivatives for theta and omega
    F[3, 4] = dt
    # F[2,2], F[4,4] are already 1

    return F

def measurement_jacobian(x):
    return np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ])

def update_Q(ekf, dt):

    pos_x, pos_y, v, theta, omega = ekf.x
    v = v[0]
    theta = theta[0]
    omega = omega[0]
    
    if abs(omega) > 1e-5:
        G = np.array([
            [v * np.cos(theta) / omega, 0],
            [v * np.sin(theta) / omega, 0],
            [dt, 0],
            [0, 1/2 * dt**2],
            [0, dt]
        ])
    else:
        G = np.array([
            [0, 0],
            [0, 0],
            [dt, 0],
            [0, 1/2 * dt**2],
            [0, dt]
        ])

    ekf.Q = G @ ekf.cov_w @ G.T

def measurement_function(x):

    return np.array([x[0], x[1], x[3]])


def predict(ekf, dt):

    update_Q(ekf, dt)
    ekf.x = state_transition(ekf.x, dt)
    ekf.F = state_transition_jacobian(ekf.x, dt)
    ekf.P = np.matmul(np.matmul(ekf.F, ekf.P), ekf.F.T) + ekf.Q

def update(ekf, z):

    # predicted values based on state update
    z_pred = measurement_function(ekf.x).reshape(3,1)
    # innovation
    y = z - z_pred
    
    # innovation covariance
    S = np.matmul(np.matmul(ekf.H, ekf.P), ekf.H.T) + ekf.R

    # Kalman Gain
    K = np.matmul(np.matmul(ekf.P, ekf.H.T), np.linalg.inv(S))

    # State update
    ekf.x = ekf.x + np.matmul(K, y)

    # Covariance Update
    I = np.eye(ekf.dim_x)
    IKH = np.subtract(I, np.matmul(K, ekf.H))
    ekf.P = np.matmul(IKH, ekf.P)

    return y, S

def read_measurements(measurements, idx):
    
    x = measurements['x'].iloc[idx]
    y = measurements['y'].iloc[idx]
    theta = measurements['yaw'].iloc[idx]
    ts = measurements['timestamp'].iloc[idx]

    return ts, np.array([
                    [x],
                    [y],
                    [theta]
                    ])

def execute_ekf(measurements):

    init_x = np.array([
        [0],
        [0],
        [0.1],
        [0],
        [0.1]
    ])  # [x, y, v, theta (rad), omega (rad/s)]
    init_P = 1e8
    sigma_a = 2e-2  # Acceleration noise standard deviation
    sigma_omega = np.deg2rad(30)  # Angular acceleration noise standard deviation
    sigma_x = 2e-2  # Measurement noise standard deviation for x
    sigma_y = 2e-2  # Measurement noise standard deviation for y
    sigma_theta = np.deg2rad(10)  # Measurement noise standard deviation for theta
    # sigma_theta = 0.01
    dt = 0.1  # Time step in seconds

    ekf = init_ekf(init_x, init_P, sigma_a, sigma_omega, sigma_x, sigma_y, sigma_theta, dt)

    assert ekf.x.shape == (ekf.dim_x, 1)
    assert ekf.F.shape == (ekf.dim_x, ekf.dim_x)
    assert ekf.P.shape == (ekf.dim_x, ekf.dim_x)

    len_data = measurements.shape[0]
    i = 0
    x_hat = []
    P_arr = []
    y_arr = []
    S_arr = []
    ts = 0

    while i < len_data:

        old_ts = ts
        ts, z = read_measurements(measurements, i)

        if i == 0:
            dt = 0.1
        else:
            dt = ts - old_ts

        predict(ekf, dt)

        # ensure shapes
        assert ekf.x.shape == (ekf.dim_x, 1)
        assert ekf.F.shape == (ekf.dim_x, ekf.dim_x)
        assert ekf.P.shape == (ekf.dim_x, ekf.dim_x)

        y, S = update(ekf, z)

        P_arr.append(ekf.P)
        x_hat.append(ekf.x)
        y_arr.append(y)
        S_arr.append(S)
        i += 1
    
    flattened_data = [arr.flatten() for arr in x_hat]
    return pd.DataFrame(flattened_data, columns=['x', 'y', 'v', 'theta', 'omega']), P_arr, y_arr, S_arr
    

class ExtendedKalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.process_sigma = None
        self.x = np.zeros((dim_x, 1))
        self.F = np.zeros((dim_x, dim_x))
        self.H = np.zeros((dim_z, dim_x))
        self.P = np.zeros((dim_x, dim_x))
        self.R = np.zeros((dim_z, dim_z))
        self.Q = np.zeros((dim_x, dim_x))