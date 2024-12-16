import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

def init_4d_kf(linear_kf_4d):

    del_t = 0.1
    init_P = 1000 # high uncertainty in our initial state values 
    measurement_sigma = 2e-2 # based on specification (I add noise to GT, so this is from Gaussian sigma)
    try:
        process_sigma = linear_kf_4d.process_sigma
    except:
        process_sigma = 2e-4
    init_Q = 1e-2 # How much uncertainty we have in state transition (low -> follow state transition more than measurements)

    linear_kf_4d.x = np.array(
                    [[0], # position x
                    [0], # position y    
                    [0], # velocity x
                    [0]] # velocity y
                    )  
    linear_kf_4d.F = np.array(
                    [[1, 0, del_t, 0],
                    [0, 1, 0, del_t],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]
                    )
    linear_kf_4d.H = np.array( 
                    [[1, 0, 0, 0],
                    [0, 1, 0, 0]]
                    )

    linear_kf_4d.P = np.array( # State covariance P
                    [[init_P, 0, 0, 0],
                    [0, init_P, 0, 0],
                    [0, 0, init_P, 0],
                    [0, 0, 0, init_P]]
                    )

    linear_kf_4d.R = np.array([ #Measurement covariance R
                    [measurement_sigma**2, 0],
                    [ 0, measurement_sigma**2]
                    ])


    linear_kf_4d.Q = process_sigma**2 * np.array( # Process Noise Covariance Q
                    [[del_t**4/4, 0, del_t**3/2, 0],
                    [0, del_t**4/4, 0, del_t**3/2],
                    [del_t**3/2, 0, del_t**2, 0],
                    [0, del_t**3/2, 0, del_t**2]]
                    )
    # linear_kf_4d.Q = 1 * np.array([ # Process Noise Covariance Q
    #                 [init_Q, 0, 0, 0],
    #                 [0, init_Q, 0, 0],
    #                 [0, 0, init_Q, 0],
    #                 [0, 0, 0, init_Q]
    #                 ])
    # linear_kf_4d.Q = 0.5 * np.array([ # Process Noise Covariance Q
    #                 [del_t**4/4, 0, 0, 0],
    #                 [0, del_t**4/4, 0, 0],
    #                 [0, 0, del_t**2, 0],
    #                 [0, 0, 0, del_t**2]
    #                 ])


def execute_kf(measurements):

    len_data = measurements.shape[0]
    x_hat = []
    i = 0
    P = []
    y = []
    S = []

    linear_kf_4d = LinearKF(dim_x = 4, dim_z = 2)
    init_4d_kf(linear_kf_4d)
    ts = 0 #arbitrarily init first timestamp.

    while i < len_data:
        old_ts = ts
        ts, z = read_measurements(measurements, i)
        del_t = ts - old_ts
        
        if i > 0:
            linear_kf_4d.updateFQ(del_t)
        linear_kf_4d.predict()
        y_estim, S_estim = linear_kf_4d.update(z)

        # Append to list so I can plot results
        x_hat.append(linear_kf_4d.x)
        P.append(linear_kf_4d.P)
        y.append(y_estim)
        S.append(S_estim)
        
        i += 1

    flattened_data = [arr.flatten() for arr in x_hat]
    df = pd.DataFrame(flattened_data, columns=['x', 'y', 'lat_vel', 'long_vel'])
    return df, P, y, S

def read_measurements(measurements, idx):
    
    x = measurements['x'].iloc[idx]
    y = measurements['y'].iloc[idx]
    ts = measurements['timestamp'].iloc[idx]

    return ts, np.array([
                    [x],
                    [y]
                    ])


class LinearKF:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.process_sigma = 2e-4 # 2e-4 works well with measurement_sigma = 2e-2 (2e-2 works well, but only for straight segment. For turning segment it freaks out.)
        self.x = np.zeros((dim_x, 1))
        self.F = np.zeros((dim_x, dim_x))
        self.H = np.zeros((dim_z, dim_x))
        self.P = np.zeros((dim_x, dim_x))
        self.R = np.zeros((dim_z, dim_z))
        self.Q = np.zeros((dim_x, dim_x))

    def predict(self):

        # State prediction
        self.x = np.matmul(self.F, self.x)

        # Covariance Prediction
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        
        # Innovation calculation
        y = np.subtract(z, np.matmul(self.H, self.x))

        # Innovation covariance
        S = np.matmul(np.matmul(self.H, self.P), self.H.T) - self.R
        
        # Kalman Gain
        K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(S))

        # Update state
        self.x = np.add(self.x, np.matmul(K, y))

        # Update covariance
        I = np.eye(self.dim_x)
        IKH = np.subtract(I, np.matmul(K, self.H))
        self.P = np.matmul(IKH, self.P)

        return y, S

    def updateFQ(self, del_t):
        # del_t = 0.1
        self.F = np.array(
                [[1, 0, del_t, 0],
                [0, 1, 0, del_t],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
                )
        
        self.Q = self.process_sigma**2 * np.array( # Process Noise Covariance Q
                [[del_t**4/4, 0, del_t**3/2, 0],
                [0, del_t**4/4, 0, del_t**3/2],
                [del_t**3/2, 0, del_t**2, 0],
                [0, del_t**3/2, 0, del_t**2]]
                )
    