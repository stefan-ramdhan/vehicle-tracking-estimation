import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

def init_4d_kf(linear_kf_4d):

    del_t = 0.1
    init_P = 1000 # high uncertainty in our initial state values 
    measurement_sigma = 0.02 # based on specification (I add noise to GT, so this is from Gaussian sigma)
    try:
        process_sigma = linear_kf_4d.process_sigma**2
    except:
        process_sigma = 1e7
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


    # linear_kf_4d.Q = process_sigma * np.array( # Process Noise Covariance Q
    #                 [[del_t**4/4, 0, del_t**3/2, 0],
    #                 [0, del_t**4/4, 0, del_t**3/2],
    #                 [del_t**3/2, 0, del_t**2, 0],
    #                 [0, del_t**3/2, 0, del_t**2]]
    #                 )
    linear_kf_4d.Q = 1 * np.array([ # Process Noise Covariance Q
                    [init_Q, 0, 0, 0],
                    [0, init_Q, 0, 0],
                    [0, 0, init_Q, 0],
                    [0, 0, 0, init_Q]
                    ])
    # linear_kf_4d.Q = 0.5 * np.array([ # Process Noise Covariance Q
    #                 [del_t**4/4, 0, 0, 0],
    #                 [0, del_t**4/4, 0, 0],
    #                 [0, 0, del_t**2, 0],
    #                 [0, 0, 0, del_t**2]
    #                 ])

def execute_kf(measurements, kf_type="4d_linear"):

    len_data = measurements.shape[0]
    x_hat = []
    i = 0

    if kf_type == "4d_linear":
        linear_kf_4d = KalmanFilter(dim_x = 4,dim_z=2)
        init_4d_kf(linear_kf_4d)
        while i < len_data:
            z = read_measurements(measurements, i)
            linear_kf_4d.predict()
            linear_kf_4d.update(z)
            x_hat.append(linear_kf_4d.x)
            i += 1

    if kf_type == "my_4d_linear":
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
            linear_kf_4d.update(z)
            x_hat.append(linear_kf_4d.x)
            i += 1

    flattened_data = [arr.flatten() for arr in x_hat]
    df = pd.DataFrame(flattened_data, columns=['x', 'y', 'lat_vel', 'long_vel'])
    return df

def read_measurements(measurements, idx):
    x = measurements['x'][idx]
    y = measurements['y'][idx]
    ts = measurements['timestamp'][idx]

    return ts, np.array([
                    [x],
                    [y]
                    ])


class LinearKF:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.process_sigma = 1e7
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

    def updateFQ(self, del_t):
        # del_t = 0.1
        self.F = np.array(
                [[1, 0, del_t, 0],
                [0, 1, 0, del_t],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
                )
        
        # self.Q = self.process_sigma**2 * np.array( # Process Noise Covariance Q
        #         [[del_t**4/4, 0, del_t**3/2, 0],
        #         [0, del_t**4/4, 0, del_t**3/2],
        #         [del_t**3/2, 0, del_t**2, 0],
        #         [0, del_t**3/2, 0, del_t**2]]
        #         )