# vehicle-estimation

Infrastructure-based tracking of a single vehicle for ECE 771, a graduate course at McMaster focused on Parameter and State Estimation.

I implement a Linear Kalman Filter, and an Extended Kalman Filter to estimate the state of a vehicle. 

I use a white noise acceleration bicycle model as my process model for the Linear KF. Since the vehicle motion is non-linear, a linear KF is insufficient to accurately estimate vehicle state. Therefore, I also implement an Extended Kalman Filter, in which I use a Constant Turn Rate and Velocity Model. 

For the dataset, use the A9 dataset \[1] for infrastructure-based lidar and camera data of urban and highway environments in Germany. 

# Citation

\[1] Creß, Christian et al. “A9-Dataset: Multi-Sensor Infrastructure-Based Dataset for Mobility Research.” 2022 IEEE Intelligent Vehicles Symposium (IV) (2022): 965-970.

