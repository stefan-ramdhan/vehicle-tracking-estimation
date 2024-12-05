from input import *
from kalman import *
from stats import *
import matplotlib.pyplot as plt
import pandas as pd


DATASET_PATH = r"a9_dataset\r2_s1\a9_dataset_r02_s01\labels_point_clouds\s110_lidar_ouster_south"
OBJECT_ID= "b89ea978-7693-420a-a153-371b12242312"
DATASET = DATASET_PATH.split("\\")[1]
'''

---------------------STRAIGHTAWAY: HIGHWAY SCENARIO---------------------

approximately constant velocity vehicle going in straight line is blue car
DATASET_PATH = "a9_dataset\straightaway\_labels\s050_camera_basler_south_16mm"
OBJECT_ID= "fcc872a1-936b-4bb7-b734-ae2ebf652e29"

NON LINEAR SCENARIO: RED VAN
red van taking offramp (constnat turn approx) (looks pretty good after calibraiton):
DATASET_PATH = "a9_dataset\straightaway\_labels\s040_camera_basler_north_50mm"
OBJECT_ID = "aa17cf54-f34c-4904-a7c2-9559fce6aa48"

same red van, from south:
DATASET_PATH = "a9_dataset\straightaway\_labels\s050_camera_basler_south_16mm"
OBJECT_ID = "a59f612e-ef00-408d-aeee-10769584b7dc"

---------------------INTERSECTION TURNING SCENARIO---------------------

red van turning:
DATASET_PATH = r"a9_dataset\r2_s1\a9_dataset_r02_s01\labels_point_clouds\s110_lidar_ouster_north"
OBJECT_ID= "fdeb9f7d-9757-4273-9168-eb155f0b7997"

Same Red Van Turning (looks really good) - can split into 3 sections (striaght, turn, straight)
DATASET_PATH = r"a9_dataset\r2_s1\a9_dataset_r02_s01\labels_point_clouds\s110_lidar_ouster_south"
OBJECT_ID= "b89ea978-7693-420a-a153-371b12242312"

Pretty straight (not amazing)
DATASET_PATH = r"a9_dataset\r2_s1\a9_dataset_r02_s01\labels_point_clouds\s110_lidar_ouster_north"
OBJECT_ID= "3fb95e77-a675-4695-9200-a0fb292d5e61"


Straight then turn:
DATASET_PATH = r"a9_dataset\r2_s1\a9_dataset_r02_s01\labels_point_clouds\s110_lidar_ouster_north"
OBJECT_ID= "511f0fee-ff02-4925-bff5-b53a2fbc5c71"
'''

def main():
    pd.set_option('display.float_format', lambda x: '%.9f' % x)
    pd.set_option('display.max_rows', 500)

    vals = load_dataset(DATASET_PATH, OBJECT_ID)
    # print(vals)


    if DATASET == "r0_s4_lidar":
        x_pos = vals['location_x']
        y_pos = vals['location_y']
        plt.plot(x_pos, y_pos)
        plt.show()
    elif DATASET == "r2_s1":

        # Select which part of the turning sequence we want
        scenario = "straight1" # straight1, turning, straight2, all

        sec1_cutoff = 46 # point at which vehicle goes from straight to turning (46 is good)
        sec2_cutoff = 85 # vehicle goes from turning to straight        

        if scenario == "straight1":
            vals = vals.iloc[:sec1_cutoff]
            vals['x'] = vals["cuboid_val"].apply(lambda x: x[0]).to_numpy()
            vals['y'] = vals["cuboid_val"].apply(lambda x: x[1]).to_numpy()
            vals['yaw'] = vals["cuboid_val"].apply(lambda x: x[5]).to_numpy()
        elif scenario == "turning":
            vals = vals.iloc[sec1_cutoff:sec2_cutoff]
            vals['x'] = vals["cuboid_val"].apply(lambda x: x[0]).to_numpy()
            vals['y'] = vals["cuboid_val"].apply(lambda x: x[1]).to_numpy()
            vals['yaw'] = vals["cuboid_val"].apply(lambda x: x[5]).to_numpy()
        elif scenario == "straight2":
            vals = vals.iloc[sec2_cutoff:]
            vals['x'] = vals["cuboid_val"].apply(lambda x: x[0]).to_numpy()
            vals['y'] = vals["cuboid_val"].apply(lambda x: x[1]).to_numpy()
            vals['yaw'] = vals["cuboid_val"].apply(lambda x: x[5]).to_numpy()
        elif scenario == "all":
            vals['x'] = vals["cuboid_val"].apply(lambda x: x[0]).to_numpy()
            vals['y'] = vals["cuboid_val"].apply(lambda x: x[1]).to_numpy()
            vals['yaw'] = vals["cuboid_val"].apply(lambda x: x[5]).to_numpy()
        else:
            print("Invalid trjaecotry segment")


        '''
        vals is given as:
        0: x
        1: y
        2: z
        3: roll
        4: pitch
        5: yaw
        6: confidence
        7: length
        8: width
        9: height
        '''

        # Compute ground truth
        vals = compute_derivatives(vals)
        # print(vals)

        # Add noise to ground truth, to simulate measurements, and compute measured lat_vel, long_vel, etc.
        sigma = 0.02
        noisy_vals = simulate_measurements(vals, sigma)

        # # Plot GT vel, measured vel (lateral)
        # plt.plot(range(len(vals['lat_accel'])), vals['lat_accel'], label='GT vel') 
        # plt.plot(range(len(noisy_vals['lat_accel'])), noisy_vals['lat_accel'], label='measured vel') 
        # plt.legend()
        # plt.show()

        # Call Kalman Filter
        x_hat = execute_kf(noisy_vals, "my_4d_linear")

        '''Plot ground truth & measurements (trajectory) '''
        plt.figure(1)
        plt.scatter(noisy_vals['x'], noisy_vals['y'], label='measured trajectory')
        plt.scatter(vals['x'], vals['y'], label="ground truth")
        plt.scatter(x_hat['x'], x_hat['y'], label='estimated trajectory')
        plt.ylim([-65, 10])
        plt.xlim([10, 55])
        plt.legend()

        # mse_x = compute_mse(vals['x'], x_hat['x'])
        # mse_y = compute_mse(vals['y'], x_hat['y'])

        '''Plot lat_vel long_vel'''
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.plot(range(len(vals['lat_vel'])), vals['lat_vel'], label='GT lat_vel')
        plt.plot(range(len(x_hat['lat_vel'])), x_hat['lat_vel'], label='estimated lat_vel')
        # plt.plot(range(len(noisy_vals['lat_vel'])), noisy_vals['lat_vel'], label='measured lat_vel')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(range(len(vals['long_vel'])), vals['long_vel'], label='GT long_vel')
        plt.plot(range(len(x_hat['long_vel'])), x_hat['long_vel'], label='estimated long_vel')
        # plt.plot(range(len(noisy_vals['long_vel'])), noisy_vals['long_vel'], label='measured long_vel')
        plt.legend()
        plt.show()

        # fig = plt.figure(2)
        # plt.subplot(2, 1, 1)
        # plt.plot(range(len(vals['x'])), mse_x, label='MSE of x')
        # plt.subplot(2, 1, 2)
        # plt.plot(range(len(vals['x'])), mse_y, label='MSE of y')
        # plt.legend()
        # plt.show()

        

    elif DATASET == "straightaway":
        x_centroid_2d = []
        y_centroid_2d = []
        for i in range(len(vals)):
            x_centroid_2d.append((vals[i]['bottom_left_front'][0] + vals[i]['bottom_right_back'][0]) / 2)
            y_centroid_2d.append((vals[i]['bottom_left_front'][1] + vals[i]['bottom_right_back'][1]) / 2)
    else:
        pass


    # for idx, box3d in enumerate(bb_vals):
    #     print(f"Frame {idx + 1}: {box3d}")

    # plt.scatter(x_centroid_2d, y_centroid_2d)
    # plt.ylim((0, 1))
    # plt.xlim((0, 1))
    # plt.gca().invert_yaxis()
    # plt.show()

if __name__=="__main__":
    main()



'''
TODO:
Produce measurements (noise to GT)
Implement KF for position for a single car, WNA model
Extend KF to vel, accel, yaw, yaw rate estimation
Implement EKF using CTRV model (turning scenario)
Implement IMM estimator to esimate both non-linear and linear case (entire trajectory)
'''


