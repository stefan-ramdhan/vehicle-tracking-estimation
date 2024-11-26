from input import *
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
            x_vals = vals["cuboid_val"].apply(lambda x: x[0]).to_numpy()[:sec1_cutoff]
            y_vals = vals["cuboid_val"].apply(lambda x: x[1]).to_numpy()[:sec1_cutoff]
            yaw = vals["cuboid_val"].apply(lambda x: x[5]).to_numpy()[:sec1_cutoff]
            vals = vals.iloc[:sec1_cutoff]
        elif scenario == "turning":
            x_vals = vals["cuboid_val"].apply(lambda x: x[0]).to_numpy()[sec1_cutoff:sec2_cutoff]
            y_vals = vals["cuboid_val"].apply(lambda x: x[1]).to_numpy()[sec1_cutoff:sec2_cutoff]
            yaw = vals["cuboid_val"].apply(lambda x: x[5]).to_numpy()[sec1_cutoff:sec2_cutoff]
            vals = vals.iloc[sec1_cutoff:sec2_cutoff]
        elif scenario == "straight2":
            x_vals = vals["cuboid_val"].apply(lambda x: x[0]).to_numpy()[sec2_cutoff:]
            y_vals = vals["cuboid_val"].apply(lambda x: x[1]).to_numpy()[sec2_cutoff:]
            yaw = vals["cuboid_val"].apply(lambda x: x[5]).to_numpy()[sec2_cutoff:]
            vals = vals.iloc[sec2_cutoff:]
        elif scenario == "all":
            x_vals = vals["cuboid_val"].apply(lambda x: x[0]).to_numpy()
            y_vals = vals["cuboid_val"].apply(lambda x: x[1]).to_numpy()
            yaw = vals["cuboid_val"].apply(lambda x: x[5]).to_numpy()

        # Add position & yaw values to df
        vals['x'] = x_vals
        vals['y'] = y_vals
        vals['yaw'] = yaw

        # Compute ground truth velocity (lat, long)
        compute_ground_truth(vals)
        print(vals)

        # Add GT vel to dataframe

        # Compute ground truth accel (lat, long)

        # Compute ground truth yaw, yaw rate


        # plt.plot(x_vals, y_vals)
        # plt.ylim([-65, 10])
        # plt.xlim([10, 55])
        # plt.show()


    elif DATASET == "straightaway":
        x_centroid_2d = []
        y_centroid_2d = []
        for i in range(len(vals)):
            x_centroid_2d.append((vals[i]['bottom_left_front'][0] + vals[i]['bottom_right_back'][0]) / 2)
            y_centroid_2d.append((vals[i]['bottom_left_front'][1] + vals[i]['bottom_right_back'][1]) / 2)
    else:
        pass

    # print(vals)
    


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


