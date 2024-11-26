import os
import json
import numpy as np
import pandas as pd


def calibration(xy, image_width, image_height, H_inv):
    # print(xy)
    x = xy[0]
    y = xy[1]

    # De-normalize pixel values
    pixel_x = x * image_width
    pixel_y = y * image_height
    pixel_coord = np.array([pixel_x, pixel_y, 1]).reshape(3, 1)
    

    # Apply inverse homography
    world_coord = np.matmul(H_inv, pixel_coord)
    # Normalize so that z value is 1 (we don't care about z value)
    world_coord /= world_coord[2, 0]
    
    return [world_coord[0][0], world_coord[1][0]]

def load_dataset(dataset_path, object_id):
    # Ensure the dataset path ends with a backslash
    if not dataset_path.endswith("\\"):
        dataset_path += "\\"

    # Split dataset path into list of directories 
    dataset = dataset_path.split("\\")[1]

    # Highway scenario: obtain ground truth Bounding Box vals for object_id
    if dataset == "straightaway":
        box3d_values_list = obtain_bb_vals(dataset_path, object_id)
        return box3d_values_list

    # Turning scenario. Obtain ground truth trajectories
    elif dataset == "r2_s1":
        traj = obtain_trajectory_r2s1(dataset_path, object_id)
        # print(traj)
        return traj
    
    # Straight line LIDAR scenario:
    elif dataset == "r0_s4_lidar":
        # traj = obtain_trajectory_r0s4(dataset_path, object_id)
        traj = None
        print("This dataset sucks.")
        return traj
    
    else:
        print("Unrecognized Dataset")
        



def find_calibration_file(dataset_path):
    # Replace single backslashes with double backslashes
    # dataset_path = dataset_path.replace("\\", "\\\\")
    
    # Split the dataset path into parts
    path_parts = dataset_path.split("\\")
    
    # Check if the path has at least three parts to avoid index errors
    if len(path_parts) < 3:
        raise ValueError("Dataset path must have at least three directories after the base.")

    # Extract the second directory and the final directory
    second_dir = path_parts[1]
    final_dir = path_parts[-2] + ".json"
    
    # Construct the calibration file path
    base_path = path_parts[0]  # Base path (e.g., "a9_dataset")
    calibration_path = os.path.join(base_path, second_dir, "_calibration", final_dir)
    
    # Add double backslashes for consistency
    # calibration_path = calibration_path.replace("\\", "\\\\")
    
    return calibration_path



def obtain_calibration_data(dataset_path):
    ''' Obtain Calibration values'''
    calibration_file = find_calibration_file(dataset_path)
    with open(calibration_file, 'r') as file:
        calibration_data = json.load(file)
    
    # Extract data
    image_height = calibration_data['image_height']
    image_width = calibration_data['image_width']
    intrinsic_matrix = np.array(calibration_data['intrinsic_matrix'])
    extrinsic_matrix = np.array(calibration_data['extrinsic_matrix'])
    projection_matrix = np.array(calibration_data['projection_matrix'])

    # P = np.vstack((projection_matrix, np.array([0, 0, 0, 1])))
    # print(P)

    # Contruct Inverse Homography matrix
    # Inverse homography turns u,v pixel values into x, y real-world values.
    r1 = extrinsic_matrix[:, 0].reshape(3, 1)
    r2 = extrinsic_matrix[:, 1].reshape(3, 1)
    t = extrinsic_matrix[:, -1].reshape(3, 1)


    H = np.matmul(intrinsic_matrix, np.column_stack((r1, r2, t)))

    H_inv = np.linalg.inv(H)
    # H_inv = np.linalg.pinv(H)
    # print(np.linalg.cond(H), np.linalg.cond(H_inv))


    #Testing Calibration
    # sample_pixel = [0.747674, 0.504582]
    # worldx, worldy = calibration(sample_pixel, image_width, image_height, H_inv)
    # print(f"pixel val = {sample_pixel} \n world coord = {worldx, worldy}")

    return image_width, image_height, H_inv


def obtain_bb_vals(dataset_path, object_id):
    # List to store box3d_projected values
    box3d_values_list = []

    image_width, image_height, H_inv = obtain_calibration_data(dataset_path)

    # Walk through all files in the dataset directory
    '''straightaway dataset: contains corners of bounding box.'''
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                # Open and parse the JSON file
                # try:
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    
                    # Loop through labels to find the object ID
                    for label in data.get("labels", []):
                        if label.get("id") == object_id:
                            box3d_projected = label.get("box3d_projected", {})

                            ## Only calibrate bottom_left_front and bottom_right_back for now
                            box3d_projected['bottom_left_front'] = calibration(box3d_projected['bottom_left_front'], image_width, image_height, H_inv)
                            box3d_projected['bottom_left_back'] = calibration(box3d_projected['bottom_left_back'], image_width, image_height, H_inv)
                            box3d_projected['bottom_right_back'] = calibration(box3d_projected['bottom_right_back'], image_width, image_height, H_inv)
                            box3d_projected['bottom_right_front'] = calibration(box3d_projected['bottom_right_front'], image_width, image_height, H_inv)
                            box3d_projected['top_left_front'] = calibration(box3d_projected['top_left_front'], image_width, image_height, H_inv)
                            box3d_projected['top_left_back'] = calibration(box3d_projected['top_left_back'], image_width, image_height, H_inv)
                            box3d_projected['top_right_back'] = calibration(box3d_projected['top_right_back'], image_width, image_height, H_inv)
                            box3d_projected['top_right_front'] = calibration(box3d_projected['top_right_front'], image_width, image_height, H_inv)

                            # box3d_projected['bottom_left_front'] = gpt_calibration(box3d_projected['bottom_left_front'], image_width, image_height, projection_matrix)
                            # box3d_projected['bottom_left_back'] = gpt_calibration(box3d_projected['bottom_left_back'], image_width, image_height, projection_matrix)
                            # box3d_projected['bottom_right_back'] = gpt_calibration(box3d_projected['bottom_right_back'], image_width, image_height, projection_matrix)
                            # box3d_projected['bottom_right_front'] = gpt_calibration(box3d_projected['bottom_right_front'], image_width, image_height, projection_matrix)
                            # box3d_projected['top_left_front'] = gpt_calibration(box3d_projected['top_left_front'], image_width, image_height, projection_matrix)
                            # box3d_projected['top_left_back'] = gpt_calibration(box3d_projected['top_left_back'], image_width, image_height, projection_matrix)
                            # box3d_projected['top_right_back'] = gpt_calibration(box3d_projected['top_right_back'], image_width, image_height, projection_matrix)
                            # box3d_projected['top_right_front'] = gpt_calibration(box3d_projected['top_right_front'], image_width, image_height, projection_matrix)
                            
                            box3d_values_list.append(box3d_projected)
                # except Exception as e:
                #     print(f"Error processing file {file_path}: {e}")
    return box3d_values_list


def obtain_trajectory_r2s1(dataset_path, object_id):
    trajectory_data = []

    # Iterate through each file in the directory
    for filename in sorted(os.listdir(dataset_path)):
        if filename.endswith(".json"):
            filepath = os.path.join(dataset_path, filename)
            try:
                # Load JSON file
                with open(filepath, 'r') as file:
                    data = json.load(file)
                
                # Extract timestamps from the filename
                parts = filename.split("_")
                epoch_seconds = int(parts[0])
                epoch_nanoseconds = int(parts[1])
                timestamp = epoch_seconds + epoch_nanoseconds * 1e-9

                # Look for the specified object
                frames = data.get("openlabel", {}).get("frames", {})
    
                for frame_id, frame_data in frames.items():
                    objects = frame_data.get("objects", {})
                    if object_id in objects:
                        # Extract the cuboid 'val' field for the given object
                        cuboid_val = objects[object_id].get("object_data", {}).get("cuboid", {}).get("val", [])
                        if cuboid_val:
                            # Append to trajectory data
                            trajectory_data.append({
                                "timestamp": timestamp,
                                "cuboid_val": cuboid_val
                            })
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(trajectory_data)
    return df


def obtain_trajectory_r0s4(dataset_path, label_id):
    data_list = []

    # Iterate through each JSON file in the directory
    for filename in sorted(os.listdir(dataset_path)):
        if filename.endswith(".json"):
            filepath = os.path.join(dataset_path, filename)
            try:
                # Load JSON file
                with open(filepath, "r") as file:
                    data = json.load(file)

                # Extract timestamps
                timestamp_secs = data.get("timestamp_secs", 0)
                timestamp_nsecs = data.get("timestamp_nsecs", 0)
                timestamp = timestamp_secs + timestamp_nsecs * 1e-9

                # Search for the specified label ID
                labels = data.get("labels", [])
                for label in labels:
                    if label.get("id") == label_id:
                        # Extract box3d data
                        box3d = label.get("box3d", {})
                        dimension = box3d.get("dimension", {})
                        location = box3d.get("location", {})
                        orientation = box3d.get("orientation", {})

                        # Append the extracted data
                        data_list.append({
                            "timestamp": timestamp,
                            "dimension_length": dimension.get("length"),
                            "dimension_width": dimension.get("width"),
                            "dimension_height": dimension.get("height"),
                            "location_x": location.get("x"),
                            "location_y": location.get("y"),
                            "location_z": location.get("z"),
                            "orientation_yaw": orientation.get("rotationYaw"),
                            "orientation_pitch": orientation.get("rotationPitch"),
                            "orientation_roll": orientation.get("rotationRoll")
                        })
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Convert to a DataFrame
    df = pd.DataFrame(data_list)
    return df

def compute_ground_truth(vals):

    x = vals['x']
    y = vals['y']
    yaw = vals['yaw']
    ts = vals['timestamp']

    lat_vel = np.diff(x)/np.diff(ts)
    long_vel = np.diff(y)/np.diff(ts)
    lat_vel = np.insert(lat_vel, 0, None)
    long_vel = np.insert(long_vel, 0, None)

    lat_accel = np.diff(lat_vel)/np.diff(ts)
    long_accel = np.diff(long_vel)/np.diff(ts)
    lat_accel = np.insert(lat_accel, 0, None)
    long_accel = np.insert(long_accel, 0, None)

    # Compute Yaw Rate
    yaw_rate = np.diff(yaw)/np.diff(ts)
    yaw_rate = np.insert(yaw_rate, 0, None)

    vals['lat_vel'] = lat_vel
    vals['long_vel'] = long_vel
    vals['lat_accel'] = lat_accel
    vals['long_accel'] = long_accel
    vals['yaw_rate'] = yaw_rate


    return None