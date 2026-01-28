import argparse
import os
import subprocess

import cv2
import pandas as pd
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.mps import read_closed_loop_trajectory
from projectaria_tools.core.mps.utils import get_nearest_pose
from projectaria_tools.core.image import InterpolationMethod
import tqdm



def extract_video_timestamps(vrs_path):
    provider = data_provider.create_vrs_data_provider(vrs_path)
    if provider is None:
        raise RuntimeError(f"Could not open VRS file: {vrs_path}")
    
    stream_id = provider.get_stream_id_from_label("camera-rgb")
    
    number_of_frames = provider.get_num_data(stream_id)
    
    timestamps = []
    
    for i in range(number_of_frames):
        _, frame = provider.get_image_data_by_index(stream_id, i)
        timestamps.append(frame.capture_timestamp_ns)
        
    return timestamps
    

def match_poses(ts_ns_list, vrs_path, trajectory_path, output_path):
    provider = data_provider.create_vrs_data_provider(vrs_path)
    device_calibration = provider.get_device_calibration()
    camera_calibration = device_calibration.get_camera_calib('camera-rgb')
    T_device_cam = camera_calibration.get_transform_device_camera()
    
    closed_loop = read_closed_loop_trajectory(trajectory_path)
    
    matched_poses = []
    
    print("matching poses")
    for ts in tqdm.tqdm(ts_ns_list):
        pose = get_nearest_pose(closed_loop, ts)
        if pose is None:
            # skip if no nearby pose found
            continue
        T_world_device = pose.transform_world_device
        T_world_cam = T_world_device @ T_device_cam
        
        T_world_cam = T_world_cam.to_quat_and_translation()[0]
        
        new_pose = dict()
        
        new_pose["tracking_timestamp_us"] = ts
        new_pose['qw_world_device'] = T_world_cam[0]
        new_pose['qx_world_device'] = T_world_cam[1]
        new_pose['qy_world_device'] = T_world_cam[2]
        new_pose['qz_world_device'] = T_world_cam[3]
        new_pose['tx_world_device'] = T_world_cam[4]
        new_pose['ty_world_device'] = T_world_cam[5]
        new_pose['tz_world_device'] = T_world_cam[6]
        
        matched_poses.append(new_pose)
    print(f"matched {len(matched_poses)} poses")
    
    # write trajectory to file
    df = pd.DataFrame(matched_poses)
    df.to_csv(os.path.join(output_path, "trajectory.csv"), index=False)
    
    return matched_poses

def calibrate_video(vrs_path, matched_poses, output_path):
    provider = data_provider.create_vrs_data_provider(vrs_path)
    provider.set_color_correction(True)
    stream_id = provider.get_stream_id_from_label("camera-rgb")
    number_of_frames = provider.get_num_data(stream_id)
    
    device_calibration = provider.get_device_calibration()
    camera_calibration = device_calibration.get_camera_calib("camera-rgb")
    
    linear_calibration = calibration.get_linear_camera_calibration(1408, 1408, 610, "camera-rgb")
    
    first_timestamp = matched_poses[0]["tracking_timestamp_us"]
    last_timestamp = matched_poses[-1]["tracking_timestamp_us"]
    
    
    video_path = os.path.join(output_path, "main.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps    = round(provider.get_nominal_rate_hz(stream_id))
    width, height = 1408, 1408
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=True)
    
    print("writing calibrated video")
    for i in tqdm.tqdm(range(number_of_frames)):
        data = provider.get_image_data_by_index(stream_id, i)
        frame_timestamp = data[1].capture_timestamp_ns
        image = data[0].to_numpy_array()
        
        if frame_timestamp<first_timestamp or frame_timestamp>last_timestamp:
            continue
        
        image = calibration.distort_by_calibration(image, linear_calibration, camera_calibration, InterpolationMethod.BILINEAR)
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        bgr = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        
    writer.release()
    print("finished writing calibrated video")
    # create web compatible mp4, the output from above doesn not work in (some?) browsers. 
    # requires ffmpeg
    print("creating web compatible mp4")
    video_web_path = os.path.join(output_path, "main_web.mp4")
    command = f"ffmpeg -y -i {video_path} -c:v libx264 -preset fast -an -movflags +faststart {video_web_path}"
    process = subprocess.run(command, shell=True)
    
    if process.returncode == 0:
        print("wrote web compatible mp4")
    else:
        raise RuntimeError("problem writing web compatible mp4")
    


def get_files(path):
    vrs_path = os.path.join(path, "main.vrs")
    trajectory_path = os.path.join(path, "mps", "slam", "closed_loop_trajectory.csv")
    if os.path.exists(vrs_path) and os.path.exists(trajectory_path):
        return vrs_path, trajectory_path
    else:
        raise RuntimeError("recording or trajectory file not found")

def main():
    """
        Given a aria recording and mps data in the format 
        
        recording/ \n
        -> main.vrs \n
        -> mps/slam/closed_loop_trajectory.csv \n
        
        creates calibrated trajectory and videos under 
        
        recording/ \n
        -> calibrated/
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording", 
        type=str, 
        help="aria recording folder with mps output",
        required=True,
        )
    
    args = parser.parse_args()
    input_path = args.recording
    calibrate(input_path=input_path)
    
    
    
def calibrate(input_path, output_path):
    if not input_path:
        raise RuntimeError("You didn't specify an input folder")
    
    if not output_path:
        output_path = os.path.join(input_path, "calibrated")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    # get paths
    vrs_path, trajectory_path = get_files(input_path)
    
    # get frame timestamps
    timestamps = extract_video_timestamps(vrs_path)
    
    # only keep relevant trajectory entries, apply calibration and write the file
    matched_poses = match_poses(timestamps, vrs_path, trajectory_path, output_path)
    
    # calibrate recording to linear camera, cut it to match trajectories
    # and generate mp4 and web compatible mp4
    calibrate_video(vrs_path, matched_poses, output_path)
    
    
    

if __name__ == '__main__':
    main()