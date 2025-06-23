
import os
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python calib_dumper.py config_file")
        sys.exit(1)

    config_file = Path(sys.argv[1])

    if not config_file.exists():
        print(f"Config file {config_file} does not exist.")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    motion_model_integration_file = Path(config.get("dump_file", ""))
    if not motion_model_integration_file:
        print("No dump file specified in the config. Make sure to run calib_dumper.py first")
        sys.exit(1)

    lidar_estimate_file = Path(config.get("laser_estimate", ""))
    if not lidar_estimate_file:
        print("No laser estimate specified in the config")
        sys.exit(1)

    motion_model_file = open(motion_model_integration_file, "r")
    lidar_estimate_file = open(lidar_estimate_file, "r")
    min_time_diff = config.get("min_time_diff", 0.025)

    lidar_estimate_data = lidar_estimate_file.readlines()

    #construct a dictionary to hold lidar estimates
    lidar_estimates = {}
    for line in lidar_estimate_data:
        data = line.strip().split()
        if len(data) < 4:
            continue
        timestamp = float(data[0]) * 1e-9
        x = float(data[4])
        y = float(data[8])
        cos_theta = float(data[1])
        sin_theta = float(data[5])
        theta = np.arctan2(sin_theta, cos_theta)
        lidar_estimates[timestamp] = (x, y, theta)

    output_file = motion_model_integration_file.stem + "_associations.txt"
    out_file = open(output_file, "w")
    lidar_ts = list(lidar_estimates.keys())

    # associate time stamps

    # loop over  motion model file lines
    for line in motion_model_file:
        line = line.strip()
        motion_model_data = line.split()

        motion_timestamp = float(motion_model_data[0])
        v_r = float(motion_model_data[1])
        v_l = float(motion_model_data[2])
        motion_x = float(motion_model_data[3])
        motion_y = float(motion_model_data[4])
        motion_theta = float(motion_model_data[5])

        # find the corresponding lidar estimate
        closest_lidar_ts = min(lidar_ts, key=lambda ts: abs(ts - motion_timestamp))
        if(abs(closest_lidar_ts - motion_timestamp) > min_time_diff):  # threshold for matching timestamps
            # out_file.write(f"{motion_timestamp} {v_r} {v_l} {motion_x} {motion_y} {motion_theta}\n")
            out_file.write(f"{line}\n")
            continue

        if closest_lidar_ts in lidar_estimates:
            lidar_x, lidar_y, lidar_theta = lidar_estimates[closest_lidar_ts]
            print(f"Found matching timestamps: {motion_timestamp}")
            print(f"Motion Model: {motion_x}, {motion_y}, {motion_theta}")
            print(f"Lidar Estimate: {lidar_x}, {lidar_y}, {lidar_theta}")
            # out_file.write(f"{motion_timestamp} {v_r} {v_l} {motion_x} {motion_y} {motion_theta} "
                        #    f"{lidar_x} {lidar_y} {lidar_theta} {closest_lidar_ts}\n")
            out_file.write(f"{line} {lidar_x} {lidar_y} {lidar_theta} {closest_lidar_ts}\n")

if __name__ == "__main__":
    main()
else:
    print("This script is intended to be run as a standalone program.")
    print("Use 'python calib_associator.py <config_file>' to execute it.")
    sys.exit(1)