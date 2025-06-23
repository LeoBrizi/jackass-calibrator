import os
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import json
from motion_model import DDBodyFrameModel, DDGyroModel
from bisect import bisect_left
from tqdm import tqdm

import geometry


class Ros2Reader:
    def __init__(self, data_dir: Path, *args, **kwargs):
        """
        :param data_dir: Directory containing rosbags or path to a rosbag file
        :param topics: Topic to read
        :param args:
        :param kwargs:
        """
        joint_topic = kwargs.pop("joint_topic")
        imu_topic = kwargs.pop("imu_topic")
        try:
            from rosbags.highlevel import AnyReader
        except ModuleNotFoundError:
            print("Rosbags library not installed, run 'pip install -U rosbags'")
            sys.exit(-1)

        self.bag = AnyReader([data_dir])

        self.bag.open()
        connection = self.bag.connections

        if not joint_topic or not imu_topic:
            raise Exception("You have to specify both joint and imu topics")

        topics = [joint_topic, imu_topic]
        print("Reading the following topic: ", topics)
        connection = [x for x in self.bag.connections if x.topic in topics]
        self.msgs = self.bag.messages(connections=connection)

        self.topics = topics

        # In __init__ (after self.bag.open())
        self.imu_messages = []
        self.joint_messages = []

        for conn, timestamp, rawdata in tqdm(self.bag.messages()):
            if conn.topic == joint_topic:
                self.joint_messages.append((conn, timestamp, rawdata))
            elif conn.topic == imu_topic:
                self.imu_messages.append((conn, timestamp, rawdata))

        # Extract IMU timestamps for fast search
        self.imu_timestamps = [t for _, t, _ in self.imu_messages]

        self.num_messages = len(self.joint_messages)
        print(
            f"Found {len(self.joint_messages)} joint messages and {len(self.imu_messages)} IMU messages."
        )

    def __len__(self):
        return self.num_messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "bag"):
            self.bag.close()

    def __getitem__(self, item) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        # connection, timestamp, rawdata = next(self.msgs)
        # msg = self.bag.deserialize(rawdata, connection.msgtype)
        # joint_msg_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # joints_names = msg.name
        # joints_positions = msg.position
        # joints_velocities = msg.velocity
        # return joint_msg_stamp, joints_names, joints_positions, joints_velocities
        conn, joint_ts, rawdata = self.joint_messages[item]
        joint_msg = self.bag.deserialize(rawdata, conn.msgtype)

        joint_stamp = joint_msg.header.stamp.sec + joint_msg.header.stamp.nanosec * 1e-9
        joint_names = joint_msg.name
        joint_positions = np.array(joint_msg.position)
        joint_velocities = np.array(joint_msg.velocity)

        # Find closest IMU message by timestamp
        i = bisect_left(self.imu_timestamps, joint_ts) + 1
        if i == len(self.imu_timestamps):
            i -= 1  # Use the last one if we're past the end
        print(
            f"Joint timestamp: {joint_ts}, IMU index: {i}, IMU timestamp: {self.imu_timestamps[i]}, difference: {(joint_ts - self.imu_timestamps[i])/1e9}"
        )
        imu_conn, imu_ts, imu_raw = self.imu_messages[i]
        imu_msg = self.bag.deserialize(imu_raw, imu_conn.msgtype)
        imu_q = imu_msg.orientation
        imu_angular_velocity = imu_msg.angular_velocity.z

        yaw = geometry.yaw_from_quaternion(imu_q)

        return joint_stamp, joint_names, joint_velocities, yaw, imu_angular_velocity


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

    data_dir = Path(config.get("bag_path", ""))
    joints_topic = config.get("joints_topic", "")
    imu_topic = config.get("imu_topic", "")
    joints_name_order = config.get("joints_name_order", [])

    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        sys.exit(1)

    lut = {name: idx for idx, name in enumerate(joints_name_order)}

    kinematic_parameters_ig = config.get("kinematic_params", [])

    motion_model = DDGyroModel(kinematic_parameters_ig)

    # open the file
    output_file_path = Path(config.get("dump_file", "calib_dump.txt"))
    if output_file_path.exists():
        print(f"Output file {output_file_path} already exists. It will be overwritten.")
    else:
        print(f"Output file {output_file_path} will be created.")
    output_file_path.touch()
    output_file = open(output_file_path, "w")

    prev_timestamp = 0
    with Ros2Reader(data_dir, joint_topic=joints_topic, imu_topic=imu_topic) as reader:
        for i in range(len(reader)):
            # timestamp, names, positions, velocities = reader[i]
            joint_stamp, joint_names, j_velocities, imu_theta, imu_angular_velocity = (
                reader[i]
            )
            print(
                f"Timestamp: {joint_stamp}, Names: {joint_names}, Velocities: {j_velocities}"
            )
            joint_velocities = np.array(
                [j_velocities[lut[name]] for name in joint_names]
            )
            print(f"Joint Velocities: {joint_velocities}")

            v_l = joint_velocities[0] + joint_velocities[1]
            v_r = joint_velocities[2] + joint_velocities[3]

            v_l = v_l / 2.0
            v_r = v_r / 2.0

            if i == 0:
                prev_timestamp = joint_stamp
                continue

            dt = joint_stamp - prev_timestamp
            motion_model.getPose(np.array([v_r, v_l, imu_theta]), dt)

            # dummp on a file
            output_file.write(
                f"{joint_stamp} {v_r} {v_l} {imu_theta} {motion_model.state[0]} {motion_model.state[1]} {motion_model.state[2]}\n"
            )

            motion_model.state

            prev_timestamp = joint_stamp


if __name__ == "__main__":
    main()
else:
    print("This script is intended to be run as a standalone program.")
    print("Use 'python calib_dumper.py <config.json>' to execute it.")
    sys.exit(1)
