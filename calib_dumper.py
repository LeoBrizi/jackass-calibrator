import os
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import json
from motion_model import DDBodyFrameModel


class Ros2Reader:
    def __init__(self, data_dir: Path, *args, **kwargs):
        """
        :param data_dir: Directory containing rosbags or path to a rosbag file
        :param topics: Topic to read
        :param args:
        :param kwargs:
        """
        topic = kwargs.pop('topic')
        try:
            from rosbags.highlevel import AnyReader
        except ModuleNotFoundError:
            print("Rosbags library not installed, run 'pip install -U rosbags'")
            sys.exit(-1)

        self.bag = AnyReader([data_dir])

        self.bag.open()
        connection = self.bag.connections

        if not topic:
            raise Exception("You have to specify a topic")

        print("Reading the following topic: ", topic)
        connection = [x for x in self.bag.connections if x.topic == topic]
        self.msgs = self.bag.messages(connections=connection)

        self.topic = topic
        self.num_messages = self.bag.topics[topic].msgcount

    def __len__(self):
        return self.num_messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "bag"):
            self.bag.close()

    def __getitem__(self, item) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        connection, timestamp, rawdata = next(self.msgs)
        msg = self.bag.deserialize(rawdata, connection.msgtype)
        joint_msg_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        joints_names = msg.name
        joints_positions = msg.position
        joints_velocities = msg.velocity
        return joint_msg_stamp, joints_names, joints_positions, joints_velocities

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
    joints_topics = config.get("joints_topic", "")
    joints_name_order = config.get("joints_name_order", [])

    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        sys.exit(1)

    lut = {name: idx for idx, name in enumerate(joints_name_order)}

    kinematic_parameters_ig = config.get("kinematic_params", [])

    motion_model = DDBodyFrameModel(kinematic_parameters_ig)

    # open the file
    output_file_path = Path(config.get("dump_file", "calib_dump.txt"))
    if output_file_path.exists():
        print(f"Output file {output_file_path} already exists. It will be overwritten.")
    else:
        print(f"Output file {output_file_path} will be created.")
    output_file_path.touch()
    output_file = open(output_file_path, "w")

    prev_timestamp = 0
    with Ros2Reader(data_dir, topic=joints_topics) as reader:
        for i in range(len(reader)):
            timestamp, names, positions, velocities = reader[i]
            print(f"Timestamp: {timestamp}, Names: {names}, Positions: {positions}, Velocities: {velocities}")
            joint_velocities = np.array([velocities[lut[name]] for name in names])
            print(f"Joint Velocities: {joint_velocities}")

            v_l = joint_velocities[0] + joint_velocities[1]
            v_r = joint_velocities[2] + joint_velocities[3]

            v_l = v_l / 2.0
            v_r = v_r / 2.0

            if i == 0:
                prev_timestamp = timestamp
                continue
            
            dt = timestamp - prev_timestamp
            motion_model.getPose(v_r, v_l, dt)

            #dummp on a file
            output_file.write(f"{timestamp} {v_r} {v_l} {motion_model.state[0]} {motion_model.state[1]} {motion_model.state[2]}\n")

            motion_model.state

            prev_timestamp = timestamp

if __name__ == "__main__":
    main()