import os
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import json

from calib_dumper import Ros2Reader
from geometry import yaw_from_quaternion


class ClearpathOdomDumper(Ros2Reader):

    def __getitem__(self, item) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        connection, timestamp, rawdata = next(self.msgs)
        msg = self.bag.deserialize(rawdata, connection.msgtype)
        msg_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = yaw_from_quaternion(q)
        return msg_stamp, x, y, theta


def main():
    if len(sys.argv) < 2:
        print("Usage: python clearpath_odom_dumper.py config_file")
        sys.exit(1)

    config_file = Path(sys.argv[1])

    if not config_file.exists():
        print(f"Config file {config_file} does not exist.")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    data_dir = Path(config.get("bag_path", ""))

    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        sys.exit(1)

    raw_file_path = Path(config.get("placeholder", "clearpath_dump.txt"))
    raw_file_path.touch()
    raw_file = open(raw_file_path, "w")

    with ClearpathOdomDumper(data_dir, topic="/j100_0819/platform/odom") as reader:
        for i in range(len(reader)):
            timestamp, x, y, th = reader[i]
            print(f"Timestamp: {timestamp}, x: {x}, y: {y}, theta: {th}")
            raw_file.write(f"{timestamp} {x} {y} {th}\n")

    filtered_file_path = Path(config.get("placeholder", "clearpath_filtered_dump.txt"))
    filtered_file_path.touch()
    filtered_file = open(filtered_file_path, "w")

    with ClearpathOdomDumper(
        data_dir, topic="/j100_0819/platform/odom/filtered"
    ) as reader:
        for i in range(len(reader)):
            timestamp, x, y, th = reader[i]
            print(f"Timestamp: {timestamp}, x: {x}, y: {y}, theta: {th}")
            filtered_file.write(f"{timestamp} {x} {y} {th}\n")


if __name__ == "__main__":
    main()
else:
    print(__name__)
    print("This script is intended to be run as a standalone program.")
    print("Use 'python clearpath_odom_dumper.py <config.json>' to execute it.")
    sys.exit(1)
