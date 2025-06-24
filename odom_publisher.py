#!/usr/bin/env python3

import sys
import json
from pathlib import Path
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist, TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time
from motion_model import DDBodyFrameModel
import numpy as np


class OdometryPublisher(Node):
    def __init__(self, joint_topics, kinematic_parameters, lut, odom_topic, odom_frame, base_frame, tf_prefix, publish_tf):
        super().__init__('odometry_publisher')
        self.publisher_ = self.create_publisher(Odometry, odom_topic, 10)
        self.subscriber_ = self.create_subscription(JointState, joint_topics, self.joint_state_callback, 10)
        self.lut = lut
        self.motion_model = DDBodyFrameModel(kinematic_parameters)
        self.prev_timestamp = 0.0
        self.publish_tf = publish_tf

        if publish_tf:
            import tf2_ros
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.tf_prefix = tf_prefix
        self.odom_frame = f"{tf_prefix}/{odom_frame}"
        self.base_frame = f"{tf_prefix}/{base_frame}"


    def joint_state_callback(self, msg):
        joint_msg_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        joints_names = msg.name
        joints_positions = msg.position
        joints_velocities = msg.velocity

        print(f"Timestamp: {joint_msg_stamp}, Names: {joints_names}, Positions: {joints_positions}, Velocities: {joints_velocities}")
        joint_velocities = np.array([joints_velocities[self.lut[name]] for name in joints_names])
        print(f"Joint Velocities: {joint_velocities}")

        v_l = joint_velocities[0] + joint_velocities[1]
        v_r = joint_velocities[2] + joint_velocities[3]

        v_l = v_l / 2.0
        v_r = v_r / 2.0

        if self.prev_timestamp == 0.0:
            self.prev_timestamp = joint_msg_stamp
            return

        dt = joint_msg_stamp - self.prev_timestamp
        self.prev_timestamp = joint_msg_stamp
        self.motion_model.getPose([v_r, v_l], dt)
        odom_state = self.motion_model.getState()

        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame

        odom_msg.pose.pose.position = Point(x=odom_state[0], y=odom_state[1], z=0.0)
        odom_msg.pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(odom_state[2] / 2.0),
            w=math.cos(odom_state[2] / 2.0)
        )

        odom_msg.twist.twist.linear.x = self.motion_model.v * math.cos(odom_state[2])
        odom_msg.twist.twist.linear.y = self.motion_model.v * math.sin(odom_state[2])
        odom_msg.twist.twist.linear.z = 0.0

        odom_msg.twist.twist.angular.z = self.motion_model.omega

        self.publisher_.publish(odom_msg)
        self.get_logger().info(f'Publishing odometry: x={odom_state[0]:.2f}, y={odom_state[1]:.2f}, theta={odom_state[2]:.2f}')

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = msg.header.stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = odom_state[0]
            t.transform.translation.y = odom_state[1]
            t.transform.translation.z = 0.0
            t.transform.rotation = odom_msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)



def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: python odom_publisher.py config_file")
        sys.exit(1)

    config_file = Path(sys.argv[1])

    if not config_file.exists():
        print(f"Config file {config_file} does not exist.")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    joint_topics = config.get("joints_topic", "")
    if not joint_topics:
        print("No joint topics specified in the config file.")
        sys.exit(1) 
    
    time_period = config.get("time_period", 0.1)
    if time_period <= 0:
        print("Invalid time period specified in the config file. Must be greater than 0.")
        sys.exit(1) 

    kinematic_parameters = config.get("kinematic_params", [])
    if not kinematic_parameters or len(kinematic_parameters) < 3:
        print("Invalid kinematic parameters specified in the config file. Must contain at least 3 values.")
        sys.exit(1)
    
    joints_name_order = config.get("joints_name_order", [])
    lut = {name: idx for idx, name in enumerate(joints_name_order)}

    if not lut:
        print("No joint names specified in the config file.")
        sys.exit(1)

    odom_topic = config.get("odom_topic", "jackass_odom")
    odom_frame = config.get("odom_frame", "odom")
    base_frame = config.get("base_frame", "base_link")
    tf_prefix = config.get("tf_prefix", "jackass")
    publish_tf = config.get("publish_tf", True)


    rclpy.init(args=args)
    node = OdometryPublisher(joint_topics, kinematic_parameters, lut, odom_topic, odom_frame, base_frame, tf_prefix, publish_tf)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()