# Jackass Calibrator

Kinematic calibration toolkit for differential drive robots. Calibrates wheel odometry parameters using lidar pose estimates as ground truth.

## Overview

This package provides tools to:
1. Extract encoder (and optionally IMU) data from ROS2 bags
2. Associate odometry with lidar-based pose estimates
3. Optimize kinematic parameters using Levenberg-Marquardt
4. Publish calibrated odometry in real-time

## Installation

```bash
pip install numpy scipy rosbags tqdm
# For real-time odometry publishing:
pip install rclpy
```

## Workflow

### 1. Dump Encoder Data

Extract wheel velocities from a ROS2 bag and integrate using an initial guess:

```bash
# Without IMU
python calib_dumper.py config.json

# With IMU (recommended for better heading estimation)
python calib_dumper_imu.py config.json
```

Outputs: `calib_dump.txt`

### 2. Associate with Lidar Estimates

Match encoder timestamps with lidar pose estimates:

```bash
python calib_associator.py config.json
```

Outputs: `calib_dump_associations.txt`

### 3. Run Calibration

Optimize kinematic parameters:

```bash
python calib_solver.py config.json
```

Outputs optimized parameters and a trajectory comparison plot.

### 4. Publish Odometry (Optional)

Run calibrated odometry in real-time:

```bash
python odom_publisher.py config.json
```

## Configuration

Example `config.json`:

```json
{
  "motion_model": "DDBodyFrame",
  "kinematic_params": [0.09, 0.09, 0.5],
  "lidar_offset": [0.0, 0.0, 0.0],

  "bag_path": "/path/to/rosbag",
  "joints_topic": "/platform/joint_states",
  "joints_name_order": [
    "rear_left_wheel_joint",
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_right_wheel_joint"
  ],
  "imu_topic": "/sensors/imu_0/data",

  "dump_file": "calib_dump.txt",
  "laser_estimate": "/path/to/lidar_poses.txt",

  "iterations": 40,
  "tolerance": 1e-8,
  "epsilon": 1e-5,
  "dumping": 1,
  "min_movement": 1,
  "min_angle": 0.2,
  "params_mask": [1, 1, 1, 1, 1, 1],

  "odom_topic": "jackass_odom",
  "odom_frame": "odom",
  "base_frame": "base_link",
  "tf_prefix": "jackass",
  "publish_tf": true
}
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `motion_model` | Model type: `DDBodyFrame`, `DDBodyFrameExact`, `DDGyro`, `SKS`, `DDAdaptiveBaseline` |
| `kinematic_params` | Initial guess for kinematic parameters (model-dependent) |
| `lidar_offset` | Lidar position offset from base_link `[x, y, theta]` |
| `params_mask` | Binary mask to fix/optimize specific parameters |
| `min_movement` | Minimum translation (m) between keyframes |
| `min_angle` | Minimum rotation (rad) between keyframes |
| `dumping` | Levenberg-Marquardt damping factor |

## Motion Models

| Model | Parameters | Description |
|-------|------------|-------------|
| `DDBodyFrame` | `[k_r, k_l, baseline]` | Standard differential drive |
| `DDBodyFrameExact` | `[k_r, k_l, baseline]` | Exact arc integration |
| `DDGyro` | `[k_r, k_l, imu_offset, imu_shift]` | Differential drive with IMU heading |
| `SKS` | `[k_r, k_l, baseline, alpha]` | Skid-steer with lateral slip |
| `DDAdaptiveBaseline` | `[k_r, k_l, b0, alpha]` | Velocity-dependent baseline |

Where:
- `k_r`, `k_l`: Right/left wheel radius multipliers
- `baseline`: Effective wheel track width
- `imu_offset`, `imu_shift`: IMU yaw scale and bias
- `alpha`: Slip/skid coefficient

## Lidar Estimate Format

The `laser_estimate` file should contain poses as 3x3 transformation matrices, one row per line:

```
timestamp r11 r12 r13 r21 r22 r23 r31 r32 r33
```

Where columns 4, 8 are x, y and arctan2(r21, r11) gives theta.

## License

MIT
