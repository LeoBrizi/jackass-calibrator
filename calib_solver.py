import os
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import json

from geometry import T,R, exp, log
from motion_model import DDBodyFrameModel, DDGlobalFrameModel
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class CalibSolver:
    def __init__(self, kinematic_model, lidar_offset,  *args, **kwargs):
        self.kinematic_model = kinematic_model

        self.kin_param = self.kinematic_model.getParams().copy()
        self.lidar_offset = lidar_offset.copy()
        self.params = np.concatenate((self.kin_param, self.lidar_offset))

        self.iterations = kwargs.pop('iterations', 30)
        self.tolerance = kwargs.pop('tolerance', 1e-6)
        self.epsilon = kwargs.pop('epsilon', 1e-6)
        self.dumping = kwargs.pop('dumping', 10)
        self.min_movement = kwargs.pop('min_movement', 1)
        self.incs = np.eye(len(self.params), dtype=np.float64) * self.epsilon

        self.stats = {}

    # def error(self, params, vel_r, vel_l, dt, lidar_pose):
    #     kin_model_temp = self.kinematic_model.deepCopy()
    #     kin_model_temp.setParams(params[:3])

    #     kin_model_temp.getPose(vel_r, vel_l, dt)
    #     odom_state = kin_model_temp.getState()

    #     T_lidar_offset = exp(params[3], params[4], params[5])
    #     T_odom = exp(odom_state[0], odom_state[1], odom_state[2])
    #     T_lidar = exp(lidar_pose[1], lidar_pose[2], lidar_pose[3])

    #     residual = np.linalg.inv(T_lidar) @ np.linalg.inv(T_lidar_offset) @ T_odom @ T_lidar_offset

    #     return log(residual).reshape((3, 1))
    
    def error_(self, params, encoder_measurements, delta_pose_lidar):
        """
        This function computes the residuals between the expected and actual poses
        based on the encoder measurements and the lidar pose change.
        """
        kin_model_temp = self.kinematic_model.deepCopy()
        kin_model_temp.reset()
        kin_model_temp.setParams(params[:3])

        prev_timestamp = encoder_measurements[0][0]  # First timestamp from the encoder data

        for ts, vel_r, vel_l in encoder_measurements:
            dt = ts - prev_timestamp
            kin_model_temp.getPose(vel_r, vel_l, dt)
            prev_timestamp = ts

        odom_state = kin_model_temp.getState()

        T_lidar_offset = exp(params[3], params[4], params[5])
        T_odom = exp(odom_state[0], odom_state[1], odom_state[2])
        # T_lidar = exp(delta_pose_lidar[0], delta_pose_lidar[1], delta_pose_lidar[2])

        prediction = np.linalg.inv(T_lidar_offset) @ T_odom @ T_lidar_offset

        residual = np.linalg.inv(delta_pose_lidar) @ prediction

        return log(residual).reshape((3, 1))

    def errorAndJacobian_(self, encoder_data, lidar_data, ass_lidar):
        """
        This function computes the residuals between the expected and actual poses
        based on the encoder data and lidar associations.
        """

        """
      
        prev_timestamp = encoder_data[0][0]

        for index, (encoder_ts, vel_r, vel_l) in enumerate(encoder_data):
            dt = encoder_ts - prev_timestamp

            motion = np.linalg.norm(self.kinematic_model.getState()[0:1])

            if index not in ass_lidar or motion < self.min_movement:
                self.kinematic_model.getPose(vel_r, vel_l, dt)
                prev_timestamp = encoder_ts
                continue

            residual = self.error_(self.params, vel_r, vel_l, dt, lidar_data[ass_lidar[index]])
            residuals.append(residual)

            jacobian = np.zeros((3, len(self.params)))

            for p in range(len(self.params)):
                e_plus  = self.error_(self.params + self.incs[p,:], vel_r, vel_l, dt, lidar_data[ass_lidar[index]])
                e_minus = self.error_(self.params - self.incs[p,:], vel_r, vel_l, dt, lidar_data[ass_lidar[index]])
                jacobian[:, p] = ((e_plus - e_minus) / (2 * self.epsilon)).flatten()

            jacobians.append(jacobian)

            self.kinematic_model.getPose(vel_r, vel_l, dt)
            prev_timestamp = encoder_ts
            prev_pose = self.kinematic_model.getState()

        """

        residuals = []
        jacobians = []

        encoder_measurements = []
        prev_lidar_pose = lidar_data[0]
        prev_lidar_pose_index = 0
        for index in tqdm(range(len(lidar_data))):
            delta_pose_lidar = np.linalg.inv(exp(prev_lidar_pose[1], prev_lidar_pose[2], prev_lidar_pose[3])) @ exp(lidar_data[index][1], lidar_data[index][2], lidar_data[index][3])

            if np.linalg.norm(log(delta_pose_lidar)[0:2]) < self.min_movement:
                continue

            # collect all the encoder measurements from prev_lidar_pose_index to current_index

            encoder_measurements = encoder_data[ass_lidar[prev_lidar_pose_index]:ass_lidar[index]]

            residual = self.error_(self.params, encoder_measurements, delta_pose_lidar)

            residuals.append(residual)

            jacobian = np.zeros((3, len(self.params)))

            def compute_jacobian_col(p):
              e_plus  = self.error_(self.params + self.incs[p,:], encoder_measurements, delta_pose_lidar)
              e_minus = self.error_(self.params - self.incs[p,:], encoder_measurements, delta_pose_lidar)
              return ((e_plus - e_minus) / (2 * self.epsilon)).flatten()

            with ThreadPoolExecutor() as executor:
              results = list(executor.map(compute_jacobian_col, range(len(self.params))))

            for p, res in enumerate(results):
              jacobian[:, p] = res

            jacobians.append(jacobian)

            encoder_measurements = []
            prev_lidar_pose = lidar_data[index]
            prev_lidar_pose_index = index

        return residuals, jacobians

    def solve(self, encoder_data, lidar_data, ass_lidar):
        
        prev_cost = np.inf

        for iteration in range(self.iterations):

            residuals, jacobians = self.errorAndJacobian_(encoder_data, lidar_data, ass_lidar)

            # #plot odom and lidar poses
            # import matplotlib.pyplot as plt

            # plt.figure(figsize=(10, 5))
            # plt.title("Odometry vs. Lidar Poses")
            # plt.plot([pose[0] for pose in odom_pose], [pose[1] for pose in odom_pose], 'r', label='Odometry')
            # plt.plot([pose[0] for pose in lidar_pose], [pose[1] for pose in lidar_pose], 'b', label='Lidar')
            # plt.xlabel("X")
            # plt.ylabel("Y")
            # plt.legend()
            # plt.show()

            H = np.ones((len(self.params), len(self.params))) * self.dumping
            b = np.zeros((len(self.params), 1))

            chi = 0

            for i in range(len(residuals)):
                
                J = jacobians[i]
                H += J.T @ J
                b += J.T @ residuals[i]

                chi += residuals[i].T @ residuals[i]

            print("H det: ", np.linalg.det(H), " num of residuals: ", len(residuals))
            delta_params = np.linalg.solve(H, -b)
            
            self.params += delta_params.flatten()

            # wrap angles
            self.params[5] = (self.params[5] + np.pi) % (2 * np.pi) - np.pi

            delta_cost = abs(chi - prev_cost)

            normalized_chi = chi / len(residuals)

            self.stats[iteration] = [chi, normalized_chi, delta_params.flatten(), delta_cost]

            print(f"Iteration {iteration}: chi = {chi}, normalized_chi = {normalized_chi}, delta_params = {delta_params.flatten()}, delta_cost = {delta_cost}, params = {self.params}")

            if delta_cost < self.tolerance:
                print(f"Converged after {iteration} iterations")
                break
            prev_cost = chi
            self.kin_param = self.params[3:]
            self.lidar_offset = self.params[3:]

        return H

def main():
    if len(sys.argv) < 2:
        print("Usage: python calib_solver.py config_file")
        sys.exit(1)

    config_file = Path(sys.argv[1])

    if not config_file.exists():
        print(f"Config file {config_file} does not exist.")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    calib_data_file_name = Path(config.get("dump_file", ""))
    calib_data_file_name = calib_data_file_name.stem + "_associations.txt"

    if not calib_data_file_name:
        print("No dump file specified in the config. Make sure to run calib_dumper.py and calib_associator.py first")
        sys.exit(1)

    lidar_pose_ig       = config.get("lidar_offset", [0.0, 0.0, 0.0])
    kinematic_params_ig = config.get("kinematic_params", [0.0, 0.0, 0.0])

    encoder_data = []
    ass_lidar = {}
    lidar_data = []

    iteration = config.get("iterations", 30)
    tolerance = config.get("tolerance", 1e-6)
    epsilon = config.get("epsilon", 1e-6)
    dumping = config.get("dumping", 10)
    min_movement = config.get("min_movement", 1)

    with open(calib_data_file_name) as calib_data_file:
        lines = calib_data_file.readlines()
        for i, line in enumerate(lines):
            data = line.strip().split()

            encoder_ts = float(data[0])
            vel_r = float(data[1])
            vel_l = float(data[2])
            
            encoder_data.append((encoder_ts, vel_r, vel_l))
            
            if len(data) < 7:
                # line without lidar estimate

                continue
            
            lidar_x = float(data[6])
            lidar_y = float(data[7])
            lidar_theta = float(data[8])
            lidar_ts = float(data[8])

            lidar_data.append((lidar_ts, lidar_x, lidar_y, lidar_theta))
            ass_lidar[len(lidar_data)-1] = i

    calib_solver = CalibSolver(DDBodyFrameModel(*kinematic_params_ig), lidar_pose_ig, iteration=iteration, tolerance=tolerance, epsilon=epsilon, dumping=dumping, min_movement=min_movement)

    calib_solver.solve(encoder_data, lidar_data, ass_lidar)

    motion_model = DDBodyFrameModel(calib_solver.params[0], calib_solver.params[1], calib_solver.params[2])

    T_offset_lidar = exp(calib_solver.params[3], calib_solver.params[4], calib_solver.params[5])

    odom_poses = []

    for i in range(len(encoder_data)):
        if i == 0:
            continue
        ts, vel_r, vel_l = encoder_data[i]
        dt = ts - encoder_data[i-1][0]
        motion_model.getPose(vel_r, vel_l, dt)
        odom_state = motion_model.getState()

        T_odom = exp(odom_state[0], odom_state[1], odom_state[2])
        T_lidar = np.linalg.inv(T_offset_lidar) @ T_odom @ T_offset_lidar
        odom_poses.append(log(T_lidar))

    # plot the odom and lidar trajectories
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.title("Odometry vs. Lidar Poses")
    plt.plot([pose[1] for pose in lidar_data], [pose[2] for pose in lidar_data], 'b', label='Lidar')
    plt.plot([pose[0] for pose in odom_poses], [pose[1] for pose in odom_poses], 'r', label='Odometry')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.show()

    # print(calib_solver.stats)

if __name__ == "__main__":
    main()
else:
    print("This script is intended to be run as a standalone program.")
    print("Use 'python calib_solver.py <config_file>' to execute it.")
    sys.exit(1)