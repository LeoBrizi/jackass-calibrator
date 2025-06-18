import os
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import json

from geometry import T,R, exp, log
from motion_model import DDBodyFrameModel, DDGlobalFrameModel

class CalibSolver:
    def __init__(self, kinematic_model, lidar_offset,  *args, **kwargs):
        self.kinematic_model = kinematic_model

        self.kin_param_init = self.kinematic_model.getParams().copy()
        self.lidar_offset = lidar_offset.copy()
        self.params = np.concatenate((self.kin_param_init, self.lidar_offset))

        self.iterations = kwargs.pop('iterations', 50)
        self.tolerance = kwargs.pop('tolerance', 1e-4)
        self.epsilon = kwargs.pop('epsilon', 1e-6)
        self.dumping = kwargs.pop('dumping', 1000)
        self.min_movement = kwargs.pop('min_movement', 0.3)
        self.incs = np.eye(len(self.params), dtype=np.float64) * self.epsilon

        self.stats = {}

    def error_(self, params, vel_r, vel_l, dt, lidar_pose):
        kin_model_temp = self.kinematic_model.deepCopy()
        kin_model_temp.setParams(params[:3])

        kin_model_temp.getPose(vel_r, vel_l, dt)
        odom_state = kin_model_temp.getState()

        T_lidar_offset = exp(params[3], params[4], params[5])
        T_odom = exp(odom_state[0], odom_state[1], odom_state[2])
        T_lidar = exp(lidar_pose[1], lidar_pose[2], lidar_pose[3])

        residual = np.linalg.inv(T_lidar) @ np.linalg.inv(T_lidar_offset) @ T_odom @ T_lidar_offset

        return log(residual).reshape((3, 1))

    def errorAndJacobian_(self, encoder_data, lidar_data, ass_lidar):
        """
        This function computes the residuals between the expected and actual poses
        based on the encoder data and lidar associations.
        """
        residuals = []
        jacobians = []
        odom_pose = []
        lidar_pose = []
        prev_timestamp = encoder_data[0][0]
        prev_pose = self.kinematic_model.getState()
        for index, (encoder_ts, vel_r, vel_l) in enumerate(encoder_data):
            dt = encoder_ts - prev_timestamp

            motion = np.linalg.norm(self.kinematic_model.getState()[0:1] - prev_pose[0:1])

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
            odom_pose.append(prev_pose)
            lidar_pose.append(lidar_data[ass_lidar[index]])

        return residuals, jacobians, odom_pose, lidar_pose

    def solve(self, encoder_data, lidar_data, ass_lidar):
        
        prev_cost = np.inf

        for iteration in range(self.iterations):

            residuals, jacobians, odom_pose, lidar_pose = self.errorAndJacobian_(encoder_data, lidar_data, ass_lidar)

            #plot odom and lidar poses
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.title("Odometry vs. Lidar Poses")
            plt.plot([pose[0] for pose in odom_pose], [pose[1] for pose in odom_pose], 'r', label='Odometry')
            plt.plot([pose[0] for pose in lidar_pose], [pose[1] for pose in lidar_pose], 'b', label='Lidar')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.show()

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

            self.kinematic_model.reset()
            self.kinematic_model.setParams(self.params[:3])
            self.lidar_offset = self.params[3:]
            self.kin_param_init = self.params[3:]

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
            ass_lidar[i] = len(lidar_data)-1

    calib_solver = CalibSolver(DDBodyFrameModel(*kinematic_params_ig), lidar_pose_ig)

    calib_solver.solve(encoder_data, lidar_data, ass_lidar)

    # print(calib_solver.stats)

if __name__ == "__main__":
    main()
else:
    print("This script is intended to be run as a standalone program.")
    print("Use 'python calib_solver.py <config_file>' to execute it.")
    sys.exit(1)