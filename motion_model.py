
import numpy as np

from geometry import compute_rotation_from_acc
from geometry import R, T, exp, log
    
class DDBodyFrameModel:
    def __init__(self, kin_params, *args, **kwargs):
        self.k_r = kin_params[0]  # vel_to_meter
        self.k_l = kin_params[1]  # vel_to_meter
        self.baseline = kin_params[2]  # Distance between the wheels

        self.state = np.zeros(3)  # [x, y, theta]
        self.v_r = 0.0  # Right wheel velocity
        self.v_l = 0.0  # Left wheel velocity
        self.v = 0.0  # Linear velocity
        self.omega = 0.0  # Angular velocity

    def getParams(self):
        """
        Get the parameters of the model.
        
        :return: Tuple containing (k_r, k_l, baseline)
        """
        return np.array([self.k_r, self.k_l, self.baseline])
    
    def setParams(self, new_params):
        """
        Set the parameters of the model.

        :param new_params: Array containing (k_r, k_l, baseline)
        """
        self.k_r = new_params[0]
        self.k_l = new_params[1]
        self.baseline = new_params[2]

    def getVel(self, vel_r, vel_l):
        """
        Calculate the pose change based on the velocities of the right and left wheels.
        
        :param vel_r: Velocity of the right wheel
        :param vel_l: Velocity of the left wheel
        :param dt: Time step
        :return: Pose change as a tuple (dx, dy, dtheta)
        """
        self.v_r = self.k_r * vel_r
        self.v_l = self.k_l * vel_l
        self.v = (self.v_r + self.v_l) / 2.0
        self.omega = (self.v_r - self.v_l) / self.baseline

        return self.v, self.omega
    
    def getPose(self, wheel_vel, dt): #vel_r, vel_l
        """
        Calculate the pose change over a time step.
        
        :param dt: Time step
        :return: Pose change as a tuple (dx, dy, dtheta)
        """
        self.getVel(wheel_vel[0], wheel_vel[1])
        dtheta = self.omega * dt
        dx = self.v * np.cos(self.state[2] + dtheta/2) * dt
        dy = self.v * np.sin(self.state[2] + dtheta/2) * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        
        return dx, dy, dtheta
    
    def getState(self):
        """
        Get the current state of the model.
        
        :return: Current state as a numpy array [x, y, theta]
        """
        return self.state.copy()
    
    def deepCopy(self):
        """
        Create a deep copy of the model.
        
        :return: A new instance of DDBodyFrameModel with the same parameters
        """
        copy = DDBodyFrameModel([self.k_r, self.k_l, self.baseline])
        copy.state = self.state.copy()
        copy.v_r = self.v_r
        copy.v_l = self.v_l
        copy.v = self.v
        copy.omega = self.omega
        return copy
    
    def reset(self):
        """
        Reset the state of the model to the initial state.
        """
        self.state = np.zeros(3)
        self.v_r = 0.0
        self.v_l = 0.0
        self.v = 0.0
        self.omega = 0.0

class SKSModel:
    def __init__(self, kin_params, *args, **kwargs) -> None:
        self.k_r = kin_params[0]  # vel_to_meter
        self.k_l = kin_params[1]  # vel_to_meter
        self.baseline = kin_params[2]  # Distance between the wheels
        self.alpha = kin_params[3]  # skid coefficient

        self.state = np.zeros(3)  # [x, y, theta]
        self.v_r = 0.0  # Right wheel velocity
        self.v_l = 0.0  # Left wheel velocity
        self.v = 0.0  # Linear velocity
        self.omega = 0.0  # Angular velocity

    def setParams(self, params):
        """
        Set the parameters of the SKS model.
        
        :param params: Parameters to set
        """
        self.k_r = params[0]
        self.k_l = params[1]
        self.baseline = params[2]
        self.alpha = params[3]

    def getParams(self):
        """
        Get the parameters of the SKS model.
        
        :return: Parameters of the SKS model
        """
        return np.array([self.k_r, self.k_l, self.baseline, self.alpha])

    def getVel(self, vel_r, vel_l):
        """
        Calculate the pose change based on the velocities of the right and left wheels.
        
        :param vel_r: Velocity of the right wheel
        :param vel_l: Velocity of the left wheel
        :param dt: Time step
        :return: Pose change as a tuple (dx, dy, dtheta)
        """
        self.v_r = self.k_r * vel_r
        self.v_l = self.k_l * vel_l
        self.v = (self.v_r + self.v_l) / 2.0
        self.omega = (self.v_r - self.v_l) / self.baseline

        return self.v, self.omega
    
    def getPose(self, wheel_vel, dt): #vel_r, vel_l
        """
        Calculate the pose change over a time step.
        
        :param dt: Time step
        :return: Pose change as a tuple (dx, dy, dtheta)
        """
        self.getVel(wheel_vel[0], wheel_vel[1])

        dtheta = self.omega * dt

        cos_theta = np.cos(self.state[2] + dtheta/2)
        sin_theta = np.sin(self.state[2] + dtheta/2)

        dx = self.v * cos_theta * dt - self.alpha * self.omega * sin_theta * dt
        dy = self.v * sin_theta * dt + self.alpha * self.omega * cos_theta * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        
        return dx, dy, dtheta
    
    def getState(self):
        """
        Get the current state of the model.
        
        :return: Current state as a numpy array [x, y, theta]
        """
        return self.state.copy()
    
    def deepCopy(self):
        """
        Create a deep copy of the model.
        
        :return: A new instance of SKSModel with the same parameters
        """
        copy = SKSModel([self.k_r, self.k_l, self.baseline, self.alpha])
        copy.state = self.state.copy()
        copy.v_r = self.v_r
        copy.v_l = self.v_l
        copy.v = self.v
        copy.omega = self.omega
        return copy
    
    def reset(self):
        """
        Reset the state of the model to the initial state.
        """
        self.state = np.zeros(3)
        self.v_r = 0.0
        self.v_l = 0.0
        self.v = 0.0
        self.omega = 0.0


class DDGyroModel:

    def __init__(self, kin_params, *args, **kwargs):
        self.k_r = kin_params[0]  # vel_to_meter
        self.k_l = kin_params[1]  # vel_to_meter
        # self.baseline = kin_params[2]  # Distance between the wheels
        self.imu_offset = kin_params[2]  # IMU offset from the center of the robot

        self.state = np.zeros(3)  # [x, y, theta]
        self.v_r = 0.0  # Right wheel velocity
        self.v_l = 0.0  # Left wheel velocity
        self.v = 0.0  # Linear velocity
        # self.omega = 0.0  # Angular velocity

        # self.imu_measurement = []  # Placeholder for IMU measurement [ts, linear_acceleration 3x1, angular_velocity 3x1]
        # self.gyro_bias = np.zeros(3)

        # self.R_imu_in_world = np.eye(3)  # Rotation matrix from IMU to world frame
        # self.estimate_g = False
        
        # self.min_time_still = kwargs.pop('min_time_still', 1.0)  # Minimum time to estimate gravity vector


    def getParams(self):
        return np.array([self.k_r, self.k_l, self.imu_offset])
    
    def setParams(self, new_params):
        self.k_r = new_params[0]
        self.k_l = new_params[1]
        self.imu_offset = new_params[2]

    def getLinearVel(self, vel_r, vel_l):
        self.v_r = self.k_r * vel_r
        self.v_l = self.k_l * vel_l
        self.v = (self.v_r + self.v_l) / 2.0
        # self.omega = (self.v_r - self.v_l) / self.baseline
        return self.v

    # def estimateBias(self):
    #     """
    #     Estimate the bias from the IMU measurement.
        
    #     :param imu_measurement: IMU measurement containing angular velocity
    #     :return: Estimated bias
    #     """
    #     self.gyro_bias = np.mean([m[2] for m in self.imu_measurement], axis=0)
    #     return self.gyro_bias
    
    # def estimateGravityVector(self):
    #     """
    #     Estimate the gravity vector from the IMU measurements.
        
    #     :return: Estimated gravity vector
    #     """
    #     if len(self.imu_measurement) == 0:
    #         print("No IMU measurements available to estimate gravity vector.")
    #         return self.estimate_g
        
    #     time_still = self.imu_measurement[-1][0] - self.imu_measurement[0][0]
        
    #     if time_still < self.min_time_still:
    #         print("Not enough time to estimate gravity vector.")
    #         return self.estimate_g
    #     # Assuming imu_measurement is a list of tuples (timestamp, linear_acceleration, angular_velocity)

    #     # Step 1: Average the accelerometer measurements
    #     acc_mean = np.mean([m[1] for m in self.imu_measurement], axis=0)
    #     # Optional: Normalize if you only care about direction
    #     acc_normalized = acc_mean / np.linalg.norm(acc_mean)

    #     self.R_imu_in_world = compute_rotation_from_acc(acc_normalized)

    #     self.estimate_g = True
    #     return self.estimate_g
    
    # def integralDeltaOmega(self, imu_measurements):
    #     dtheta = 0.0
    #     for index,imu_measurement in enumerate(imu_measurements):
    #         if index == 0:
    #             continue
    #         ts, _, angular_velocity = imu_measurement
    #         # Subtract the bias from the angular velocity
    #         angular_velocity -= self.gyro_bias
            
    #         vel_aligned = R @ angular_velocity
    #         omega = vel_aligned[2]
    #         inc_theta = omega * (ts - imu_measurements[0][0])

    def getPose(self, vels_imu, dt): #vel_r, vel_l, imu_theta
        """
        Calculate the pose change over a time step.
        
        :param vel_r: Velocity of the right wheel
        :param vel_l: Velocity of the left wheel
        :param dt: Time step
        :return: Pose change as a tuple (dx, dy, dtheta)
        """
        self.getLinearVel(vels_imu[0], vels_imu[1])
        
        # if self.v == 0.0:
        #     self.imu_measurement.extend(imu_measurements)
        #     return 0.0, 0.0, 0.0
        # else:
        #     if not self.estimate_g:
        #         self.estimateGravityVector()

        #     if len(self.imu_measurement) > 1:
        #         time_still = self.imu_measurement[-1][0] - self.imu_measurement[0][0]
        #         if time_still > self.min_time_still: 
        #             self.estimateBias()
        #     self.imu_measurement = []  # Clear the IMU measurements after processing
        imu_theta_in_robot = -vels_imu[2] + self.imu_offset

        dtheta = imu_theta_in_robot - self.state[2]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
            
        dx = self.v * np.cos(self.state[2] + dtheta/2) * dt
        dy = self.v * np.sin(self.state[2] + dtheta/2) * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] = imu_theta_in_robot  # Update the state with the IMU theta
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        
        return dx, dy, dtheta
    
    def getState(self):
        """
        Get the current state of the model.
        
        :return: Current state as a numpy array [x, y, theta]
        """
        return self.state.copy()
    
    def deepCopy(self):
        """
        Create a deep copy of the model.
        
        :return: A new instance of DDBodyFrameModel with the same parameters
        """
        copy = DDGyroModel([self.k_r, self.k_l, self.imu_offset])
        copy.state = self.state.copy()
        copy.v_r = self.v_r
        copy.v_l = self.v_l
        copy.v = self.v
        return copy
    
    def reset(self):
        """
        Reset the state of the model to the initial state.
        """
        self.state = np.zeros(3)
        self.v_r = 0.0
        self.v_l = 0.0
        self.v = 0.0