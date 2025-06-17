
import numpy as np
    
class DDBodyFrameModel:
    def __init__(self, k_r, k_l, baseline):
        self.k_r = k_r  # vel_to_meter
        self.k_l = k_l  # vel_to_meter
        self.baseline = baseline  # Distance between the wheels

        self.state = np.zeros(3)  # [x, y, theta]
        self.v_r = 0.0  # Right wheel velocity
        self.v_l = 0.0  # Left wheel velocity
        self.v = 0.0  # Linear velocity
        self.omega = 0.0  # Angular velocity

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
    
    def getPose(self, vel_r, vel_l, dt):
        """
        Calculate the pose change over a time step.
        
        :param dt: Time step
        :return: Pose change as a tuple (dx, dy, dtheta)
        """
        self.getVel(vel_r, vel_l)
        dtheta = self.omega * dt
        dx = self.v * np.cos(self.state[2] + dtheta/2) * dt
        dy = self.v * np.sin(self.state[2] + dtheta/2) * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        
        return dx, dy, dtheta

class DDGlobalFrameModel:
    def __init__(self, k_r, k_l, baseline):
        self.k_r = k_r
        self.k_l = k_l
        self.baseline = baseline

        self.state = np.zeros(3)  # [x, y, theta]
        self.v_r = 0.0  # Right wheel velocity
        self.v_l = 0.0  # Left wheel velocity
        self.v = 0.0  # Linear velocity
        self.omega = 0.0  # Angular velocity

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
    
    def getPose(self, vel_r, vel_l, dt):
        """
        Calculate the pose change over a time step.
        
        :param dt: Time step
        :return: Pose change as a tuple (dx, dy, dtheta)
        """
        self.getVel(vel_r, vel_l)
        dtheta = self.omega * dt
        dx = self.v * np.sin(self.state[2] + dtheta/2) / (self.state[2] + dtheta/2) * dt
        dy = self.v * (1 - np.cos(self.state[2] + dtheta/2)) / (self.state[2] + dtheta/2) * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        
        return dx, dy, dtheta
