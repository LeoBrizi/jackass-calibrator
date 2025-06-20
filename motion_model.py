
import numpy as np
    
class DDBodyFrameModel:
    def __init__(self, kin_params):
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
    def __init__(self, kin_params) -> None:
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
    
    def getPose(self, vel_r, vel_l, dt):
        """
        Calculate the pose change over a time step.
        
        :param dt: Time step
        :return: Pose change as a tuple (dx, dy, dtheta)
        """
        self.getVel(vel_r, vel_l)

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
