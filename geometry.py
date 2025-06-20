import numpy as np

def R(theta: float) -> np.ndarray:
    """
    Returns a 2D rotation matrix for a given angle in radians.
    
    :param theta: Angle in radians
    :return: 2D rotation matrix
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]], dtype=np.float32)

def T(x: float, y: float, R: np.ndarray) -> np.ndarray:
    """
    Returns a 2D translation matrix for a given translation vector (x, y).
    
    :param x: Translation in the x direction
    :param y: Translation in the y direction
    :param R: Rotation matrix
    :return: 2D transformation matrix
    """
    return np.array([[R[0, 0], R[0, 1], x],
                     [R[1, 0], R[1, 1], y],
                     [0, 0, 1]], dtype=np.float32)

def exp(x,y,theta):
    """
    Returns the exponential map of a 2D vector (x, y) and an angle theta.
    """
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]], dtype=np.float32)

def log(T: np.ndarray) -> np.ndarray:
    """
    Returns the logarithm of a 2D transformation matrix T.
    
    :param T: 2D transformation matrix
    :return: Logarithm of the transformation matrix as a 3D vector (x, y, theta)
    """
    theta = np.arctan2(T[1, 0], T[0, 0])
    x = T[0, 2]
    y = T[1, 2]
    return np.array([x, y, theta], dtype=np.float32)