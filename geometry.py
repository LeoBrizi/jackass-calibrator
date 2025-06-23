import numpy as np
from scipy.spatial.transform import Rotation


def R(theta: float) -> np.ndarray:
    """
    Returns a 2D rotation matrix for a given angle in radians.

    :param theta: Angle in radians
    :return: 2D rotation matrix
    """
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )


def T(x: float, y: float, R: np.ndarray) -> np.ndarray:
    """
    Returns a 2D translation matrix for a given translation vector (x, y).

    :param x: Translation in the x direction
    :param y: Translation in the y direction
    :param R: Rotation matrix
    :return: 2D transformation matrix
    """
    return np.array(
        [[R[0, 0], R[0, 1], x], [R[1, 0], R[1, 1], y], [0, 0, 1]], dtype=np.float32
    )


def exp(x, y, theta):
    """
    Returns the exponential map of a 2D vector (x, y) and an angle theta.
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


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


def compute_rotation_from_acc(acc_normalized):
    a_x, a_y, a_z = acc_normalized

    if a_z >= 0:
        t = a_z + 1
        k = np.sqrt(2 * t)
        qw = np.sqrt(t / 2)
        qx = -a_y / k
        qy = a_x / k
        qz = 0.0
    else:
        t = 1.0 - a_z
        k = np.sqrt(2 * t)
        qw = -a_y / k
        qx = np.sqrt(t / 2)
        qy = 0.0
        qz = a_x / k

    # Create quaternion [x, y, z, w]
    q_acc = Rotation.from_quat([qx, qy, qz, qw])

    # Equivalent to: q_acc.normalized().toRotationMatrix().transpose().cast<float>()
    R_imu_in_world = q_acc.as_matrix().T.astype(np.float32)

    return R_imu_in_world


def yaw_from_quaternion(q):
    r = Rotation.from_quat([q.x, q.y, q.z, q.w])
    roll, pitch, yaw = r.as_euler("xyz", degrees=False)
    return yaw
