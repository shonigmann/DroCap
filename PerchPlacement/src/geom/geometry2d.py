import numpy as np
from shapely.geometry import Polygon, Point


def _rot2eul(R, unit="degrees"):
    # from pg 6 in https://apps.dtic.mil/dtic/tr/fulltext/u2/a497476.pdf
    # needs verification
    tilt = np.arctan2(-R[1, 3], R[2, 2])
    pan = np.arcsin(R[0, 2])
    roll = np.arctan2(-R[0, 1], R[0, 0])
    eul = np.array([roll, tilt, pan])
    if unit == "degrees":
        eul = eul * 180 / np.pi

    return eul


def eul2vec_x(euler, unit="degrees"):
    """
    DEPRECATED

    Function to convert euler angles (roll, pitch, yaw) into a direction vector (roll is ignored).
    :param euler: euler angles (roll, pitch, yaw)
    :param unit: whether the euler angles are in degrees or radians
    :return: the unit vector corresponding with the new x axis direction after roll and pitch are applied
    """
    if unit == "degrees":
        # roll = euler[0] / 180 * np.pi  # don't actually care about rol, only that it's pointed at the right point
        pitch = euler[1] / 180 * np.pi
        yaw = euler[2] / 180 * np.pi
    else:
        pitch = euler[1]
        yaw = euler[2]

    # print("Yaw: " + str(yaw))
    # print("Pitch: " + str(pitch))
    # print("Euler: " + str(euler))

    if np.abs(yaw) == np.inf or np.abs(pitch) == np.inf:
        print("WARNING: INVALID ANGLE")
        print(pitch)
        print(yaw)

    x = np.cos(yaw) * np.cos(pitch)
    y = np.sin(yaw) * np.cos(pitch)
    z = np.sin(pitch)

    return np.array([x, y, z])


def vec2eul(vec, unit="degrees"):
    """
    Converts a directional vector into pitch and yaw angles (no roll). Assumes FWD-LEFT-UP convention
    :param vec: normalized direction vector
    :param unit: whether the output should be in degrees or radians
    :return:
    """
    vec /= np.linalg.norm(vec)
    pan = np.arctan2(vec[1], vec[0])
    tilt = np.arcsin(-vec[2])  # positive tilt results in tilting down in the z direction (hence the negative sign)
    roll = 0

    if unit == "degrees":
        pan = pan * 180 / np.pi
        tilt = tilt * 180 / np.pi

    eul = np.array([roll, tilt, pan])

    return eul


def rot2d(th, unit="degrees"):
    """
    Simple function that returns a 2D rotation matrix for a given angle
    :param th: angle
    :param unit: specifies whether the angle is in degrees or radians
    :return:
    """
    if unit == "degrees":
        th = th/180*np.pi
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def is_in_region(point, poly_bounds):
    """
    Determines if a point is in a polygon. Only works for 2D (x,y) polygons as Shapely Polygon library ignores Z axis
    :param point:
    :param poly_bounds:
    :return: boolean, True if point is in polygon
    """

    polygon = Polygon(poly_bounds)
    point = Point(point)

    return polygon.contains(point)


def min_ang_dist(a1, a2, units="degrees"):
    """
    Determine the minimum angular difference, on [-180,180) between angle1 and angle2,
    regardless of how a1 and a2 are represented
    :param a1: float, angle 1
    :param a2: float, angle 2
    :param units: string, if "degrees" use 360, else use pi
    :return: min_diff, minimum angle between the two angles
    """
    if units == "degrees":
        delta_angle = (a1 - a2 + 180) % 360 - 180
        if delta_angle < -180:
            delta_angle += 360
    else:
        a1 *= 180/np.pi  # convert to degrees to take advantage of modulo operator
        a2 *= 180/np.pi
        delta_angle = (a1 - a2 + 180) % 360 - 180
        if delta_angle < -180:
            delta_angle += 360
        delta_angle *= np.pi/180

    return delta_angle
