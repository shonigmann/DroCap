import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString, LinearRing, Point

from geom.geometry2d import rot2d
from geom.tools import get_inv_sqr_coefficients

def fill_cone_2d(pose=np.array([0, 0, 0]), fov=70.0, camera_range=np.array([0.1, 3]), alpha=0.5, color="r"):
    """
    Draws a trapezoidal (truncated cone) segment in the last used plot according to input parameters. Used to vis
    2D camera placement

    :param pose: 2D pose of cone tip
    :param fov: cone tip angle
    :param camera_range: range of heights measured from the cone tip, [min_range, max_range]
    :param alpha: opacity value
    :param color: color value
    :return:
    """
    p0 = pose[0:2]
    Rp = rot2d(pose[2]+fov/2)
    Rm = rot2d(pose[2]-fov/2)

    # clockwise near
    cw_n = p0 + np.dot(Rm, np.array([camera_range[0], 0]))
    # ccw near
    ccw_n = p0 + np.dot(Rp, np.array([camera_range[0], 0]))
    # cw far
    cw_f = p0 + np.dot(Rm, np.array([camera_range[1], 0]))
    # ccw far
    ccw_f = p0 + np.dot(Rp, np.array([camera_range[1], 0]))

    pts = np.array([[cw_n], [ccw_n], [ccw_f], [cw_f]]).squeeze()
    xs, ys = zip(*pts)
    plt.fill(xs, ys, color, alpha=alpha)


def fill_cone_gradient_2d(pose=np.array([0, 0, 0]), fov=30.0, camera_range=np.array([0.1, 3]),
                          alphas=np.array([1.0, 0.1]), function="INV_SQR", b=0, n=1, color="c", steps=20):
    """
    Draws a more realistic camera in the last used plot window. An inverse square or linear gradient is approximated by
    a discrete number of steps with a constant opacity.

    :param pose: 2D pose of cone tip
    :param fov: cone tip angle
    :param camera_range: range of heights measured from the cone tip, [min_range, max_range]
    :param alphas: 2D array corresponding to the minimum and maximum opacity to use
    :param function: either "INV_SQR" or "LINEAR" function to determine how opacity decays with distance from tip
    :param color: color of cone as matplotlib code (e.g. "r") or rgb value
    :param steps: number of discrete steps over which to approximate the gradient
    :return:
    """
    x1 = camera_range[0]
    y1 = alphas[0]
    x2 = camera_range[1]
    y2 = alphas[1]

    if function == "INV_SQR":
        # inverse square law...
        # solve the quadratic equation for vars
        a = lambda r_: n/(np.power(r_-b, 2))

    else:
        # linear gradient
        b = (y2-y1)/(y2*x2-y1*x1)
        n = y1*(x1-b)
        a = lambda r_: n/(r_-b)

    step_size = (camera_range[1]-camera_range[0])/steps
    for i in range(steps):
        r = np.array([camera_range[0]+i*step_size, camera_range[0]+(i+1)*step_size])
        r_avg = 1/2*(r[0]+r[1])
        fill_cone_2d(pose, fov, r, a(r_avg), color)


def place_cameras_2d(cameras, environment, color='c', tgt_steps=20):
    """
    Plots a list of cameras as 2D gradient cones in the most recently open plot

    :param cameras: a list of Camera objects corresponding to the cameras which have already been placed in an
        environment
    :param environment: an Environment class instance
    :param color: color of the camera cones to be plotted
    :param tgt_steps: number of steps over which discrete gradient should be plotted
    :return:
    """

    if environment.obstacles is None:
        for camera in cameras:
            fill_cone_gradient_2d(pose=np.array([camera.pose[0], camera.pose[1], camera.pose[-1]]), fov=camera.fov,
                                  camera_range=camera.range, b=camera.b, n=camera.n,
                                  alphas=camera.score_range, color=color)
    else:
        for i in range(len(cameras)):
            # break FOV into ~20 triangles
            dth = cameras[i].fov[0] / tgt_steps
            angles = np.linspace(cameras[i].pose[-1]-cameras[i].fov[0]/2+dth/2,
                                 cameras[i].pose[-1]+cameras[i].fov[0]/2-dth/2, tgt_steps)
            sub_pose = cameras[i].pose.copy()

            for th in angles:
                r_ = cameras[i].range.copy()  # reset range for each sub-segment
                sub_pose[-1] = th
                R = rot2d(th)

                end_point = sub_pose[:2] + np.dot(R, np.array([r_[-1], 0]))
                line = LineString(np.array([sub_pose[:2], end_point]))
                dist_to_obs = np.inf
                for obs in environment.obstacles:
                    obs_edges = LinearRing(obs)
                    intersection = obs_edges.intersection(line)
                    if intersection.type == 'Point':  # single intersection...
                        d = intersection.distance(Point(sub_pose[:2]))
                        dist_to_obs = np.min([d, dist_to_obs])
                    elif intersection.type == 'MultiPoint':  # double intersection...
                        for p in intersection:
                            d = p.distance(Point(sub_pose[:2]))
                            dist_to_obs = np.min([d, dist_to_obs])

                for e in range(len(r_)):
                    r_[e] = np.min([r_[e], dist_to_obs])

                if r_[-1] != cameras[i].range[-1]:
                    # compute new alpha values. default is 1 to 0.1 between r[0] and r[1]
                    b, n = get_inv_sqr_coefficients(cameras[i].range, np.array([1, 0.1]))
                    alpha_r_ = n/np.power(r_[-1]-b, 2)

                    a_ = np.array([1, alpha_r_])
                else:
                    a_ = np.array([1, 0.1])

                fill_cone_gradient_2d(pose=np.array([sub_pose[0], sub_pose[1], sub_pose[-1]]), fov=dth, camera_range=r_,
                                      alphas=a_, color=color)


def draw_boundary_2d(vertices, color="b"):
    """
    Draws the boundary of a polygon in 2D specified by an array vertices on the most recently used plot window

    :param vertices: np.array containing the 2D coordinates of the polygon's vertices
    :param color: color to draw the edges in
    :return:
    """
    vertices = np.concatenate((vertices, np.array([vertices[0, :]])))
    xs, ys = zip(*vertices)  # create lists of x and y values
    plt.plot(xs, ys, color, alpha=1)
