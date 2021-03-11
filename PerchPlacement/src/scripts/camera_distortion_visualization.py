import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sim.cameras import Camera
import cv2 as cv

# TODO: IF USING THIS FOR ANYTHING, ENSURE CAMERA FORWARD IS ITS Z DIRECTION

# def distort_image(xu, yu):
#     # camera intrinsic parameters
#     t = 0  # pixel skew - ~0 for modern cameras
#     fx = 1  # focal length, x
#     fy = 1  # n * fx  # focal length, y
#     cx = px_x / 2  # optical center, x
#     cy = px_y / 2  # optical center, y
#
#     K = np.array([[fx, t, cx], [0, fy, cy], [0, 0, 1]])
#
#     # Radial distortion parameters
#     k1 = 5e-5
#     k2 = 5e-8
#     k3 = 5e-10
#
#     # Tangential distortion parameters
#     p1 = 0.0001
#     p2 = 0.00005
#
#     # convert image pixel positions to world units
#     r = ((xu - cx) ** 2 + (yu - cy) ** 2) ** 0.5
#     d_r = (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
#
#     xy = xu * yu
#     xc = (xu - cx) * d_r + 2 * p1 * xy + p2 * (r ** 2 + 2 * xu ** 2)
#     yc = (yu - cy) * d_r + p1 * (r ** 2 + 2 * yu ** 2) + 2 * p2 * xy
#
#     return xc, yc
#


def plot_distorted_frustum(_fig, _i, _cam, title, degrees_per_step=5):

    plt.subplot(2, 3, _i+1)
    fov = _cam.fov
    # fov_overshoot = .2
    u_ = np.linspace(0, _cam.resolution[0], int(fov[0] / degrees_per_step) + 1)  # Convert FOV into  pixels (pixel coords are 0 at top left
    v_ = np.linspace(0, _cam.resolution[1], int(fov[1] / degrees_per_step) + 1)
    # u_ = u_*fov_overshoot - _cam.resolution[0]*fov_overshoot/2
    # v_ = u_*fov_overshoot - _cam.resolution[0]*fov_overshoot/2

    M = len(u_)
    N = len(v_)
    u, v = np.meshgrid(u_, v_)  # create meshgrid
    u = u.reshape(-1)  # stack all grid points into column to pass to undistort
    v = v.reshape(-1)
    uv_ = np.expand_dims(np.vstack([u, v]).T, axis=1)

    distCoeffs = np.array([_cam.k1, _cam.k2, _cam.p1, _cam.p2, _cam.k3])
    K_new = np.array([[_cam.resolution[0] / 2, 0, -.5],
                      [0, _cam.resolution[0] / 2, -.5],
                      [0, 0, 1]])
    uv_[:, :, 0] = (uv_[:, :, 0] - np.mean(uv_[:, :, 0])) / _cam.K[0, 0]
    uv_[:, :, 1] = (uv_[:, :, 1] - np.mean(uv_[:, :, 1])) / _cam.K[0, 0]
    pts = cv.convertPointsToHomogeneous(src=uv_)
    undistored_pts = cv.undistortPoints(src=uv_, cameraMatrix=_cam.K, distCoeffs=distCoeffs, P=K_new).squeeze()

    projected_pts = cv.projectPoints(pts, rvec=np.zeros(3), tvec=np.zeros(3), cameraMatrix=_cam.K,
                                     distCoeffs=distCoeffs)[0].squeeze()

    projected_pts = projected_pts.reshape(N, M, 2)

    undistored_pts = undistored_pts.reshape(N, M, 2)
    xs = projected_pts[:,:,0]
    ys = projected_pts[:,:,1]
    xs -= np.mean(xs)
    ys -= np.mean(ys)
    plt.plot(xs, ys, color='royalblue', marker='o', linestyle='-')
    plt.plot(xs.T, ys.T, color='royalblue', marker='o', linestyle='-')
    if (i)%3 == 0:
        plt.ylabel('World Y Coordinate')
    if (i>=3):
        plt.xlabel('World X Coordinate')
    plt.title(title)
    plt.xlim([-1200, 1200])
    plt.ylim([-800, 800])

    # plt.scatter(xs, ys, 'g')
    # plt.scatter(xs, ys, 'r')


def plot_distorted_image(_fig, _i, _cam, title, degrees_per_step=5):
    plt.subplot(2, 3, _i + 1)
    fov = _cam.fov
    # fov_overshoot = .2
    u_ = np.linspace(0, _cam.resolution[0],
                     int(fov[0] / degrees_per_step) + 1)  # Convert FOV into  pixels (pixel coords are 0 at top left
    v_ = np.linspace(0, _cam.resolution[1], int(fov[1] / degrees_per_step) + 1)
    # u_ = u_*fov_overshoot - _cam.resolution[0]*fov_overshoot/2
    # v_ = u_*fov_overshoot - _cam.resolution[0]*fov_overshoot/2

    M = len(u_)
    N = len(v_)
    u, v = np.meshgrid(u_, v_)  # create meshgrid
    #
    # plt.plot(u-np.max(u)/2, v - np.max(v)/2, color='r', linestyle='-', linewidth=.5)
    # plt.plot(u.T - np.max(u)/2, v.T - np.max(v)/2, color='r', linestyle='-', linewidth=.5)

    u = u.reshape(-1)  # stack all grid points into column to pass to undistort
    v = v.reshape(-1)
    uv_ = np.expand_dims(np.vstack([u, v]).T, axis=1)

    distCoeffs = np.array([_cam.k1, _cam.k2, _cam.p1, _cam.p2, _cam.k3])
    K_new = np.array([[_cam.resolution[0] / 2, 0, -.5],
                      [0, _cam.resolution[0] / 2, -.5],
                      [0, 0, 1]])

    pts = cv.convertPointsToHomogeneous(src=uv_)
    undistored_pts = cv.undistortPoints(src=uv_, cameraMatrix=_cam.K, distCoeffs=distCoeffs, P=K_new).squeeze()

    undistored_pts = undistored_pts.reshape(N, M, 2)
    xs = undistored_pts[:, :, 0]
    ys = undistored_pts[:, :, 1]

    plt.plot(xs, ys, color='royalblue', marker='o', markersize=3, linestyle='-')
    plt.plot(xs.T, ys.T, color='royalblue', marker='o', markersize=3, linestyle='-')
    if (i) % 3 == 0:
        plt.ylabel('Undistored Image Y Coordinate')
    else:
        plt.yticks([])
    if (i >= 3):
        plt.xlabel('Undistorted Image X Coordinate')
    else:
        plt.xticks([])
    plt.title(title)
    plt.xlim([-1400, 1400])
    plt.ylim([-800, 800])


def plot_distorted_camera(_fig, _i, _cam):

    xs = range(-10, 11, 2)
    ys = range(-10, 11, 2)
    img_coords = np.zeros([len(xs), len(ys), 2])
    plt.subplot(2, 3, _i+1)
    for x in range(len(xs)):
        for y in range(len(ys)):

            # undistored_pts = cv.undistortPoints(src=uv_, cameraMatrix=_cam.K, distCoeffs=distCoeffs, P=K_new).squeeze()
            # undistored_pts = undistored_pts.reshape(N, M, 2)
            # 
            # for z in range(-10, 10):
            z = 10
            u_u, v_u = _cam.map_3D_to_img(np.array([z, ys[y], xs[x]]))
            img_coords[x, y, 0] = u_u
            img_coords[x, y, 0] = v_u

            if _cam.is_in_fov_3D_distorted(np.array([z, ys[y], xs[x]])):
                plt.scatter(xs[x], ys[y], c='g')
            else:
                plt.scatter(xs[x], ys[y], c='r')


cam = Camera()
# cam.fov = cam.fov*0.8
# cam.fov[0] = cam.fov[0]*0.8
# cam.K[0, 2] += 150

k1 = [0, 0, 0, .12, -.05,  .22]
k2 = [0, 0, 0, 0, 0,      -.14]
p1 = [0,.05,0,0,0,0]
p2 = [0,0,.05,0,0,0]

fig = plt.figure()
titles = ['No Distortion', 'Tangential Distortion \n(p1='+str(p1[1])+')',
          'Tangential Distortion \n(p2='+str(p2[2])+')', 'Barrel Distortion \n(k1='+str(k1[3])+')',
          'Pincushion Distortion \n(k1='+str(k1[4])+')',
          'Mustache Distortion \n(k1='+str(k1[5])+', k2='+str(k2[5])+')']
for i in range(6):
    cam.k1 = k1[i]
    cam.k2 = k2[i]
    cam.p1 = p1[i]
    cam.p2 = p2[i]
    plot_distorted_image(fig, i, cam, titles[i])
    # plot_distorted_camera(fig, i, cam)

fig.tight_layout(pad=2.0)
plt.show()
