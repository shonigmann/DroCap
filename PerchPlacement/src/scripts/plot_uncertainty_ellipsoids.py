from sim.cameras import Camera
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
s = 8
c1 = Camera(pose=np.array([-4, 0, 0, 0, 0, 0])/s)
c2 = Camera(pose=np.array([-2, -3, 0, 0, -30, 0])/s)

fig = plt.figure(1, figsize=plt.figaspect(1))  # Square figure
fig2 = plt.figure(2, figsize=plt.figaspect(1))  # Square figure
ax1 = fig.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')
axes = [ax1, ax2]

pts = np.array([[0.1, 0.05, .05],
                [.1, -.05, .05],
                [-.0, -.05, -.05],
                [-.0, .05, -.05]]) * 20 / s
i = 0
for ax in axes:
    if i == 0:
        c1.plot_covariance_ellipsoids_at_points(pts[1:], ax, show=False, color='r')
        c2.plot_covariance_ellipsoids_at_points(pts, ax, show=False, color='b')
    else:
        C1 = c1.estimate_covariance(pts)
        C2 = c2.estimate_covariance(pts)
        j = 0
        for pt in pts:
            if j==0:
                C_net = C2[0]
            else:
                C_net = np.linalg.inv(np.linalg.inv(C1[j]) + np.linalg.inv(C2[j]))*1.5
            c1.plot_matching_error_ellipsoid(pt, C_net, ax, color='c', show=False)

            j+=1
    i += 1

    ax.scatter3D(c1.pose[0], c1.pose[1], c1.pose[2], c='r')
    ax.scatter3D(c2.pose[0], c2.pose[1], c2.pose[2], c='b')

    from tools.tools import set_axes_equal
    set_axes_equal(ax)
    ax.view_init(elev=90., azim=180)

    for pt in pts:
        x = np.array([pt[0], c1.pose[0]])
        y = np.array([pt[1], c1.pose[1]])
        z = np.array([pt[2], c1.pose[2]])
        ax.plot3D(x,y,z, '--', color='grey')
        x = np.array([pt[0], c2.pose[0]])
        y = np.array([pt[1], c2.pose[1]])
        z = np.array([pt[2], c2.pose[2]])
        ax.plot3D(x,y,z, '--', color='grey')

    pt = c1.pose[:3] + (pts[0] - c1.pose[:3])*.8
    ax.scatter3D(pt[0], pt[1], pt[2], c='r', marker='x')

    line = np.array([pt+np.array([0, -.05, 0]), pt+np.array([0, .05, 0])]).T
    ax.plot3D(line[0], line[1], line[2], 'r')

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    ax.set_xlim([-.9*5/s, 3./s])
    ax.set_ylim([-.8*5/s, .5*5/s])

plt.show()
print("done")
