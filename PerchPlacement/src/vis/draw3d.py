import numpy as np
import warnings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geom.geometry3d import rot3d_from_rtp, dist, generate_points_on_fov_front
from tools.tools import set_axes_equal


def draw_poly_3d(vertices, ax=None, color="b", fill_alpha=0.3, plot_border=False):
    """
    Draws a polygon in 3D using matplotlib

    :param vertices: array of 3D vertex locations
    :param ax: ax of the 3D plot on which to draw the polygon
    :param color: color of the polygon and edges
    :param fill_alpha: the opacity value [0, 1] of the polygon fill
    :param plot_border: whether or not the edges should be drawn
    :return:
    """
    if ax is None:
        fig = plt.figure(0)
        ax = Axes3D(fig)

    if not np.array_equal(vertices[-1, :], vertices[0, :]):
        vertices = np.concatenate((vertices, np.array([vertices[0, :]])))  # close the loop
    if plot_border:
        # plot border
        ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], color)

    # fill polygon
    collection = convert_points_to_Poly3D(vertices, line_width=0, alpha=fill_alpha)
    if plot_border:
        collection.set_edgecolor(color)
    collection.set_facecolor(color)
    ax.add_collection3d(collection)


def draw_room_3d(environment, ax=None, cameras=None, placed_cameras=None, ground_plane_offset=2.5, fill_alpha=0.1,
                 plot_target=False, ):
    """
    Draws a room in 3D, including the walls, obstructions, cameras, and a ground plane.
    :param environment: Environment class instance
    :param ax: plot ax handle of the 3D plot on which to draw the room
    :param cameras: a list of Camera objects corresponding to the cameras which have already been placed in an
        environment
    :param placed_cameras: PlacedCameras class instance
    :param ground_plane_offset: distance offset around wall perimeter to extend ground plane
    :param fill_alpha: opacity value used when plotting walls
    :return:
    """
    if ax is None:
        fig = plt.figure(1)
        ax = Axes3D(fig)

    if environment.dimension == 3:
        if cameras is not None:
            for camera in cameras:
                draw_camera_3d(camera=camera, ax=ax, environment=environment)  # , obstacles=environment.obstacles)
                draw_quad_3d(camera, ax, "orange")

        if placed_cameras.cameras is not None:
            for placed_camera in placed_cameras.cameras:
                draw_camera_3d(camera=placed_camera, ax=ax, color="m", environment=environment)
                draw_quad_3d(placed_camera, ax, "orange")
        # commenting out because obstacles contain the walls (more or less). no sense double drawing...
        # for polygon in environment.walls:
        #     draw_poly_3d(polygon, ax, "b", fill_alpha)
        if type(environment).__name__ == 'MeshEnvironment':
            for mesh in environment.perch_regions:
                if len(mesh.perimeter_loop) >= 1:
                    for pl in mesh.perimeter_loop:
                        draw_poly_3d(np.squeeze(mesh.points[pl, :]), ax, "r", fill_alpha, plot_border=True)
                else:
                    mesh.plot_mesh(ax, color=np.array([255, 0, 255]), opacity=fill_alpha*255)

            for mesh in environment.obstacles:
                if mesh.is_valid:
                    if len(mesh.perimeter_loop) >= 1:
                        for pl in mesh.perimeter_loop:
                            draw_poly_3d(np.squeeze(mesh.points[pl, :]), ax, "r", fill_alpha, plot_border=True)
                    else:
                        mesh.plot_mesh(ax, color=np.array([255, 0, 0]), opacity=fill_alpha*255)

            if plot_target:
                environment.target.plot_mesh(ax, color=np.array([0, 200, 0]))
            ax.scatter3D(environment.sampling_points[:, 0], environment.sampling_points[:, 1],
                         environment.sampling_points[:, 2])
                # environment.target.plot_multi_colored_mesh(ax)

        else:
            for polygon in environment.perch_regions:
                draw_poly_3d(polygon, ax, "y", fill_alpha, plot_border=True)
            for polygon in environment.obstacles:
                draw_poly_3d(polygon, ax, "r", fill_alpha, plot_border=True)

            draw_poly_3d(environment.target, ax, "g", fill_alpha, plot_border=True)

            # find ground plane limits:
            x_lim = environment.x_lim
            y_lim = environment.y_lim
            ground_plane = np.array([[x_lim[0]-ground_plane_offset, y_lim[0]-ground_plane_offset, 0],
                                     [x_lim[1]+ground_plane_offset, y_lim[0]-ground_plane_offset, 0],
                                     [x_lim[1]+ground_plane_offset, y_lim[1]+ground_plane_offset, 0],
                                     [x_lim[0]-ground_plane_offset, y_lim[1]+ground_plane_offset, 0]])
            draw_poly_3d(ground_plane, ax, "grey", fill_alpha=0.6)

        set_axes_equal(ax)

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    else:
        print("ERROR: INCORRECT ENVIRONMENT DIMENSION")


def draw_quad_3d(camera, ax, frame_color="k", prop_color=None, opt_options=None, fill_alpha=0.5):
    if opt_options is not None:
        r_prop = camera.prop_rad
        frame_size = camera.frame_rad*2
    else:
        r_prop = 0.1
        frame_size = 0.4

    if prop_color is None:
        prop_color = frame_color
    pos = camera.pose
    norm = camera.wall_normal

    pitch = np.arccos(norm[2])
    if np.abs(pitch % 2*np.pi) <= 0.00001:
        warnings.warn("Normal Vector: " + str(norm) + " results in an invalid conversion to euler angles")
        yaw = 0  # purely cosmetic, but could be nice to deal with this...
    else:
        yaw = np.arccos((norm[0] / np.sin(pitch)))
    roll = 0

    R = rot3d_from_rtp(np.array([roll, pitch, yaw]), unit="radians")

    # draw frame
    frame_verts = [
        frame_size / 2 * np.array([[np.sqrt(2) / 2, np.sqrt(2) / 2, 0], [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0]]),
        frame_size / 2 * np.array([[np.sqrt(2) / 2, -np.sqrt(2) / 2, 0], [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0]])]
    for i in range(2):
        for j in range(2):
            frame_verts[i][j, :] = np.dot(R, frame_verts[i][j, :]) + pos[:3]

    ax.plot(frame_verts[0][:, 0], frame_verts[0][:, 1], frame_verts[0][:, 2], frame_color)
    ax.plot(frame_verts[1][:, 0], frame_verts[1][:, 1], frame_verts[1][:, 2], frame_color)

    # draw props
    R = r_prop  # np.linspace(0, r_prop, 2)
    u = np.linspace(0, 2 * np.pi, 8)
    x = np.squeeze(np.outer(R, np.cos(u)))
    y = np.squeeze(np.outer(R, np.sin(u)))
    z = np.zeros(x.size)
    prop_coords = np.array([x, y, z]).transpose()
    prop_coords = np.vstack((prop_coords, prop_coords[0, :]))

    # rotate!
    for p in range(len(prop_coords)):
        prop_coords[p, :] = np.dot(R, prop_coords[p, :])

    # shift and plot
    for i in range(2):
        for j in range(2):
            pc = prop_coords + frame_verts[i][j, :]

            # fill polygon
            collection = convert_points_to_Poly3D(pc, line_width=0.0, alpha=fill_alpha)
            collection.set_edgecolor(prop_color)
            collection.set_facecolor(prop_color)
            ax.add_collection3d(collection)
            ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], "black")  # for testing


def convert_points_to_Poly3D(pts, line_width=0.0, alpha=1.0):
    x = pts[:, 0].tolist()
    y = pts[:, 1].tolist()
    z = pts[:, 2].tolist()
    verts = [list(zip(x, y, z))]
    collection = Poly3DCollection(verts, linewidths=line_width, alpha=alpha)
    return collection


def draw_pyramid_3d(position, R, fov, height_range, ax, draw_walls=np.array([True, True, True, True, True, True]),
                    color='c', alpha=0.3):
    """
    This function draws a truncated rectangular pyramid given a tip pose, height range (measured from the tip), and
    tip angles specified by FOV

    :param position: 3d np.array containing tip position
    :param R: 3x3 rotation matrix corresponding to the orientation of the pyramid's central axis (tip to base center)
    :param fov: 2d np.array containing the pyramid tip angles in primary and secondary axes
    :param height_range: 2d np.array containing the range of heights (measured from the tip) over which to plot the
    pyramid
    :param ax: plot ax handle, of the plot on which to draw this pyramid
    :param draw_walls: boolean list corresponding to whether or not the left, top, right, bottom, front and back sides
    of the pyramid should be visualized
    :param color: color of pyramid fill and edges
    :param alpha: pyramid surface opacity
    :return:
    """

    # each row is a relative vector from the camera to (in order): near top left -> cw -> nbl, far top left -> cw -> fbl
    vectors = np.zeros([2, 4, 3])
    vectors[0, :, :] = generate_points_on_fov_front(position=position, R=R, fov=fov,  # np.array([fov[1], fov[0]]),
                                                    radius=height_range[0], n_horiz=2, n_vert=2).reshape([4, 3])
    vectors[1, :, :] = generate_points_on_fov_front(position=position, R=R, fov=fov,  # np.array([fov[1], fov[0]]),
                                                    radius=height_range[1], n_horiz=2, n_vert=2).reshape([4, 3])
    # reorder to get proper shape...
    tmp = vectors[:, 3, :].copy()
    vectors[:, 3, :] = vectors[:, 2, :]
    vectors[:, 2, :] = tmp

    # create and plot a quadrilateral polygon in 3D space for each side of the pyramid
    # (in order: left side, top side, right side, bottom side)  -> feels like i need to rotate this 90deg TODO (trying -1 now)
    for i in range(4):
        if draw_walls[i-3]:
            vertices = np.array([vectors[0, i], vectors[0, i-1], vectors[1, i-1], vectors[1, i]])
            draw_poly_3d(vertices=vertices, ax=ax, color=color, fill_alpha=alpha)

    # draw ends of the truncated pyramid:
    for i in range(2):
        if draw_walls[i+4]:
            vertices = vectors[i, :, :]
            draw_poly_3d(vertices=vertices, ax=ax, color=color, fill_alpha=alpha)


def draw_camera_3d(camera, ax, color='c', tgt_steps=4, fov_steps=7, use_score_range=False, environment=None):
    """
    Draws a camera's FOV in 3D

    :param camera: Camera class instance containing camera parameters and pose
    :param ax: figure ax handle
    :param color: color code for the camera
    :param tgt_steps: number of steps over which to approximate opacity (corresponding to information density) change
     of camera
    :param fov_steps: number of steps in both horizontal and vertical fov directions. used to approximate camera view
     being limited by obstacles. Definitely more computationally expensive to have this enabled.
    :param use_score_range: (defaults to false) determines whether the camera's score range should be used over the
     default opacity values (which make the visualization a bit cleaner).
    :param environment: class instance containing environment information. if this parameter is left empty, then the
     naive visualization mode will be used (fov_steps=1, range is left unlimited)
    :return:
    """

    d_to_floor = camera.pose[2]  # assumes target is on the ground.
    d_to_ceil = 2.7 - camera.pose[2]

    if environment is None:
        fov_steps = 1
        obstacles = None
        mesh_env = False

    else:
        obstacles = environment.obstacles
        mesh_env = environment.opt_options.mesh_env
        if environment.floor is not None:
            d_to_floor = np.dot(camera.pose[:3] - environment.floor.centroid, -environment.gravity_direction)
        if environment.ceil is not None:
            d_to_ceil = np.dot(camera.pose[:3] - environment.floor.centroid, -environment.gravity_direction)

    r_steps = np.linspace(camera.range[0], camera.range[1], tgt_steps+1)
    if use_score_range:
        a_steps = np.linspace(camera.score_range[0], camera.score_range[1], tgt_steps)
    else:
        a_steps = np.linspace(0.6, 0.2, tgt_steps)

    for h in range(fov_steps):
        for v in range(fov_steps):
            # see if index is on one of the pyramid edges. if not, only draw end cap
            draw_walls = np.array([False, False, False, False, True, True])

            if h == 0:  # left side
                draw_walls[0] = True
            if h == fov_steps-1:  # right side
                draw_walls[2] = True
            if v == 0:  # bottom side
                draw_walls[3] = True
            if v == fov_steps-1:  # top side
                draw_walls[1] = True

            # check intersection of center of sub-pyramid
            sub_fov = camera.fov/(fov_steps)  # TODO: make sure things are oriented appropriately. in plot pyramid, things seem to be rotated 90

            # RECALL: CAMERA FWD IS Z, RIGHT IS X, DOWN IS Y; ROLL IS Z, PITCH IS X, YAW IS Y
            # TODO: VERIFY
            # reorient camera euler angles to match drawing function's orientation
            # (from X right, Y down, Z fwd -> X fwd, Y left, Z up)
            cam_pose = np.array([camera.pose[-1], -camera.pose[-3], -camera.pose[-2]])  # TODO VERIFY SIGN CHANGE
            R_sub = rot3d_from_rtp(cam_pose +  # start with initial camera pose
                                   np.array([0, (v+.5)*sub_fov[1], (h+.5)*sub_fov[0]]) -  # add on the sub_pose
                                   # deviation (plus a half step to get to the center of the step position)
                                   np.array([0, camera.fov[1]/2, camera.fov[0]/2]))  # and subtract off half the fov
            # R_sub = rot3d_from_rtp(camera.pose[3:] +  # start with initial camera pose
            #                        np.array([0, (v+.5)*sub_fov[1], (h+.5)*sub_fov[0]]) -  # add on the sub_pose
            #                        # deviation (plus a half step to get to the center of the step position)
            #                        np.array([0, camera.fov[1]/2, camera.fov[0]/2]))  # and subtract off half the fov
            # R_sub = rot3d_from_rtp(-camera.pose[3:] +  # start with initial camera pose
            #                        np.array([(v + .5) * sub_fov[1], (h + .5) * sub_fov[0], 0]) -  # add on the sub_pose
            #                        # deviation (plus a half step to get to the center of the step position)
            #                        np.array([camera.fov[1]/2, camera.fov[0]/2, 0]))  # and subtract off half the fov

            # find intersection...
            vec_dir = R_sub[:, 0]
            if vec_dir[-1] > 0:  # pointing upwards. look at ceiling intersection
                d = np.abs(d_to_ceil / vec_dir[-1])  # length of vector to ground in direction defined by R_sub
            else:  # pointing downwards, look at floor intersection
                d = np.abs(d_to_floor / vec_dir[-1])  # length of vector to ground in direction defined by R_sub

            d = min([camera.range[-1], d])  # take min of the ground intersection distance and the camera range
            e = d*vec_dir + camera.pose[:3]  # point representing the end of the field of view of the current sub-view

            if mesh_env:
                # obstructed, obstruction_point, _ = camera.get_mesh_obstruction(point=e,
                #                                                                clusters_remeshed=
                #                                                                environment.cluster_env_remeshed,
                #                                                                opt_options=environment.opt_options,
                #                                                                tree=environment.tree,
                #                                                                find_closest_intersection=True)
                obstructed, obstruction_point, _ = camera.get_obstruction_mesh_obb_tree(points=np.array([e]),
                                                                                        environment=environment,
                                                                                        find_closest_intersection=True)
                if obstructed[0]:
                    d = dist(obstruction_point[0], camera.pose[:3])
            elif obstacles is not None:
                obstructed, obstruction_point = camera.get_obstruction(e, obstacles)
                if obstructed:
                    d = dist(obstruction_point, camera.pose[:3])

            for i in range(tgt_steps):
                d_range = np.copy(r_steps[i:i+2])

                # Reset front and back
                draw_walls[4] = False
                draw_walls[5] = False
                if i == 0:  # draw the front surface on the starting point
                    draw_walls[4] = True

                if d_range[0] >= d:
                    break
                elif d_range[1] > d:
                    d_range[-1] = d
                    # if obstructed, or if it's the last step, draw the back wall
                    draw_walls[5] = True
                elif i == tgt_steps-1:
                    draw_walls[5] = True
                if any(draw_walls):
                    draw_pyramid_3d(position=camera.pose[:3], R=R_sub, fov=sub_fov, height_range=d_range, ax=ax,
                                    color=color, alpha=a_steps[i], draw_walls=draw_walls)


def draw_basis_3d(c, R, ax, scale=0.1):
    v_x = R[:, 0]
    v_y = R[:, 1]
    v_z = R[:, 2]

    x = np.ones([2, 3])*c
    y = np.ones([2, 3])*c
    z = np.ones([2, 3])*c

    x[1] += v_x*scale
    y[1] += v_y*scale
    z[1] += v_z*scale

    ax.plot3D(x[:, 0], x[:, 1], x[:, 2], c='r')
    ax.plot3D(y[:, 0], y[:, 1], y[:, 2], c='b')
    ax.plot3D(z[:, 0], z[:, 1], z[:, 2], c='g')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
