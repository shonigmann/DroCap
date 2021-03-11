import numpy as np

from sim.cameras import PlacedCameras

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def evaluate_arrangement(environment, cameras, placed_cameras=PlacedCameras(), debug=False):
#     """
#     DEPRECATED
#     Scores a given arrangement of cameras. Scoring is based on visibility, distance to the point and the camera's
#     respective covariance vs range characteristics,
#     :param environment: Environment Class instance containing the environment in which to place cameras
#     :param cameras: a list of Camera objects corresponding to the cameras which have already been placed in an
#     environment
#     :param placed_cameras: PlacedCameras class instance containing any previously placed cameras and their respective
#     scores
#     :param debug: flag, if true print and draw relevant information.
#     :return: returns the score associated with a given arrangement of cameras
#     """
#
#     n_placed_cameras = len(placed_cameras.cameras)
#     n_cameras = len(cameras)
#     num_points = environment.n_points
#
#     # assume best camera is first in list... use it to determine max score per point
#     max_pt_score = cameras[0].score_range[0]
#     tot_num_cams = n_cameras + n_placed_cameras
#     max_score = num_points * tot_num_cams
#
#     if np.isnan(cameras[0].pose).any():
#         # Pose should only be Nan when opt_options.map_to_flat_surface is true and the particle is not in a valid region
#         assert(environment.opt_options.map_to_flat_surface is True)
#         return max_score*2
#
#     # store point score vectors for each camera. initially, each point has 0 value
#     scores = np.zeros([num_points, n_cameras + n_placed_cameras])
#     directions = np.zeros([num_points, 3, n_cameras + n_placed_cameras])
#
#     for i in range(n_placed_cameras):
#         for j in range(n_cameras):
#             x1 = placed_cameras.cameras[i].pose[:3]
#             x2 = cameras[j].pose[:3]
#             dist_between_cams = np.linalg.norm(x1-x2)
#             if dist_between_cams < cameras[j].frame_rad + placed_cameras.cameras[i].frame_rad + \
#                     np.max(environment.opt_options.min_perch_window):
#                 # conservative perch window consideration... ideally would consider direction in a smarter way
#                 return max_score
#         scores[:, i+n_cameras] = placed_cameras.scores[i]
#         directions[:, :, i+n_cameras] = placed_cameras.directions[i]
#
#     for i in range(n_cameras):
#         # numerical approximation of the camera score integration over the target
#         # scores[:, i], directions[:, :, i] = evaluate_over_bounds(environment, cameras[i])
#         scores[:, i], directions[:, :, i] = evaluate_camera_vectorized(cameras[i], environment)  # Trying a vectorized approach
#
#     # sum over each camera
#     # scores = np.sum(scores, -1)
#     # if n_cameras+n_placed_cameras > 1:
#     #     for i in range(n_cameras):
#     #         # nested loop isn't great... but... in most applications, outer loop only runs once.
#     #         # Inner loop runs 1-3 times... should be fine
#     #         P = np.ones_like(scores[:, 0])
#     #         penalty = 1 / (n_placed_cameras + 2)  # decreasing convergence penalty as more cameras are added
#     #         for j in range(n_placed_cameras):
#     #             P *= 1-penalty*(np.sum(directions[:, :, i]*directions[:, :, n_cameras+j], axis=1)**20)
#     #             # on second thought, the approximation is perhaps closer to a 1/cos curve...
#     #         scores[:, i] *= P
#
#     # using 0.9*x*S_max for the saturation limit which should help encourage completeness of coverage.
#     # I experimented briefly with adding exponential decay (e.g 0.95^x * x * S_max), but for large number
#     # of cameras this would just saturate everything. Best to stick with linear I think.
#     # scores[scores > 0.9*tot_num_cams*max_pt_score] = 0.9*tot_num_cams*max_pt_score  # Eliminating saturation function
#
#     # score = max_score - np.sum(scores[scores > 0])  # sum values, (excluding negative values)
#     score = max_score - np.sum(scores)
#
#     if debug:
#         print("Score: " + str(score))
#
#     return score


# def evaluate_over_bounds(environment, camera):
#     """
#     DEPRECATED
#     Function that evaluates a single camera for the given environment, ignoring any other cameras.
#     :param environment: Environment Class instance containing the environment in which to place cameras
#     :param camera: a single Camera instance
#     :return: an array of the camera's scores at each integration point with dimension [N, 3]
#     where N is the number of integration points defined in Environment
#     """
#     n_points = environment.n_points
#     scores = np.zeros(n_points)  # store point score vectors for current camera
#     directions = environment.sampling_points - camera.pose[:3]
#     directions /= (np.sum(directions**2, axis=1)**.5).reshape([n_points, 1])
#
#     scores, directions = evaluate_camera_vectorized(camera, environment)
#     # idp = 0
#     # for p in environment.sampling_points:
#     #     if environment.opt_options.mesh_env:
#     #         point_score = evaluate_point_score_mesh(p, camera, environment)
#     #     else:
#     #         point_score = evaluate_point_score(p, camera)
#     #
#     #     scores[idp] = point_score
#     #     idp = idp + 1
#
#     return scores, directions


# def evaluate_point_score(point, camera, obstacles=None):
#     """
#     Evaluates a single camera's ability to capture a single point in space
#
#     :param point: the point at which you want to get the Camera's score
#     :param camera: a single Camera instance
#     :param obstacles: a list of obstacle polygons in the environment
#     :return: the heuristic score for the given camera to observe the given point
#     """
#     point_score = 0
#
#     in_range, d = camera.is_in_range(point)  # if in range of the camera
#     if in_range:
#         if camera.is_in_fov_3D_distorted(point):  # if the point is in the FOV of the camera
#             if obstacles is None or not camera.is_obstructed(point, obstacles):
#                 # inverse square score, applied to camera unit vector direction
#                 point_score = camera.get_camera_score_at_point(d)
#     return point_score

# def evaluate_point_score_mesh(point, camera, environment):
#     """
#     Evaluates a single camera's ability to capture a single point in space
#
#     :param point: the point at which you want to get the Camera's score
#     :param camera: a single Camera instance
#     :param environment: the full mesh environment instance
#
#     :return: the heuristic score for the given camera to observe the given point
#     """
#     point_score = 0
#
#     in_range, d = camera.is_in_range(point)  # if in range of the camera
#     if in_range:
#         # in_fov_simple = camera.is_in_fov_3d(point)
#         in_fov_distorted = camera.is_in_fov_3D_distorted(point)
#         # if in_fov_simple != in_fov_distorted:  # TODO: REMOVE WHEN DEBUGGED
#         #     in_fov_simple = camera.is_in_fov_3d(point)
#         #     in_fov_distorted = camera.is_in_fov_3D_distorted(point)
#
#         if in_fov_distorted:  # if the point is in the FOV of the camera
#             if environment.cluster_env is not None:
#                 obstructed, _, _ = camera.get_mesh_obstruction(point=point,
#                                                                clusters_remeshed=environment.cluster_env_remeshed,
#                                                                opt_options=environment.opt_options,
#                                                                tree=environment.tree, find_closest_intersection=False)
#                 if not obstructed:
#                     # inverse square score, applied to camera unit vector direction
#                     point_score = camera.get_camera_score_at_point(d)
#             else:
#                 point_score = camera.get_camera_score_at_point(d)
#     return point_score


# def evaluate_camera_vectorized(camera, environment):
#     """
#     DEPRECATED
#     Evaluates a single camera's ability to capture all sampling points in the environment at once
#
#     :param camera: a single Camera instance
#     :param environment: the full mesh environment instance
#
#     :return: the heuristic score for the given camera to observe the given point
#     """
#     n_points = environment.n_points
#     scores = np.nan*np.ones(n_points)
#     # covariance = np.nan*np.ones(n_points, 3, 3)
#     directions = environment.sampling_points - camera.pose[:3]
#     distances = np.sum(directions**2, axis=1) ** .5
#
#     in_view = np.logical_and(distances <= camera.range[1], distances >= camera.range[0])  # check range
#     in_view[in_view] *= camera.is_in_fov_3D_distorted(environment.sampling_points[in_view])
#
#     directions /= distances.reshape([n_points, 1])
#
#     if environment.cluster_env is not None:
#         obstructed = camera.is_obstructed_mesh_obb_tree(points=environment.sampling_points[in_view],
#                                                         environment=environment)
#
#         in_view[in_view] *= obstructed
#         scores[in_view] = camera.get_camera_score_at_point(distances[in_view])
#
#         # covariance[in_view] = camera.estimate_covariance_score(environment.sampling_points[in_view])
#
#     return scores, directions  # TODO: change score and directions to covariances...
#

def evaluate_arrangement_covariance(environment, cameras, placed_cameras=PlacedCameras(), debug=False,
                                    maximization=False):
    """
    Scores a given arrangement of cameras. Scoring is based on visibility, distance to the point and the camera's
    respective covariance vs range characteristics,
    :param environment: Environment Class instance containing the environment in which to place cameras
    :param cameras: a list of Camera objects corresponding to the cameras which have already been placed in an
    environment
    :param placed_cameras: PlacedCameras class instance containing any previously placed cameras and their respective
    scores
    :param debug: flag, if true print and draw relevant information.
    :return: returns the score associated with a given arrangement of cameras
    """

    n_placed_cameras = len(placed_cameras.cameras)
    n_cameras = len(cameras)

    if environment.noise_resistant_search:
        num_points = len(environment.noise_resist_sampling_points)
    else:
        num_points = environment.n_points

    # assume best camera is first in list... use it to determine max score per point

    max_score = num_points

    if np.isnan(cameras[0].pose).any():
        # Pose should only be Nan when opt_options.map_to_flat_surface is true and the particle is not in a valid region
        assert(environment.opt_options.map_to_flat_surface is True)
        return max_score*2

    # first check proximity to other cameras:
    for i in range(n_placed_cameras):
        for j in range(n_cameras):
            x1 = placed_cameras.cameras[i].pose[:3]
            x2 = cameras[j].pose[:3]
            dist_between_cams = np.linalg.norm(x1 - x2)
            min_dist = cameras[j].frame_rad + placed_cameras.cameras[i].frame_rad + np.max(
                environment.opt_options.min_perch_window)
            if dist_between_cams < min_dist:
                # conservative perch window consideration... ideally would consider direction in a smarter way
                return max_score

    # NOISE RESISTANCE CONFLICTS WITH THIS METHOD....
    covariances = evaluate_camera_covariance(environment=environment, cameras=cameras)
    if len(covariances) > environment.n_points:
        covariances = covariances[:environment.n_points]

    if placed_cameras.covariances is not None:
        covariance_all = np.vstack([covariances, placed_cameras.covariances])
    else:
        covariance_all = covariances
    # axes are N cameras, M points, 3 x 3; want to preserve axis1; Use this indexing to prevent all-nan slices
    visible_points = np.logical_not(np.isnan(covariance_all).all(axis=(0, 2, 3)))

    vp_debug = np.logical_not(np.isnan(covariance_all).all(axis=(2, 3)))

    # weighted least squares combination (sum of inverses, inverted)
    wls_covariance = np.linalg.inv(np.nansum(np.linalg.inv(covariance_all[:, visible_points, :, :]), axis=0))

    # divide by max covariance score to normalize
    scores = np.linalg.norm(np.linalg.eigvals(wls_covariance), axis=1) / cameras[0].max_covariance_score

    if maximization:
        score = max_score - np.sum(scores)
    else:
        # different, hopefully more efficient way of writing: N - sum(1-c/c_max); as there is only one matrix operation
        score = np.sum(scores)
    # print(score)
    if debug:
        print("Score: " + str(score))

    return score


def evaluate_camera_covariance(cameras, environment, pts=None, plot_visible=False):
    """
    Evaluates a single camera's ability to capture all sampling points in the environment at once

    :param camera: a single Camera instance
    :param environment: the full mesh environment instance

    :return: the heuristic score for the given camera to observe the given point
    """
    N = len(cameras)

    if pts is None:
        if environment.noise_resistant_search:
            pts = environment.noise_resist_sampling_points
        else:
            pts = environment.sampling_points
    n_points = len(pts)

    covariances = np.nan * np.ones([N, n_points, 3, 3])
    for i in range(N):
        directions = pts - cameras[i].pose[:3]
        distances = np.sum(directions ** 2, axis=1) ** .5

        in_view = np.logical_and(distances <= cameras[i].range[1], distances >= cameras[i].range[0])  # check range

        # test1 = cameras[i].is_in_fov_3D_distorted(pts[in_view])
        # test2 = cameras[i].is_in_fov_3d(pts[in_view])

        in_view[in_view] *= cameras[i].is_in_fov_3D_distorted(pts[in_view])

        directions /= distances.reshape([n_points, 1])

        if environment.cluster_env is not None:
            obstructed = cameras[i].is_obstructed_mesh_obb_tree(points=pts[in_view],
                                                            environment=environment)

            in_view[in_view] *= np.logical_not(obstructed)
            # scores[in_view] = camera.get_camera_score_at_point(distances[in_view])

        covariances[i, in_view] = cameras[i].estimate_covariance(pts[in_view])

        # DEBUGGING:
        if plot_visible:
            x, y, z = pts[in_view].T
            fig = plt.figure()
            ax = Axes3D(fig)

            ax.scatter3D(x, y, z)
            x, y, z = pts[np.logical_not(in_view)].T
            ax.scatter3D(x, y, z, c='r', marker='x')

            plt.waitforbuttonpress(5)

    return covariances


def evaluate_discrete_coverage(num_points, placed_cameras, plot=False):
    covariances = placed_cameras.covariances
    visibility_per_num_cams = np.sum(np.logical_not(np.isnan(covariances).all(axis=(2, 3))), axis=0)
    n = len(placed_cameras.cameras) + 1
    pct = np.zeros(n)
    for i in range(n, -1, -1):
        pct[i] = np.sum(visibility_per_num_cams == i) / num_points * 100.0
        print(str(pct) + "% of target volume is covered by " + str(i) + " camera(s)")

    if plot:
        cum_sums = [np.sum(pct[i:]) for i in range(1, n+1)]
        plt.plot(np.arange(1, n+1), cum_sums)
        plt.ylabel('Percent Coverage')
        plt.xlabel('Minimum Number of Cameras')
        plt.title('Multi-Camera Coverage')

    return pct
