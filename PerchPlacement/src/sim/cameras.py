import numpy as np
import pymesh
import trimesh
from shapely.geometry import LineString, LinearRing
from geom.tools import get_inv_sqr_coefficients
from geom.geometry3d import dist, generate_fov_end_mesh, rot3d_from_rtp, rot3d_from_x_vec, rot3d_from_z_vec
import open3d as o3d
import cv2 as cv
# import vtk  # TRY TO REMOVE AND REPLACE WITH VEDO
import vedo

from pyswarms.utils.reporter.reporter import Reporter
from vis.draw3d import draw_basis_3d
from tools.tools import set_axes_equal

from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Camera:
    """
    Class containing the properties of a simple simulated 3D pinhole camera
    """
    def __init__(self, fov=np.array([90, 57]), camera_range=np.array([0.2, 10.0]), score_range=np.array([1.0, 0.1]),
                 pose=np.zeros(6), camera_offset=np.array([0, 0, 0.05]), camera_rotation=np.eye(3), max_zoom=1,
                 gimbal_limit=np.array([0, 0]), wall_normal=np.array([1, 0, 0]), frame_rad=0.33, prop_rad=0.05,
                 camera_id=0, intrinsic=None, distortion=np.zeros([5]), resolution=np.array([1920, 1080]),
                 sensor_horizontal=0.005, camera_baseline=0.05, camera_subpixel=0.08):
        """
        Camera initialization function. All parameters are optional.

        :param fov: a 2D np.array containing the [horizontal fov, vertical fov] of the camera
        :param camera_range: a 2D np.array containing [min_dist, max_dist] from the camera for which
        usable data can be generated
        :param score_range: a 2D np.array containing the score assigned to a point when a point is at the min or max
        values of the camera range. An inverse square function is fit between these points. These function is also used
        as the opacity when visualizing the camera's FOV
        :param pose: a 6D np.array containing the camera's 3D cartesian and rotational pose [x,y,z,roll,pitch,yaw]
        :param camera_offset: a 3D np.array containing values corresponding with the camera's translational offset from
        the center of the contact face of the drone's perching mechanism.
        :param camera_rotation: a 3x3 np.array representing the rotation between the drone's coordinate frame and the
        camera's coordinate frame. In this simulated environment, the camera's Z direction is in the range axis. X and Y
        are the right horizontal and down vertical FOVs respectively (directions assigned in increasing pixel order).
        The drone is assumed to perch by pitching (+90deg about x). By default, the camera is assumed to be aligned with
        the drone's coordinate frame (when perching, x axis is the horizon, y axis is the vertical)
        :param max_zoom: if defined, this value determines the maximum zoom factor possible for the camera. Zoom can be
        varied during the opt procedure.
        :param gimbal_limit: a 2D np.array containing the half-range of the tilt and pan axes of the gimbal such that
        the possible angles can be expressed on the range [-gimbal_limit[i], gimbal_limit[i]].
        E.g. for a gimbal with +/- 30 degrees tilt and continuous pan, [30, 180]
        :param wall_normal: a 3D np.array containing the normal direction vector of the wall on which the camera is
        placed
        :param camera_id: int serving to identify a given camera/drone
        :param intrinsic: 3x3 np.array containing the camera intrinsics, in the form [[fx, t, cx][0, fy, cy][0, 0, 1]]
         where fx and fy are the x and y axis focal lengths, cx and cy are the optical centers, and t is the skew
         coefficient
        :param distortion: 1x5 np.array containing [k1,k2,p1,p2,k3] where k_i is the r^(2*i) radial distortion
         coefficient, and p1 and p2 are the x and y-axis tangential distortion coefficients respectively. The order
         follows the opencv standard.
        :param resolution: the camera's horizontal and vertical resolution
        """
        self.pose = pose
        self.surf_id = -1
        self.wall_normal = wall_normal
        self.base_fov = fov  # store the original value in here
        self.fov = fov  # store the current fov given the current zoom value
        self.range = camera_range
        self.score_range = score_range  # DEPRECATED
        self.b, self.n = get_inv_sqr_coefficients(self.range, self.score_range)
        self.max_zoom = max_zoom

        self.camera_offset = camera_offset
        self.camera_rotation = camera_rotation

        self.gimbal_limit = gimbal_limit
        self.prop_rad = prop_rad
        self.frame_rad = frame_rad
        self.camera_id = camera_id

        self.resolution = resolution
        self.sensor_horizontal = sensor_horizontal

        if intrinsic is not None:
            self.K = intrinsic
        else:
            f = self.resolution[0]/2/np.tan(fov[0]/2*np.pi/180)
            self.K = np.array([[f, 0, resolution[0]/2-.5],
                               [0, f, resolution[1]/2-.5],
                               [0, 0, 1]])

        self.k1 = distortion[0]  # r^2 radial distortion coefficient
        self.k2 = distortion[1]  # r^4 radial distortion coefficient
        self.k3 = distortion[4]  # r^6 radial distortion coefficient
        self.p1 = distortion[2]  # x axis tangential distortion coefficient
        self.p2 = distortion[3]  # y axis tangential distortion coefficient

        self.subpixel = camera_subpixel
        self.baseline = camera_baseline

        self.max_covariance_score = self.estimate_covariance_score(np.array([self.pose[:3] +
                                                                             np.array([self.range[1], 0, 0])]))

    def estimate_2cam_variance(self, L, alpha):
        # f = self.K[0, 0] * C  # want focal length in m
        # d_p =   # pixel size, m
        # sigma =   # either standard deviation of the camera, or matching error coefficient...
        # alpha =   # convergence angle between cameras
        # Ux = (L-f)*d_p*sigma/(f*np.cos(alpha/2))  # from file:///home/simon/Downloads/ao-58-24-6535.pdf, eq 19
        # Uy = (L-f)*d_p*sigma/f
        # Uz = (L-f)*d_p*sigma/(f*np.sin(alpha/2))
        return -1

    def estimate_matching_error(self, pts):
        v = pts - self.pose[:3]
        r = np.linalg.norm(v, axis=-1)

        t_min = self.get_minimum_feature_size_at_point(r) / 2  # tangential error estimate
        r_min = self.get_depth_uncertainty_estimate(r)  # radial/axial error estimate
        if len(pts.shape) > 1:
            pt_dir = np.einsum('ij,i->ij', v, 1/r)
        else:
            pt_dir = v/r
        R_dir = rot3d_from_x_vec(pt_dir)

        return R_dir, r_min, t_min

    def estimate_covariance(self, pts):
        R_dir, r_min, t_min = self.estimate_matching_error(pts)
        # C_cam = np.diag([t_min, t_min, r_min])
        a = np.array([r_min, t_min, t_min]).transpose()
        m, n = a.shape
        C_cam = np.zeros((m, n, n), dtype=a.dtype)
        C_cam.reshape(-1, n ** 2)[..., ::n + 1] = a

        # C_world = R_dir.dot(C_cam.dot(R_dir.transpose()))
        C_world = np.einsum('mij,mjk,mkl->mil', R_dir, C_cam, R_dir.transpose(0, 2, 1))
        return C_world

    def estimate_covariance_score(self, pts):
        C_world = self.estimate_covariance(pts)
        # want C_world to be [Nx3x3]
        if np.isnan(C_world).any():
            C_norm = C_world
        else:
            C_norm = np.linalg.norm(np.linalg.eigvals(C_world), axis=1)
        return C_norm

    def plot_covariance_ellipsoids_at_points(self, pts, ax=None, color='b', show=False):
        C_dir = self.estimate_covariance(pts)

        for i in range(len(pts)):
            self.plot_matching_error_ellipsoid(pts[i], C_dir=C_dir[i], ax=ax,
                                               color=color, show=show)

    @staticmethod
    def plot_matching_error_ellipsoid(pt, C_dir, ax, color='b', show=False):
        if ax is None:
            fig = plt.figure(1, figsize=plt.figaspect(1))  # Square figure
            ax = fig.add_subplot(111, projection='3d')
        C_dir = C_dir*130
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        pts = np.array([x, y, z])
        for i in range(pts.shape[-1]):
            pts[:, :, i] = (np.dot(C_dir, pts[:, :, i]).transpose() + pt).transpose()

        ax.plot_surface(pts[0], pts[1], pts[2], rstride=4, cstride=4, color=color, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        set_axes_equal(ax)
        if show:
            plt.show()

    def get_depth_uncertainty_estimate(self, r):
        focal_length = self.K[0, 0]
        return ((r ** 2) * self.subpixel) / (focal_length * self.baseline)

    def get_pixel_area_at_point(self, r):
        px_h = self.fov[0]/self.resolution[0] * np.pi/180  # horizontal angle encompassed by one pixel, in radians
        px_v = self.fov[1]/self.resolution[1] * np.pi/180
        px_A = r**2 * px_h * px_v

        return px_A  # m^2 / pixel

    def get_minimum_feature_size_at_point(self, r):
        A_px = self.get_pixel_area_at_point(r)
        r_min = np.sqrt(A_px/np.pi)
        return 2*r_min

    def get_camera_score_at_point(self, r):
        """
        This function computes the score of a point at some radial distance r based on the specified score range and
        camera range. Currently, this is based on the theoretical Depth RMS error limit curve for stereo cameras.
        TODO: implement noise models for different cameras.

        :param r: distance from the camera
        :return score: the score assigned to any point at a radial distance r from the camera. Points outside the
        specified min and max range of the camera receive a score of 0
        """
        # if self.range[0] <= r <= self.range[1]:  # removing: already checked in evaluate.py
        # in m, from http://robotsforroboticists.com/wordpress/wp-content/uploads/2019/09/realsense-sep-2019.pdf
        RMS = self.get_depth_uncertainty_estimate(r)
        # min_RMS = self.get_depth_uncertainty_estimate(self.range[0])
        max_RMS = self.get_depth_uncertainty_estimate(self.range[1])
        score = 1-RMS/max_RMS/4  # experimenting with the max penalty being 25% at the end of the range
        # TODO: figure out reasonable way to combine this uncertainty with convergence uncertainty
        # TODO: figure out reasonable way to factor in uncertainty in both depth and in image plane
        # else:
        #     score = 0
        return score

    def get_convergent_angle(self, cam2):
        """
        DEPRECATED: returns the convergence angle between two cameras. only implemented in 2D. No longer used.

        :param cam2: Camera 2
        :return:
        """
        return (self.pose[-1] - cam2.pose[-1] + 360) % 360

    def estimate_convergence_penalty(self, camera_poses, camera_number, debug=False):
        """
        DEPRECATED. Returns a penalty coefficient to punish cameras for having too similar an angle. Returns a penalty
        for one camera, given all previously placed cameras.

        :param camera_poses: list of camera poses
        :param camera_number: which camera is of interest
        :param debug: if true, print out debug information
        :return:
        """

        convergence_penalty = 1
        for j in range(camera_number):
            # get difference in angle between current camera, i, and previous cameras, j, in rads
            dth = (self.pose[-1] - camera_poses[j, -1]) * np.pi / 180

            # use sine function to prioritize ~90deg convergence angles
            convergence_penalty = convergence_penalty * np.abs(np.sin(dth))

        if debug:
            print("Convergence Multiplier for camera " + str(camera_number) + ": " +
                  str(convergence_penalty))

        return convergence_penalty

    def is_in_fov_2d(self, point):
        """
        Determines whether or not a given point is in the camera's 2D (horizontal) FOV
        :param point: target point coordinates, in 2D
        :return: boolean, True if in FOV
        """

        # assume th is on [-180, 180]
        # arc tan2 returns on [-pi pi]
        dp = point - self.pose[0:2]
        angle = (np.arctan2(dp[1], dp[0]) * 180 / np.pi + 360) % 360
        cam_angle = (self.pose[-1] + 360) % 360
        if np.abs(angle - cam_angle) < self.fov[-1] / 2:
            in_fov = True
        else:
            in_fov = False
        return in_fov

    def is_in_fov_3d(self, points):
        """
        Determines whether a 3D point is within the 3D FOV of a camera.
        Recall: the Z axis is the cameras range axis. X is horizontal (pitch), Y is vertical (yaw)

        :param points: Nx3 np.array corresponding with the target point coordinates
        :return: boolean, true if point is in FOV
        """
        in_fov = np.ones(len(points), dtype=bool)

        dp = points - self.pose[0:3]  # vector between points
        dp = np.einsum('ij,i->ji', dp, 1/np.linalg.norm(dp, axis=-1))  # normalized vector in direction of segment between points

        # convert vector to pan and tilt, convert to degrees, then shift range to [0, 360)
        pans = (np.arctan2(dp[1], dp[0]) * 180 / np.pi + 360) % 360
        cam_pan = (-self.pose[-2] + 360) % 360
        min_pan_diff = pans - cam_pan
        min_pan_diff[min_pan_diff > 180] -= 360
        min_pan_diff[min_pan_diff < -180] += 360

        in_fov *= np.abs(min_pan_diff) < self.fov[0] / 2

        tilt = (np.arcsin(dp[2]) * 180 / np.pi + 360) % 360  # TODO: verify signs
        cam_tilt = (-self.pose[-3] + 360) % 360
        min_tilt_diff = tilt - cam_tilt
        min_tilt_diff[min_tilt_diff > 180] -= 360
        min_tilt_diff[min_tilt_diff < -180] += 360
        in_fov *= np.abs(min_pan_diff) < self.fov[1] / 2

        return in_fov

    def is_in_range(self, point):
        """
        Determine if point is in camera's range

        :param point: target point
        :return: true if in range
        """
        d = dist(point, self.pose[:len(point)])
        in_range = self.range[0] <= d <= self.range[1]
        return in_range, d  # return true if in range of the camera; also return distance

    def is_obstructed(self, point, obstructions):
        """
        Returns whether any obstructions exist between the specified point and the camera.

        :param point: target point
        :param obstructions: list of all obstruction polygons.
        :return: boolean, True if point is obstructed
        """
        dim = len(point)
        obstructed = False
        intersection_point = np.ones(dim)*np.inf

        # Highly simplified computation, as checking intersection with a mesh would be computationally costly (e.g.
        # O(n^3) for naive approach). Starting with 2D representation. IF there is a 2D intersection, find Z coord of
        # intersection and compare with obstacle height
        line = LineString(np.array([point, self.pose[:dim]]))
        for obs in obstructions:
            obs_poly = LinearRing(obs)

            if obs_poly.intersects(line):
                if dim == 2:
                    return obstructed
                else:
                    z_range = np.array([np.min(obs[:, -1]), np.max(obs[:, -1])])
                    intersections = obs_poly.intersection(line)
                    if intersections.type == 'Point':
                        # assuming obs_poly has uniform height range.... this will likely need to change when moving to
                        # a mesh anyways...
                        obstructed, _ = self.check_vertical_intersection(intersections, intersection_point, z_range)
                        if obstructed:
                            return obstructed

                    else:  # if not a point, it is a collection of points. check each
                        for intersection in intersections:
                            obstructed, _ = self.check_vertical_intersection(intersection, intersection_point, z_range)
                            if obstructed:
                                return obstructed

        return obstructed

    def get_obstruction(self, point, obstructions):
        """
        Returns whether any obstructions exist between the specified point and the camera. CURRENTLY ONLY A 2D FUNCTION

        :param point: target point
        :param obstructions: list of all obstruction polygons.
        :return: boolean, True if point is obstructed, and the nearest obstruction point
        """
        dim = len(point)
        obstructed = False
        intersection_point = np.ones(dim) * np.inf

        # Highly simplified computation, as checking intersection with a mesh would be computationally costly (e.g.
        # O(n^3) for naive approach). Starting with 2D representation. IF there is a 2D intersection, find Z coord of
        # intersection and compare with obstacle height
        line = LineString(np.array([point, self.pose[:dim]]))
        for obs in obstructions:
            obs_poly = LinearRing(obs)

            if obs_poly.intersects(line):
                intersections = obs_poly.intersection(line)
                if intersections.type == 'Point':
                    # assuming obs_poly has uniform height range.... this will likely need to change when moving to a
                    # mesh anyways...
                    if dim == 3:
                        z_range = np.array([np.min(obs[:, -1]), np.max(obs[:, -1])])
                        obstructed, intersection_point = self.check_vertical_intersection(intersections,
                                                                                          intersection_point, z_range)
                    else:
                        obstructed = True
                        intersection_point = np.asarray(intersections.coords)

                else:  # if not a point, it is a collection of points. check each
                    for intersection in intersections:
                        if dim == 3:
                            z_range = np.array([np.min(obs[:, -1]), np.max(obs[:, -1])])
                            obstructed, intersection_point = self.check_vertical_intersection(intersection,
                                                                                              intersection_point,
                                                                                              z_range)
                        else:
                            intersection_point = np.squeeze(np.asarray(intersection.coords))

        return obstructed, intersection_point

    def get_mesh_obstruction(self, point, clusters_remeshed, opt_options, tree, find_closest_intersection=False,
                             search_radius=None):
        # DEPRECATED
        # TODO: two possible improvements:
        #  1. using iterative bounding box complexity to quickly rule out obstacles that aren't even close. If the line
        #     of sight crosses a bounding box, then look at specific faces of that cluster
        #  2. Represent all obstructions as convex hulls, then can use GJK algorithm
        #     https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm
        #  See also: http://what-when-how.com/advanced-methods-in-computer-graphics/collision-detection-advanced-methods-in-computer-graphics-part-2/
        obstructed = False  # TODO: vectorize...
        obstructing_face = -1
        closest_intersection = point
        all_intersections = []

        # using Trumbore Intersection Algorithm on all triangles near points along the ray
        O = self.pose[:3].copy()
        D = point - O
        d = np.linalg.norm(D)

        # in pre-processing, the average of the maximum radii required to circumscribe at least one point in the
        # triangle is taken over all faces in the mesh. perhaps a bit unnecessary... consider simplifying with something
        # faster to compute
        if search_radius is None:
            search_radius = opt_options.dist_threshold
            # TODO: consider re-meshing cluster env to be certain that smaller search radius will work. Will result in
            #  more points but could still result in a speed increase by/ points on surfaces will scale with L^2 while
            #  search volume scales with L^3

        if d > search_radius:
            pts = np.arange(0, d, search_radius)
            pts = pts.reshape([len(pts), 1]) * D / d + O
        elif search_radius >= d/2:
            pts = np.array([(O+D)/2])  # if the radius is large enough, just search one large sphere from the center
        else:
            # if the radius isn't quite that large, put one sphere at the origin and one at the point
            pts = np.array([O, D])

        fids = np.zeros([0, 1], dtype=int)
        fid = tree.query_radius(pts, search_radius)
        #
        # fig = plt.figure(1)
        # ax = Axes3D(fig)

        #TODO  CONSIDER QUICK FILTERING OF REMAINING POINTS BASED ON CAMERA FRUSTUM

        for f in fid:
            fids = np.vstack([fids, f.reshape([-1, 1])])
            # if f.size > 0:
            #     for x in f:
            #         fids.append(x)

        if len(fids) > 0:
            idx = np.unique(fids.squeeze())  # list of all unique faces whose centers are within the radius

            min_dist = np.finfo(np.float32).max

            for fid in idx:  # TODO: try to eliminate this loop...
                A = clusters_remeshed.points[clusters_remeshed.faces[fid, 0]]
                B = clusters_remeshed.points[clusters_remeshed.faces[fid, 1]]
                C = clusters_remeshed.points[clusters_remeshed.faces[fid, 2]]
                e1 = B-A
                e2 = C-A
                N = np.cross(e1, e2)
                det = -np.dot(D, N)
                inv_det = 1.0/det
                AO = O - A
                DAO = np.cross(AO, D)
                u = np.dot(DAO, e2) * inv_det
                v = -np.dot(DAO, e1) * inv_det
                t = np.dot(AO, N) * inv_det
                if det >= 1e-6 and 0.0 <= t <= 1.0 and u >= 0.0 and v >= 0.0 and (u + v) <= 1.0:
                    obstructed = True
                    all_intersections.append(fid)

                    if find_closest_intersection:
                        if min_dist > t:
                            min_dist = t
                            closest_intersection = O+np.dot(t, D)

                        # l = np.array([O, point])
                        # ax.plot3D(l[:, 0], l[:, 1], l[:, 2])
                        # tri = np.array([A, B, C, A])
                        # ax.plot3D(tri[:, 0], tri[:, 1], tri[:, 2])
                        # plt.waitforbuttonpress()

                    else:
                        return True, O+np.dot(t, D), all_intersections
        # if len(all_intersections) > 0:
        #     print(all_intersections)

        return obstructed, closest_intersection, all_intersections

    # def is_obstructed_mesh_vectorized(self, points, environment):
    #     # NOT COMPLETE: REVISIT IF NECESSARY
    #     if points.shape[0] < 1:
    #         return False
    #
    #     obstructed = np.zeros([points.shape[0]], dtype=bool)
    #
    #     # vectorized approach... rather than checking along rays, check in minimum radius containing points + origin
    #     O = self.pose[:3]
    #     D = points - O
    #     dists = np.linalg.norm(D, axis=1)
    #     d = np.argmin(dists)
    #     pt = np.array([(O+D[d])/2])
    #     search_radius = dists[d]/2
    #
    #     fid = environment.obb_tree.query_radius(pt, search_radius)
    #     fids = fid[0]
    #
    #     # TODO: implement oriented bounding boxes...
    #     # TODO  CONSIDER QUICK FILTERING OF REMAINING POINTS BASED ON CAMERA FRUSTUM
    #
    #     if len(fids) > 0:
    #         idx = np.unique(fids)  # list of all unique faces whose centers are within the radius
    #
    #         min_dist = np.finfo(np.float32).max
    #
    #         for fid in idx:  # TODO: try to eliminate this loop...
    #             continue
    #
    #
    #     return obstructed

    def is_obstructed_mesh_obb_tree(self, points, environment):
        obstructed = np.zeros(len(points), dtype=bool)
        for p in range(len(points)):
            p1 = (self.pose[0], self.pose[1], self.pose[2])
            p2 = (points[p, 0], points[p, 1], points[p, 2])

            if environment.vedo_mesh is None:
                vedo_mesh = vedo.mesh.Mesh(environment.obs_mesh)
                environment.vedo_mesh = vedo_mesh
            else:
                vedo_mesh = environment.vedo_mesh

            intersection_points = vedo_mesh.intersectWithLine(p1, p2)
            for pt in intersection_points:
                pt = np.asarray(pt)
                d = dist(self.pose[:3], pt)
                s = np.dot(pt-self.pose[:3], points[p]-self.pose[:3])
                # used to quickly check that the intersection is actually in front of the camera...
                # shouldnt need this if all works as expected

                if d <= self.range[1] and s > 0:
                    obstructed[p] = True
                    break

            # pointsVTKintersection = vtk.vtkPoints()
            # code = environment.caster.IntersectWithLine(self.pose[:3], points[p], pointsVTKintersection, None)
            # # if code == -1 or code == 0:
            # #     obstructed[p] = False  # already false...
            # if code != -1 and code != 0:
            #     # obstructed[p] = True
            #     pointsVTKIntersectionData = pointsVTKintersection.GetData()
            #     # get the number of tuples
            #     noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
            #     # ensure intersection points are within range
            #     for idx in range(noPointsVTKIntersection):
            #         _tup = pointsVTKIntersectionData.GetTuple3(idx)
            #         if self.range[0] <= np.linalg.norm(self.pose[:3] - np.asarray(_tup)) <= self.range[1]:
            #             obstructed[p] = True
            #             break
        return obstructed

    def get_obstruction_mesh_obb_tree(self, points, environment, find_closest_intersection=False):
        N = len(points)
        obstructed = np.zeros(N, dtype=bool)  # initialize as false for all points
        closest_intersection = points.copy()
        all_intersections = [[] for i in range(N)]
        min_dist = np.finfo(float).max

        for p in range(N):
            p1 = (self.pose[0], self.pose[1], self.pose[2])
            p2 = (points[p, 0], points[p, 1], points[p, 2])

            if environment.vedo_mesh is None:
                vedo_mesh = vedo.mesh.Mesh(environment.obs_mesh)
            else:
                vedo_mesh = environment.vedo_mesh

            intersection_points = vedo_mesh.intersectWithLine(p1, p2)
            for pt in intersection_points:
                pt = np.asarray(pt)
                d = dist(self.pose[:3], pt)
                s = np.dot(pt - self.pose[:3], points[p] - self.pose[:3])
                # used to quickly check that the intersection is actually in front of the camera...
                # shouldnt need this if all works as expected

                if d <= self.range[1] and s > 0:
                    obstructed[p] = True
                    all_intersections[p].append(pt)
                    if find_closest_intersection:
                        closest_intersection[p] = pt
                        break
                    else:
                        if d < min_dist:
                            min_dist = d
                            closest_intersection[p] = pt
            # pointsVTKintersection = vtk.vtkPoints()
            # code = environment.caster.IntersectWithLine(self.pose[:3], points[p], pointsVTKintersection, None)
            # if code != -1 and code != 0:
            #     pointsVTKIntersectionData = pointsVTKintersection.GetData()
            #     # get the number of tuples
            #     numPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
            #     # ensure intersection points are within range
            #     for idx in range(numPointsVTKIntersection):
            #         _tup = np.asarray(pointsVTKIntersectionData.GetTuple3(idx))
            #
            #         d = np.linalg.norm(self.pose[:3] - _tup)
            #         dir = np.dot()
            #         if self.range[0] <= d <= self.range[1]:
            #             obstructed[p] = True
            #             all_intersections[p].append(_tup)
            #             if not find_closest_intersection:
            #                 break
            #             else:
            #                 if d < min_dist:
            #                     closest_intersection[p] = _tup

        return obstructed, closest_intersection, all_intersections

    def is_obstructed_mesh(self, point, clusters_remeshed, opt_options, tree, search_radius=None):
        obstructed, _, _ = self.get_mesh_obstruction(point, clusters_remeshed, opt_options, tree,
                                                     find_closest_intersection=True, search_radius=search_radius)
        return obstructed

    def get_mesh_fov_obstructions(self, env):
        # consider all points here. should only run once per camera.
        face_centers = env.cluster_env.face_centers
        face_ids = np.arange(len(face_centers))

        # only care about dists < range[1]
        dists = np.sum((face_centers-self.pose[:3])**2, axis=1)**0.5

        R_cam = rot3d_from_rtp(self.pose[3:])
        cam_x = R_cam[:, 0]  # camera right
        cam_y = R_cam[:, 1]  # camera down
        cam_z = R_cam[:, 2]  # camera fwd

        # find direction of the center of each fov edge (e.g. top-center, bottom-center, right-center, left-center)
        R_u = rot3d_from_rtp(np.array([0, self.fov[1]/2, 0]))
        R_b = rot3d_from_rtp(np.array([0, -self.fov[1]/2, 0]))
        R_l = rot3d_from_rtp(np.array([0, 0, self.fov[0]/2]))
        R_r = rot3d_from_rtp(np.array([0, 0, -self.fov[0]/2]))

        tc = np.dot(R_u, cam_z)
        bc = np.dot(R_b, cam_z)
        lc = np.dot(R_l, cam_z)
        rc = np.dot(R_r, cam_z)

        # find normals of the planes corresponding with the FOV boundaries
        P_T = np.cross(cam_y, tc)
        P_B = np.cross(bc, cam_y)
        P_L = np.cross(lc, cam_x)
        P_R = np.cross(cam_x, rc)

        # ensure all normals are in the correct direction...
        # shouldn't be needed if vector order in cross product is correct
        if np.dot(cam_x, P_T) <= 0:
            P_T *= -1
        if np.dot(cam_x, P_B) <= 0:
            P_B *= -1
        if np.dot(cam_x, P_L) <= 0:
            P_L *= -1
        if np.dot(cam_x, P_R) <= 0:
            P_R *= -1

        # only keep points within range
        in_range = (dists <= self.range[1]) * (self.range[0] <= dists)
        obstructions = face_centers[in_range]
        face_ids = face_ids[in_range]

        obstructions -= self.pose[:3]  # subtract off camera origin

        # only keep points within cone
        in_fov = np.dot(obstructions, P_T) >= 0
        obstructions = obstructions[in_fov]
        face_ids = face_ids[in_fov]

        in_fov = np.dot(obstructions, P_B) >= 0
        obstructions = obstructions[in_fov]
        face_ids = face_ids[in_fov]

        in_fov = np.dot(obstructions, P_R) >= 0
        obstructions = obstructions[in_fov]
        face_ids = face_ids[in_fov]

        in_fov = np.dot(obstructions, P_L) >= 0
        # obstructions = obstructions[in_fov]
        # obstructions += self.pose[:3]  # read camera origin
        face_ids = face_ids[in_fov]

        # TODO: keeping connected faces together could speed things up quite a bit

        return face_ids

    def check_vertical_intersection(self, intersection, prev_intersection_point, z_range):
        """
        DEPRECATED

        Simple check to see if a line crosses a rectangular polygon in 3D
        :param intersection: new intersection point between x-y line and x-y polygon
        :param prev_intersection_point: prior intersection point, used to see which is closest to camera. set to np.inf
         if unset
        :param z_range: range of z values of the polygon
        :return: boolean, true if obstructed, and the closest intersection point
        """
        intersection_point = prev_intersection_point
        if intersection_point[0] == np.inf:
            obstructed = False
        else:
            obstructed = True

        if z_range[0] <= intersection.z <= z_range[1]:
            obstructed = True
            intersection_point_ = np.squeeze(np.asarray(intersection.coords))
            if prev_intersection_point[0] == np.inf:  # if not set, take coordinates
                intersection_point = intersection_point_
            elif dist(prev_intersection_point, self.pose[:3]) > dist(intersection_point_, self.pose[:3]):
                # if already set, compare and take closest
                intersection_point = intersection_point_

        return obstructed, intersection_point

    def generate_camera_mesh(self, num_curve_points, environment=None, inwards=True):
        """

        """

        O = self.pose[:3]
        r_min = self.range[0]
        r_max = self.range[1]

        R = rot3d_from_rtp(self.pose[3:])
        fov = self.fov

        # generate near and far sphere ends...
        if inwards:
            n1 = "in"
            n2 = "out"
        else:
            n1 = "out"
            n2 = "in"
        pts1, verts1, faces1 = generate_fov_end_mesh(O=O, r=r_min, R=R, fov=fov, num_curve_points=num_curve_points,
                                                     normal=n1)
        pts2, verts2, faces2 = generate_fov_end_mesh(O=O, r=r_max, R=R, fov=fov, num_curve_points=num_curve_points,
                                                     normal=n2)

        # generate top, bottom, left, and right faces...
        tpts = np.zeros([num_curve_points * 2, 3])
        bpts = np.zeros([num_curve_points * 2, 3])
        lpts = np.zeros([num_curve_points * 2, 3])
        rpts = np.zeros([num_curve_points * 2, 3])
        outside_faces = []
        inside_faces = []

        for i in range(num_curve_points):
            tpts[i, :] = pts1[i, 0]
            tpts[i + num_curve_points, :] = pts2[i, 0]

            bpts[i, :] = pts1[i, -1]
            bpts[i + num_curve_points, :] = pts2[i, -1]

            lpts[i, :] = pts1[0, i]
            lpts[i + num_curve_points, :] = pts2[0, i]

            rpts[i, :] = pts1[-1, i]
            rpts[i + num_curve_points, :] = pts2[-1, i]

            if i < num_curve_points - 1:
                outside_faces.append(np.array([i, i + 1, 1 + i + num_curve_points]))
                outside_faces.append(np.array([i, i + 1 + num_curve_points, i + num_curve_points]))

                inside_faces.append(np.array([i + 1, i, 1 + i + num_curve_points]))
                inside_faces.append(np.array([i + 1 + num_curve_points, i, i + num_curve_points]))

        inside_faces = np.asarray(inside_faces)
        outside_faces = np.asarray(outside_faces)

        if inwards:
            tfs = outside_faces.copy()
            bfs = inside_faces.copy()
            lfs = inside_faces.copy()
            rfs = outside_faces.copy()
        else:
            tfs = inside_faces.copy()
            bfs = outside_faces.copy()
            lfs = outside_faces.copy()
            rfs = inside_faces.copy()

        i1 = 0
        i2 = i1 + len(verts1)
        i3 = i2 + len(verts2)
        i4 = i3 + len(tpts)
        i5 = i4 + len(bpts)
        i6 = i5 + len(lpts)

        vs = np.vstack([verts1, verts2, tpts, bpts, lpts, rpts])
        fs = np.vstack([faces1, faces2 + i2, tfs + i3, bfs + i4, lfs + i5, rfs + i6])

        cam_mesh = pymesh.form_mesh(vs, fs)

        if environment is not None:
            obs_faces = self.get_mesh_fov_obstructions(env=environment)

            count = 0
            obs_meshes = []
            for f in obs_faces:
                print("creating obstacle mesh #: " + str(count))
                count += 1
                obs_pts = environment.cluster_env.points[environment.cluster_env.faces[f], :]
                obs_meshes.append(self.generate_obstruction_prism(obs_pts))

            while len(obs_meshes) > 1:
                print("# of obstacles = " + str(len(obs_meshes)))
                for i in range(len(obs_meshes)//2):
                    obs_meshes[i] = pymesh.boolean(obs_meshes[i], obs_meshes[i+1], "union")

                    pymesh.save_mesh("_temp_.ply", obs_meshes[i])
                    temp = o3d.io.read_triangle_mesh("_temp_.ply")
                    o3d.visualization.draw_geometries([temp])

                    _ = obs_meshes.pop(i+1)
            pymesh.save_mesh("Obstructions_Merged.ply", obs_meshes[0])  # TODO: consider simplifying meshes after a certain number of steps
            # now merge with camera
            print("Subtracting Obstacles from Camera Mesh")
            cam_mesh = pymesh.boolean(cam_mesh, obs_meshes[0], "difference")

            pymesh.save_mesh("CameraLessObstructions.ply", cam_mesh)
            # obs_mesh = None
            # for f in obs_faces:
            #     print("Obstacle Mesh Processing: " + str(count) + " of " + str(len(obs_faces)))
            #     count += 1
            #     obs_pts = environment.cluster_env.points[environment.cluster_env.faces[f], :]
            #     # obs_mesh = self.generate_obstruction_prism(obs_pts)
            #     # cam_mesh = pymesh.boolean(cam_mesh, obs_mesh, "difference")
            #     # obs_pts_ = np.array([obs_pts[1], obs_pts[0], obs_pts[2]])
            #     # obs_mesh_ = self.generate_obstruction_prism(obs_pts_)
            #     # cam_mesh = pymesh.boolean(cam_mesh, obs_mesh_, "difference")
            #     if obs_mesh is None:
            #         obs_mesh = self.generate_obstruction_prism(obs_pts)
            #     else:
            #         obs_mesh_ = self.generate_obstruction_prism(obs_pts)
            #         obs_mesh = pymesh.boolean(obs_mesh)

        return cam_mesh

    def generate_obstruction_prism(self, obs_pts):
        pts = np.zeros([6, 3])
        O = self.pose[:3]
        pts[:3, :] = obs_pts
        for i in range(3):
            v = obs_pts[i] - O
            v /= np.linalg.norm(v)
            pts[3 + i, :] = 1.5 * self.range[1] * v + O  # using double the range to ensure the intersection is complete

        faces = np.zeros([8, 3])
        faces[0, :] = np.array([0, 1, 2])  # front
        faces[1, :] = np.array([4, 3, 5])  # back

        for i in range(3):
            faces[2+2*i, :] = np.array([(i+1) % 3, i, i+3])  # side i, tri 1
            faces[2+2*i+1, :] = np.array([i+3, (i+1) % 3 + 3, (i+1) % 3])  # side i, tri 2

        return pymesh.form_mesh(pts, faces)

    def generate_discrete_camera_mesh(self, degrees_per_step, environment=None, apply_distortion=True):
        """

        """

        O = self.pose[:3]
        R = rot3d_from_rtp(np.array([self.pose[-1], -self.pose[-3], -self.pose[-2]]))
        fov = self.fov

        # fov_h = np.linspace(-0.5*fov[0], 0.5*fov[0], int(fov[0]/degrees_per_step)+1)  # steps over FOV
        # fov_v = np.linspace(-0.5*fov[1], 0.5*fov[1], int(fov[1]/degrees_per_step)+1)

        u_ = np.linspace(0, self.resolution[0], int(fov[0]/degrees_per_step)+1)  # Convert FOV into  pixels (pixel coords are 0 at top left
        v_ = np.linspace(0, self.resolution[1], int(fov[1]/degrees_per_step)+1)
        M = len(u_)
        N = len(v_)
        u, v = np.meshgrid(u_, v_)  # create meshgrid
        u = u.reshape(-1)  # stack all grid points into column to pass to undistort
        v = v.reshape(-1)
        uv_ = np.expand_dims(np.vstack([u, v]).T, axis=1)

        distCoeffs = np.array([self.k1, self.k2, self.p1, self.p2, self.k3])
        K_new = np.array([[self.resolution[0]/2, 0, -.5],
                          [0, self.resolution[0]/2, -.5],
                          [0, 0, 1]])
        pts = cv.convertPointsToHomogeneous(src=uv_)
        if apply_distortion:
            undistorted_pts = cv.undistortPoints(src=uv_, cameraMatrix=self.K, distCoeffs=distCoeffs, P=K_new).squeeze()
        else:
            undistorted_pts = uv_
        # projected_pts = cv.projectPoints(pts, rvec=np.eye(3), tvec=np.zeros(3), cameraMatrix=self.K,
        #                                  distCoeffs=distCoeffs)
        # print("Min: " + str(np.min(undistorted_pts, axis=0)))
        # print("Max: " + str(np.max(undistorted_pts, axis=0)))
        undistorted_pts = undistorted_pts.reshape(N, M, 2)
        # for each grid point, apply ray tracing to find the nearest obstacle in that direction
        obstructions = np.zeros([M, N, 2, 3])

        vertices = np.zeros([M*N*2, 3])
        faces = np.zeros([4*((N-1)*(M-1)+N+M-2), 3])

        # count = 1

        rep1 = Reporter()
        # rep2 = Reporter()
        focal_length = self.K[0, 0]  # focal length, in pixels

        for h in rep1.pbar(M, "Finding FOV Ray Obstructions"):
            # for v in rep2.pbar(N, "Finding Vertical FOV Ray Obstructions"):
            for v in range(N):
                # print("finding obstructions for ray " + str(count) + " of " + str(M*N))
                # count += 1
                px_u, px_v = undistorted_pts[v, h, :]
                v_pt = np.array([focal_length, -px_u, -px_v])
                d = v_pt / np.linalg.norm(v_pt)
                # R_ = rot3d_from_x_vec(v_pt)
                # d = R_[:, 0]
                obstructions[h, v, 0, :] = np.dot(R, d)*self.range[0] + O

                if environment is not None:
                    point = O + np.dot(R, d)*self.range[1]
                    _, closest_obstruction, _ = self.get_obstruction_mesh_obb_tree(points=np.array([point]),
                                                                                        environment=environment,
                                                                                        find_closest_intersection=True)
                    obstructions[h, v, 1, :] = closest_obstruction[0].copy()

                else:
                    obstructions[h, v, 1, :] = O + np.dot(R, d)*self.range[1]

                vertices[v+h*N, :] = obstructions[h, v, 0, :]
                vertices[v+h*N + M*N, :] = obstructions[h, v, 1, :]

        f = 0
        for h in range(M-1):
            for v in range(N-1):
                tri = np.array([[v+h*N, v+(h+1)*N, v+1+h*N],
                               [v+1+h*N, v+(h+1)*N, v+1+(h+1)*N]])

                # front faces
                faces[f:f+2, :] = tri
                f += 2
                # back faces
                tri = np.array([[v+(h+1)*N, v+h*N, v+1+h*N],
                               [v+(h+1)*N, v+1+h*N, v+1+(h+1)*N]])
                faces[f:f+2, :] = tri + N*M
                f += 2

                # top faces
                if v == N-2:
                    faces[f:f+2, :] = np.array([[v+1+h*N, v+1+(h+1)*N, N*M + v+1+h*N],
                                                [N*M + v+1+h*N, v+1+(h+1)*N, N*M + v+1+(h+1)*N]])
                    f += 2

                # bottom faces
                if v == 0:
                    faces[f:f+2, :] = np.array([[(h+1)*N, h*N, N*M + h*N],
                                                [(h+1)*N, N*M + h*N, N*M + (h+1)*N]])
                    f += 2

                # right faces
                if h == M-2:
                    faces[f:f+2, :] = np.array([[v+1+(h+1)*N, v+(h+1)*N, v+(h+1)*N + M*N],
                                                [v+1+(h+1)*N, v+(h+1)*N + M*N, v+1+(h+1)*N + M*N]])
                    f += 2

                # left faces
                if h == 0:
                    faces[f:f+2, :] = np.array([[v, v+1, v+M*N],
                                                [v+M*N, v+1, v+1+M*N]])
                    f += 2
        ccw = True
        if ccw:  # invert vertex ordering to change mesh normal direction
            faces = np.array([faces[:, 1], faces[:, 0], faces[:, 2]]).transpose()

        cam_mesh = pymesh.form_mesh(vertices, faces)
        # pymesh.save_mesh("c_mesh_2.ply", cam_mesh)
        return cam_mesh

    def map_3D_to_img(self, point):
        """
        converts 3D coordinates into image pixel position, including radial distortions and camera intrinsic and
        extrinsic.

        :param point - np.array, point in R3

        :return u_u, v_u - the horizontal and vertical pixel coordinates of the 3D point
        """
        # shift so point is at origin
        p = (point - self.pose[:3])

        u_0 = self.K[0, 2]
        v_0 = self.K[1, 2]

        f = self.K[0, 0]

        R_b = rot3d_from_rtp(np.array([self.pose[-1], -self.pose[-3], -self.pose[-2]]))  # CHECK FORM VECTORIZE
        v2 = np.dot(R_b.transpose(), p.transpose())
        # TODO: VERIFY INDEXING....
        x = v2[1]/v2[0]
        y = v2[2]/v2[0]
        r2 = x**2 + y**2

        u = -v2[1]*f/v2[0] + u_0
        v = -v2[2]*f/v2[0] + v_0

        u_u = u + (u - u_0) * (self.k1*r2 + self.k2*r2**2 + self.k3*r2**3) + (2*self.p1*x*y + self.p2*(r2+2*x**2))
        v_u = v + (v - v_0) * (self.k1*r2 + self.k2*r2**2 + self.k3*r2**3) + (2*self.p2*x*y + self.p1*(r2+2*y**2))

        return u_u, v_u

    def is_in_fov_3D_distorted(self, point):
        """
        Determines whether a 3D point is within the 3D FOV of a camera when using the distorted camera model and camera
        intrinsic provided.

        :param point: 3D np.array corresponding with the target point coordinates
        :return: boolean, true if point is in FOV
        """
        u_u, v_u = self.map_3D_to_img(point)

        # if 0.0 <= u_u <= self.resolution[0] and 0.0 <= v_u <= self.resolution[1]:
        #     return True
        # else:
        #     return False

        return (0.0 <= u_u)*(u_u <= self.resolution[0]) * (0.0 <= v_u)*(v_u <= self.resolution[1])

    def update_range_to_minimum_feature_size(self, min_diameter_meters, min_diameter_pixel):
        # update max range to ensure a feature with min_radius_meters will have at least min_radius_pixel
        thetas = (self.fov/self.resolution*min_diameter_pixel/2)
        Rs = min_diameter_meters/2/np.sin(thetas*np.pi/180)
        max_range = min(Rs)
        self.range[1] = min(max_range, self.range[1])  # keep more limiting of camera FOV and min feature requirement
        self.max_covariance_score = self.estimate_covariance_score(np.array([self.pose[:3] +
                                                                             np.array([self.range[1], 0, 0])]))


class PlacedCameras:
    # TODO: could make this part of Environment rather than passing it separately
    """
    Class containing a collection of Camera objects and their respective scores for a given environment. This class is
    primarily used when implementing a greedy search strategy, to track previously placed cameras.
    """
    def __init__(self, cameras=None, covariances=None):
        """
        Initialize the list of already placed cameras. If no inputs are specified, lists are initialized as empty. Note:
        The order of the lists should match each other.

        :param cameras: a list of Camera objects corresponding to the cameras which have already been placed in an
        environment
        :param covariances: a MxNx3x3 np array corresponding to the covariance of each placed camera, M, over each each
         sample point, N, in 3D space.
        """
        if cameras is None:
            self.cameras = []
        else:
            self.cameras = cameras

        # scores should be a list of score arrays (e.g.: [score1, score2, ..., scoreN]
        # where scoreN has the form: np.zeros([num_points, 3])
        self.covariances = covariances

    def append_covariances(self, covariances):
        if self.covariances is None:
            self.covariances = covariances.copy()
        else:
            self.covariances = np.concatenate([self.covariances, covariances])
