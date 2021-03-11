import random

import rospy

import copy
import numpy as np
from shapely.geometry import Polygon

from opt.optimization_options import CameraPlacementOptions
from sim.cameras import Camera
# from vis.draw2d import draw_boundary_2d
from geom.geometry2d import is_in_region
from geom.geometry3d import dist, get_polygon_perimeter, rot3d_from_rtp, rot3d_from_z_vec, generate_cylinder_mesh, \
    generate_circle_mesh, generate_extruded_mesh, generate_box_mesh, generate_box_mesh_components
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tools.tools import set_axes_equal
from sklearn.neighbors import KDTree

# import pymesh
# import open3d
# import pyvista
# import vtk
# import vedo
import trimesh
import time


class Environment:
    """
    DEPRECATED. A basic class capable of representing 2D or 3D environments. This class contains geometric information
    for the walls, obstacles, target, and perchable surfaces in the environment.
    """

    def __init__(self, walls, n_points=50, dimension=3, target=None, obstacles=None, perch_regions=None,
                 opt_options=None, ceiling=None):
        """
        Initialization function for the Environment class

        :param walls: (required) A 2D polygon (np.array) OR A set of 3D surfaces (list of np.arrays) corresponding with
        the boundaries of the environment.
        :param n_points: Number of numerical integration points in the target region. n_points will be randomly sampled
        over the region.
        :param dimension: Dimensionality of the problem and visualization. Must be 2 or 3.
        :param target: A polygon representing the target region you wish to capture. If no target is specified, the
        entire room is used as the target
        :param obstacles: A list of polygons  which can obstruct camera views of the target. If no obstacles are
        specified, the walls are used as obstacles.
        :param perch_regions: A list of polygons representing the surfaces on which cameras can be placed. If none are
        specified, the walls are used.
        :param opt_options: CameraPlacementOptions object containing the min_vertices parameter. If set, the
        set of perch_regions will be further restricted to regions which have a line of sight to at least min_vertices
        of the target polygon.
        :param ceiling: A polygon (or list of polygons if there are holes or islands) corresponding to the room's
        ceiling. If not specified, the walls will be used to determine a ceiling region.
        """
        self.walls = walls
        if target is None:
            self.target = walls
        else:
            self.target = target

        if ceiling is None:
            self.ceiling = walls  # this is invalid in 3D (I THINK?)
        else:
            self.ceiling = ceiling

        self.obstacles = obstacles
        self.n_points = n_points
        self.opt_options = opt_options
        self.wall_perimeter = get_polygon_perimeter(self.ceiling)
        self.x_lim = np.array([np.min(self.ceiling[:, 0]), np.max(self.ceiling[:, 0])])
        self.y_lim = np.array([np.min(self.ceiling[:, 1]), np.max(self.ceiling[:, 1])])

        if dimension == 2:
            self.z_lim = np.array([0, 2.75])
        else:
            self.z_lim = np.array([np.min(self.ceiling[:, 2]), np.max(self.ceiling[:, 2])])

        self.dimension = dimension

        self.sampling_points = np.zeros([self.n_points, self.dimension])
        self.generate_integration_points()

        self.target_centroid = np.zeros(3)
        self.centroid = np.zeros(3)
        self.find_target_centroid()
        self.max_height = self.z_lim[1]

        self.perch_perimeter = self.wall_perimeter

        if perch_regions is None:
            self.perch_regions = walls
            self.perch_perimeter = self.wall_perimeter

        else:
            self.perch_regions = perch_regions
            if self.dimension == 2:
                self.update_perch_perimeter_2d()

        if opt_options is not None:
            if opt_options.min_vertices > 0 and self.dimension == 2:
                self.restrict_perching_area(opt_options.min_vertices)
            self.opt_options = opt_options
        else:
            self.opt_options = CameraPlacementOptions()
            # consider implementing in 3D case

    def generate_integration_points(self):
        """
        Generates n_points over which to integrate. These are set in self.sampling_points and are used to evaluate
        camera placements

        :return:
        """
        n = 0

        while n < self.n_points:
            pt = np.array([random.uniform(self.x_lim[0], self.x_lim[1]),
                           random.uniform(self.y_lim[0], self.y_lim[1]),
                           0])
            if is_in_region(pt[:2], self.target):
                self.sampling_points[n, :] = pt
                n = n + 1

    def find_target_centroid(self):
        """
        Finds the centroid of the target region polygon. NOTE: uses the shapely Polygon class which ignores z
        coordinates. Thus, the centroid will always be specified at z=0.

        :return: a 3D np.array corresponding with the centroid of the target region.
        """
        target_poly = Polygon(self.target)
        x, y = target_poly.centroid.xy
        self.centroid = np.array([x[0], y[0], 0])

    def restrict_perching_area(self, min_vertices, n_divs=20):
        """
        A function to restrict the possible perching areas in the environment based on the minimum number of vertices
        which must be visible from any given perching point. This is meant to reduce the search space of the problem.
        This function is currently only implemented in 2D. To simplify computation, visibility is assessed at n_divs
        discrete points, rather than over the continuous function. This function is only evaluated once, therefore the
        cost of increasing n_divs is limited.

        :param min_vertices: the minimum number of target_region vertices which must by visible by a given point
        :param n_divs: the number of divisions along each segment in the original perch area to assess visibility.
        Defaults to 20.
        :return:
        """
        # 2D approach:
        # Starting with an approximate numerical approach...
        new_perch_areas = np.empty([0, 2, 2])
        dummy_cam = Camera()

        for l in self.perch_regions:  # for each perchable area...
            p_list = np.transpose(np.array([np.linspace(l[0, 0], l[1, 0], n_divs),
                                            np.linspace(l[0, 1], l[1, 1], n_divs)]))
            valid_points = -np.ones([n_divs])
            for i in range(n_divs):
                visible_vertices = 0
                for v in self.target:  # and each target vertex...
                    dummy_cam.pose[:2] = p_list[i, :]

                    obstructed = dummy_cam.is_obstructed(v, self.obstacles)
                    if not obstructed:
                        visible_vertices = visible_vertices + 1
                        if visible_vertices >= min_vertices:
                            # if you pass min threshold, point is valid and included. no need to test rest
                            valid_points[i] = 1
                            break

            # now break p_list into continuous ranges of values...
            # edge case: only 1s
            if sum(valid_points == len(valid_points)):
                # take first and last point
                new_perch_areas = np.append(new_perch_areas, np.array([p_list[[0, -1], :]]), 0)
            else:
                # find transitions from -1 to 1
                # transitions = np.multiply(valid_points[:-1], valid_points[1:]) == -1
                j = 0
                while j < n_divs:
                    if valid_points[j] == 1:
                        start_point = j
                        while j < n_divs and valid_points[j] == 1:
                            j = j + 1
                        end_point = j - 1
                        new_perch_areas = np.append(new_perch_areas, np.array([p_list[[start_point, end_point], :]]), 0)
                    else:
                        j = j + 1
        self.perch_regions = new_perch_areas
        if self.dimension == 2:
            self.update_perch_perimeter_2d()

    def update_perch_perimeter_2d(self):
        """
        Determines the perimeter of the perch regions for the 2D problem. This function updates self.perch_perimeter

        :return:
        """
        self.perch_perimeter = 0
        for i in range(np.shape(self.perch_regions)[0]):
            self.perch_perimeter = self.perch_perimeter + dist(self.perch_regions[i, 0, :], self.perch_regions[i, 1, :])

    def update_perch_perimeter_3d(self):
        """
        Updates the perch_perimeter value in the 3D case. This function will be depricated when moving to a full 3D mesh
        representation of the environment.

        :return:
        """
        self.perch_perimeter = 0
        for i in range(np.shape(self.perch_regions)[0]):
            self.perch_perimeter = self.perch_perimeter + dist(self.perch_regions[i][0, :], self.perch_regions[i][1, :])


class MeshEnvironment:
    def __init__(self, surfaces, n_points=50, target=None, obstacles=None, clust_env=None,
                 full_env=None, opt_options=None, reorient_mesh=False, cluster_env_remeshed=None,
                 clust_env_path=None, full_env_path=None, name=""):  # , surface_number=-1):
        t0 = time.time()
        rospy.loginfo(rospy.get_caller_id() + " Setting up MeshEnvironment")
        self.name = name

        self.surfaces = surfaces  # total set of segmented surfaces, meeting minimum area requirement

        self.removed_regions = []

        self.gravity_direction = opt_options.gravity_direction

        self.full_env = full_env
        self.cluster_env = clust_env  # ADDING COPY OF CLUSTER ENV - 1 to preserve original colors,
                                      # other for uniform meshing to improve search speed
        self.cluster_env_remeshed = cluster_env_remeshed  # TODO: remove

        if opt_options is not None:
            self.opt_options = opt_options
        else:
            self.opt_options = CameraPlacementOptions()

        use_floor_as_target = False
        if target is None:
            self.target = None
            use_floor_as_target = True
        else:
            self.target = target

        self.R = np.eye(3)  # rotation matrix to be used IF the environment is realigned
        self.reorient_mesh = reorient_mesh

        if reorient_mesh:
            self.R = self.opt_options.world_frame
            self.align_meshes_with_gravity()
        else:
            self.R = np.eye(3)
        self.gravity_direction = np.array([0, 0, -1])

        t1 = time.time()
        # print("Reorienting Mesh Takes: " + str(t1-t0))
        # testing...
        # obs_mesh = pymesh.form_mesh(vertices=self.cluster_env.points, faces=self.cluster_env.faces)
        # self.cluster_env_path = "/home/simon/catkin_ws/src/perch_placement/models/rotated_obstacle_mesh.ply"
        # pymesh.save_mesh(self.cluster_env_path, obs_mesh)
        self.obs_mesh = trimesh.Trimesh(self.cluster_env.points, self.cluster_env.faces)
        # self.obs_mesh = vedo.load(cluster_env_path)
        self.vedo_mesh = None

        t2 = time.time()
        # print("Setting up OBB Tree Takes: " + str(t2-t1))

        if self.cluster_env_remeshed is not None:  # TODO: REMOVE
            self.tree = KDTree(self.cluster_env_remeshed.face_centers)  # make tree with set of unique points
            # consider using face centers...
        else:
            self.tree = None

        self.full_env_path = full_env_path
        self.clust_env_path = clust_env_path

        t3 = time.time()
        # print("Setting up KD Tree Takes: " + str(t3 - t2))

        self.centroid = np.zeros(3)
        self.net_area = 0
        self.avg_circumscription_radius = 0
        self.estimate_centroid()
        self.correct_normals()

        t4 = time.time()
        # print("Finding Normals and Centroids Takes: " + str(t4-t3))

        self.floor = self.extract_boundary_plane(self.gravity_direction)

        if opt_options.perch_on_ceiling:
            self.ceil = self.extract_boundary_plane(-self.gravity_direction)
        else:
            self.ceil = None

        # Check floor to ceiling distance...
        if self.floor is not None and self.ceil is not None:
            if np.dot(self.floor.centroid - self.ceil.centroid, self.gravity_direction) < \
                    self.opt_options.min_room_height:
                # assume ceiling is at least Xm above the floor...
                # this prevents tables, etc, from being considered ceilings
                self.ceil = None

        if use_floor_as_target:
            assert self.floor is not None
            self.target = self.floor
        self.target_centroid = copy.deepcopy(self.target.centroid)

        t5 = time.time()
        # print("Extracting Boundary Planes Takes: " + str(t5-t4))

        if obstacles is None:
            self.obstacles = surfaces
        else:
            self.obstacles = obstacles

        self.n_points = n_points
        self.dimension = 3

        self.x_lim = np.zeros(2)
        self.y_lim = np.zeros(2)
        self.z_lim = np.zeros(2)
        self.find_env_limits()

        self.sampling_points = np.zeros([self.n_points, self.dimension])
        self.noise_resist_sampling_points = None
        self.noise_resistant_search = False
        self.generate_integration_points()

        self.perch_regions = []
        self.perch_area = 0
        self.set_surface_as_perchable()  # surface_number)

        self.bins = np.zeros(len(self.perch_regions))
        self.perch_region_areas = np.zeros(len(self.perch_regions))
        self.perch_region_n_faces = np.zeros(len(self.perch_regions))
        self.perch_region_normals = np.zeros(len(self.perch_regions))
        self.perch_region_bins = []
        self.update_bins()

        # debugging:
        for s in self.surfaces:
            if s.faces is None or s.points is None:
                print('PROBLEM')

    def update_bins(self):
        self.bins = np.zeros(len(self.perch_regions))
        for i in range(len(self.perch_regions)):
            self.perch_region_areas[i] = self.perch_regions[i].net_area
            self.perch_region_n_faces[i] = len(self.perch_regions[i].faces)
            self.perch_region_normals[i] = self.perch_regions[i].mesh_normal
        self.bins[1:] = np.cumsum(self.perch_region_areas)
        self.perch_region_bins = np.array([self.perch_regions[i].bins for i in range(len(self.perch_regions))],
                                          dtype=object)

    def apply_area_filter(self, min_area=0.5):
        for s in range(len(self.perch_regions)):
            if self.perch_regions[s].is_valid:
                if self.perch_regions[s].net_area < min_area:
                    self.perch_regions[s].is_valid = False
                    # self.removed_regions.append(self.perch_regions[s].copy())
        self.update_bins()

    def post_process_environment(self, apply_height_filter=True, apply_proximity_filter=True,
                                 apply_line_of_sight_filter=True, apply_perch_window_filter=True):

        for s in range(len(self.perch_regions)):  # only post process meshes that are actually in search space
            if self.perch_regions[s].is_valid:
                # if s == 4:
                #     self.perch_regions[s].plot_mesh()
                if s == 7 or s == 13:
                    ax = Axes3D(plt.figure())
                    self.perch_regions[s].plot_mesh(ax)

                if self.perch_regions[s].faces is None or self.perch_regions[s].points is None:
                    print("Problem! Debugging... this shouldn't happen")
                if self.perch_regions[s].classification == "wall":
                    if self.opt_options.min_recovery_height > 0 and apply_height_filter:
                        # self.perch_regions[s].enforce_minimum_height(floor_centroid=self.floor.centroid,
                        #                                         min_height=self.opt_options.min_recovery_height,
                        #                                         gravity_direction=self.gravity_direction)
                        # height_cutoff = np.zeros([0, 3])

                        # horizontal direction
                        # v_h = np.cross(self.gravity_direction, self.perch_regions[s].eigen_vectors[-1])
                        # dp = np.dot(self.perch_regions[s].points, v_h)
                        # h_range = np.array([np.min(dp), np.max(dp)])
                        # n_pts = 100
                        # hc = np.linspace(h_range[0], h_range[1] + (h_range[1]-h_range[0])/n_pts, n_pts)

                        # i was here =======================================
                        height_cutoff = self.perch_regions[s].find_min_height(self.opt_options, self.gravity_direction)
                        # i was here =======================================

                        self.perch_regions[s].enforce_minimum_height2(height_cutoff,
                                                                      gravity_direction=self.gravity_direction,
                                                                      px_per_m=100)

                        if not self.perch_regions[s].is_valid:
                            continue

                # Find all obstructions to center point and add those faces to surface.obstacles_near_mesh
                if apply_line_of_sight_filter:
                    c = Camera()
                    if self.perch_regions[s].is_valid:
                        self.perch_regions[s].target_center = self.target_centroid
                        for i in range(len(self.perch_regions[s].face_centers)):
                            c.pose[:3] = self.perch_regions[s].face_centers[i]
                            obstructed, _, obstructing_faces = c.get_mesh_obstruction(point=self.target_centroid,
                                                                        clusters_remeshed=self.cluster_env_remeshed,
                                                                        opt_options=self.opt_options,
                                                                        tree=self.tree,
                                                                        find_closest_intersection=False,
                                                                        search_radius=self.opt_options.dist_threshold)
                            # obstructed, _, obstructing_face = c.get_obstruction_mesh_obb_tree(points=np.array([self.target_centroid]),
                            #                                                                     environment=self,
                            #                                                                     find_closest_intersection=False)
                            # Ideally would us OBB tree as its much faster... buutttt.... need a face id as a return to properly match...

                            for obs_face in obstructing_faces:
                                self.perch_regions[s].target_obstructions.append(
                                    self.cluster_env_remeshed.points[self.cluster_env_remeshed.faces[obs_face, :], :])
                else:
                    self.perch_regions[s].target_obstructions = []
                if apply_proximity_filter:
                    self.opt_options.nearest_neighbor_restriction = True
                else:
                    self.opt_options.nearest_neighbor_restriction = False
                    self.perch_regions[s].obstacles_near_mesh = []

                if not apply_perch_window_filter:
                    self.opt_options.min_perch_window = np.array([0.0, 0.0])
                elif (self.opt_options.min_perch_window == 0).any():
                    self.opt_options.min_perch_window = np.array([.3, .3])

                self.perch_regions[s].post_process_mesh(self.opt_options)

        # not in love with redoing these for all surfaces as its wasteful and as the
        #  perch_regions may be linked to surfaces
        self.update_perch_area()
        self.correct_normals()
        self.update_bins()

    def remove_rejected_from_perch_space(self, camera, r=0.2):
        surf_id = camera.surf_id
        O = camera.pose[:3]
        H_vec = camera.wall_normal / np.linalg.norm(camera.wall_normal)

        # generate approximate circle around the position, oriented with plane
        pts, faces = generate_circle_mesh(O, r, H_vec, num_pts=20)
        regions = pts[faces]

        self.perch_regions[surf_id].remove_regions_from_mesh(regions)
        self.update_perch_area()
        self.correct_normals()  # could save previous results and save some work here...
        self.update_bins()

    def generate_integration_points(self, distribution="uniform", shape="box"):
        if distribution == "random":
            self.sampling_points = self.generate_random_target_points()
            if self.opt_options.noise_resistant_particles > 0:
                self.noise_resist_sampling_points = \
                    self.generate_random_target_points(self.opt_options.noise_resistant_sample_size)
        elif distribution == "uniform":
            if shape=="cylinder":
                sampling_points = self.generate_uniform_target_points()
            else:
                sampling_points = self.generate_uniform_target_points_rectangular()
            if len(sampling_points) != self.n_points:
                self.sampling_points = sampling_points
                self.n_points = len(self.sampling_points)

                if self.opt_options.noise_resistant_particles > 0:
                    if shape=="cylinder":
                        self.noise_resist_sampling_points = np.vstack([self.sampling_points,
                            self.generate_uniform_target_points(N_target=self.opt_options.noise_resistant_sample_size)])
                    else:
                        self.noise_resist_sampling_points = np.vstack([self.sampling_points,
                            self.generate_uniform_target_points_rectangular(N_target=self.opt_options.noise_resistant_sample_size)])

        else:
            warnings.warn("Invalid integration point generation method selected. Options are \"random\" or \"uniform\"")

        self.target_centroid += -self.gravity_direction * self.opt_options.target_volume_height/2
        return -1

    def map_particle_to_surface(self, particle, surface_index=None, weight_by_area=True):
        n = len(particle)
        # possible cases:
        # generate point on specific surface, random face center placement (N=1)
        # generate point on specific surface, random placement on random face (N=3)
        # generate point on random surface, face center placement (N=2)
        # generate point on random surface, random placement on random face placement (N=4)
        # If weight_by_area is true, surface and face selection will by weighted by their respective areas

        if surface_index is not None and surface_index >= 0:
            surf = self.perch_regions[surface_index]
        elif n == 2 or n == 4:
            # map last particle index to surface list
            if weight_by_area:
                p_scaled = particle[-1] * self.perch_area
                cumulative_area = self.perch_regions[0].net_area
                surface_index = 0
                while cumulative_area < p_scaled:
                    surface_index = surface_index + 1
                    cumulative_area = cumulative_area + self.perch_regions[surface_index].net_area
                surf = self.perch_regions[surface_index]
            else:
                surf = self.perch_regions[np.floor(particle[-1] * len(self.perch_regions))]
                surface_index = np.floor(particle[-1] * len(self.perch_regions))
        else:
            warnings.warn("IMPROPER PARTICLE SIZE. Particle must have dimension 1 or 3 for local-indexed, 2 or 4 for "
                          "global")
            return -1

        if 0 < n <= 4:
            # randomly select face from surface using the first particle component
            if weight_by_area:
                p_scaled = particle[0] * surf.net_area
                cumulative_area = surf.face_areas[0]
                i = 0

                # occasionally get index out of bounds when zero-tolerance comparison is used
                while cumulative_area < p_scaled - 1e-8:
                    i = i + 1
                    cumulative_area = cumulative_area + surf.face_areas[
                        i]  # TODO: this can definitely be more efficient by precomputing cumulative area and doing a binary search
                f = i
            else:
                f = np.floor(particle[0] * surf.num_faces)

            normal_direction = surf.mesh_normal
            assert np.abs(np.linalg.norm(normal_direction) - 1) <= np.finfo(float).eps

            # normal_direction = surf.face_normals[f]
            # print("Mesh Norm 1: " + str(surf.mesh_normal) + "; Face Norm 2: " + str(surf.face_normals[f]))
            if n >= 3:
                # select point on face
                point = surf.generate_points_on_mesh(n=1, f=f, r1=particle[1], r2=particle[2])
            else:
                point = surf.face_centers[f]
        else:
            warnings.warn("IMPROPER PARTICLE SIZE. Particle must have dimension 1 or 3 for local-indexed, 2 or 4 for "
                          "global")
            return -1

        return point, normal_direction, False, surface_index

    def map_particle_to_flattened_surface(self, particle, surface_index=None, weight_by_area=True):
        n = len(particle)
        # possible cases:
        # generate point on specific surface, random 2D placement (N=2)
        # generate point on random surface, random 2D placement (N=3)
        # If weight_by_area is true, surface selection will by weighted by their respective areas
        if surface_index is not None and n == 2:
            surf = self.perch_regions[surface_index]
        elif (surface_index is None or surface_index < 0) and n == 3:
            # map last particle index to surface list
            if weight_by_area:
                p_scaled = particle[-1] * self.perch_area
                cumulative_area = self.perch_regions[0].net_area
                surface_index = 0
                while cumulative_area < p_scaled - np.finfo(float).eps:
                    surface_index = surface_index + 1
                    if surface_index >= len(self.perch_regions):
                        warnings.warn("Cumulative area exceeds perchable area. Particle: "
                                      + str(particle[-1]) + "; cumulative area: " + str(cumulative_area)
                                      + "; p_scaled = " + str(p_scaled))
                    cumulative_area = cumulative_area + self.perch_regions[surface_index].net_area
                surf = self.perch_regions[surface_index]
            else:
                surf = self.perch_regions[np.floor(particle[-1] * len(self.perch_regions))]
                surface_index = np.floor(particle[-1] * len(self.perch_regions))
        else:
            warnings.warn("IMPROPER PARTICLE SIZE. Particle must have dimension 1 or 3 for local-indexed, 2 or 4 for "
                          "global")
            return -1

        if 2 <= n <= 3:
            # randomly select face from surface using the first particle component
            normal_direction = surf.mesh_normal
            assert np.abs(np.linalg.norm(normal_direction) - 1) <= np.finfo(float).eps
            prj_lim = np.array([np.min(surf.projected_points, axis=0), np.max(surf.projected_points, axis=0)])
            prj_min = np.min(prj_lim, axis=0)
            prj_range = prj_lim[1, :] - prj_lim[0, :]

            point = particle[:2] * prj_range + prj_min
            # first check if the point is in an outer polygon...
            in_loop = False
            for outer_loop in range(len(surf.perimeter_loop)):
                if is_in_region(point, surf.projected_points[surf.perimeter_loop[outer_loop]]):
                    # draw_boundary_2d(surf.projected_points[surf.perimeter_loop[outer_loop]])
                    # plt.scatter(point[0], point[1])
                    # plt.pause(1)
                    in_loop = True
                    break
            if not in_loop:
                point = np.ones_like(point) * np.nan

            if not np.isnan(point).any():
                for inner_loop in range(len(surf.inner_loops)):
                    if is_in_region(point, surf.projected_points[surf.inner_loops[inner_loop]]):
                        # draw_boundary_2d(surf.projected_points[surf.inner_loops[inner_loop]])
                        # plt.scatter(point[0], point[1])
                        # plt.pause(1)
                        point = np.ones_like(point) * np.nan
                        break

            # draw_boundary_2d(surf.projected_points[surf.inner_loops[inner_loop]])
            # plt.scatter(point[0], point[1])
            # plt.pause(1)

            point = np.dot(point, surf.eigen_vectors[:2, :]) + surf.pca_mean

        else:
            warnings.warn("IMPROPER PARTICLE SIZE. Particle must have dimension 1 or 3 for local-indexed, 2 or 4 for "
                          "global")
            return -1

        return point, normal_direction, False, surface_index

    def map_particle_to_surface_vectorized(self, particles, surface_indices=None, weight_by_area=True,
                                           map_to_flat=False):
        n = len(particles[0])
        # possible cases:
        # generate point on specific surface, random face center placement (N=1)
        # generate point on specific surface, random placement on random face (N=3)
        # generate point on random surface, face center placement (N=2)
        # generate point on random surface, random placement on random face placement (N=4)
        # If weight_by_area is true, surface and face selection will by weighted by their respective areas
        if (n == 2 or n == 4 and not map_to_flat) or (n == 3 and map_to_flat):
            # map last particle index to surface list
            if weight_by_area:
                p_scaled = particles[:, -1] * self.perch_area
                surface_indices = np.digitize(p_scaled, self.bins) - 1
            else:
                surface_indices = np.floor(particles[:, -1] * len(self.perch_regions))

        elif not (surface_indices is not None and (surface_indices >= 0).all()):
            warnings.warn("IMPROPER PARTICLE SIZE. Particle must have dimension 1 or 3 for local-indexed, 2 or 4 for "
                          "global")
            return -1

        if 0 < n <= 4 and not map_to_flat:
            # randomly select face from surface using the first particle component
            if weight_by_area:
                p_scaled = particles[:, 0] * self.perch_region_areas[surface_indices]
                fs = np.zeros_like(particles[:, 0])

                # TODO: if possible, replace this loop but I can't think of a way to do so right now...
                for i in range(len(fs)):
                    bins = self.perch_region_bins[surface_indices[i]]
                    fs[i] = np.digitize(p_scaled[i], bins)

            else:
                fs = np.floor(particles[:, 0] * self.perch_region_n_faces[surface_indices])

            normal_directions = self.perch_region_normals[surface_indices]

            # surfs = np.array(self.perch_regions[surface_indices], dtype=object)

            if n >= 3:
                # select point on face
                points = np.array([self.perch_regions[surface_indices[i]].generate_points_on_mesh(n=1,
                                                                                                  f=fs[i],
                                                                                                  r1=particles[i, 1],
                                                                                                  r2=particles[i, 2])
                                   for i in range(len(surface_indices))])
            else:
                points = np.array([self.perch_regions[surface_indices[i]].face_centers[fs[i]]
                                   for i in range(len(surface_indices))])

        elif map_to_flat and 2 <= n <= 3:
            # randomly select face from surface using the first particle component
            normal_directions = self.perch_region_normals[surface_indices]
            surfs = np.array(self.perch_regions[surface_indices], dtype=object)
            N_particles = len(particles[:, 0])

            points = np.ones([N_particles, len(surfs[0].projected_points)])

            for i in range(len(particles[:, 0])):
                prj_lim = np.array([np.min(surfs[i].projected_points, axis=0),
                                    np.max(surfs[i].projected_points, axis=0)])
                prj_min = np.min(prj_lim, axis=0)
                prj_range = prj_lim[1, :] - prj_lim[0, :]

                points[i] = particles[i, :2] * prj_range + prj_min

                # first check if the point is in an outer polygon...
                in_loop = False

                for outer_loop in range(len(surfs[i].perimeter_loop)):
                    if is_in_region(points[i], surfs[i].projected_points[surfs[i].perimeter_loop[outer_loop]]):
                        in_loop = True
                        break
                if not in_loop:
                    points[i] = np.ones_like(points[i]) * np.nan

                if not np.isnan(points[i]).any():
                    for inner_loop in range(len(surfs[i].inner_loops)):
                        if is_in_region(points[i], surfs[i].projected_points[surfs[i].inner_loops[inner_loop]]):
                            point = np.ones_like(points[i]) * np.nan
                            break
                points[i] = np.dot(points[i], surfs[i].eigen_vectors[:2, :]) + surfs[i].pca_mean

        else:
            warnings.warn("IMPROPER PARTICLE SIZE. Particle must have dimension 1 or 3 for local-indexed, 2 or 4 for "
                          "global")
            return -1

        return points, normal_directions, False, surface_indices

    def extract_boundary_plane(self, direction):
        max_dist = 0
        aligned_surfs = []
        bound_plane = None

        for surf in self.surfaces:
            # check if the surfaces is approximately horizontal
            # surf.plot_mesh()
            if np.arccos(np.abs(np.dot(surf.mesh_normal, direction)))*180/np.pi < self.opt_options.angle_threshold:
                # score surf as potential candidate
                aligned_surfs.append(surf)
                # dot product of vector between surface centroid and environment centroid and the negative gravity
                # vector (upwards direction) gives the height of a surface, relative to the centroid
                v = surf.centroid - self.centroid
                norm_dist = np.dot(v, direction)

                if max_dist < norm_dist:
                    max_dist = copy.deepcopy(norm_dist)
                    bound_plane = copy.deepcopy(surf)

        if bound_plane is not None:
            for surf in aligned_surfs:
                # Find farthest planes in target direction
                if not np.array_equal(surf.centroid, bound_plane.centroid):
                    v = bound_plane.centroid - surf.centroid
                    norm_dist = np.dot(v, direction)

                    if norm_dist < self.opt_options.dist_threshold:
                        bound_plane = bound_plane + surf  # merge meshes for the floor

            return bound_plane

        else:
            warnings.warn("NO BOUNDARY PLANE FOUND FOR v=" + str(direction) + ". CHECK YOUR MODEL")
            return None

    def estimate_centroid(self):
        num_faces = 0
        net_area = 0
        centroid = np.zeros(3)

        for surf in self.surfaces:
            centroid += surf.centroid*surf.net_area
            net_area += surf.net_area
            num_faces += surf.num_faces

        self.centroid = centroid/net_area
        self.net_area = net_area

    def correct_normals(self):
        for s in range(len(self.surfaces)):
            if self.surfaces[s].is_valid:
                if np.dot(self.surfaces[s].mesh_normal, self.centroid - self.surfaces[s].centroid) < 0:
                    self.surfaces[s].mesh_normal = -self.surfaces[s].mesh_normal
                    # self.surfaces[s].eigen_vectors *= -1

                # optional step. also correct all face normals in the mesh... can comment out if desired
                for f in range(len(self.surfaces[s].faces)):
                    if np.dot(self.surfaces[s].mesh_normal, self.surfaces[s].face_normals[f]) < 0:
                        self.surfaces[s].face_normals[f] = -self.surfaces[s].face_normals[f]
        if self.target is not None:
            if np.dot(self.target.mesh_normal, self.centroid - self.target.centroid) < 0:
                self.target.mesh_normal *= -1
            # if np.dot(self.target.eigen_vectors[2], self.centroid - self.target.centroid) < 0:
                # self.target.eigen_vectors *= -1

    def align_meshes_with_gravity(self):
        # Should only be used to simplify debugging... In reality, its most efficient to leave in input orientation,
        # as this will also be the output orientation...
        # make sure gravity is normalized; invert to express +z direction
        R = self.R

        for s in range(len(self.surfaces)):
            self.surfaces[s].reorient_mesh(R)
        self.target.reorient_mesh(R)
        self.cluster_env.reorient_mesh(R)
        self.cluster_env_remeshed.reorient_mesh(R)
        if self.full_env is not None:
            self.full_env.reorient_mesh(R)

    def find_env_limits(self):
        surf = self.surfaces[0]
        self.x_lim = np.array([np.min(surf.points[:, 0]), np.max(surf.points[:, 0])])
        self.y_lim = np.array([np.min(surf.points[:, 1]), np.max(surf.points[:, 1])])
        self.z_lim = np.array([np.min(surf.points[:, 2]), np.max(surf.points[:, 2])])

        for surf in self.surfaces:
            self.x_lim = np.array([np.min([np.min(surf.points[:, 0]), self.x_lim[0]]),
                                   np.max([np.max(surf.points[:, 0]), self.x_lim[1]])])
            self.y_lim = np.array([np.min([np.min(surf.points[:, 1]), self.y_lim[0]]),
                                   np.max([np.max(surf.points[:, 1]), self.y_lim[1]])])
            self.z_lim = np.array([np.min([np.min(surf.points[:, 2]), self.z_lim[0]]),
                                   np.max([np.max(surf.points[:, 2]), self.z_lim[1]])])

    def plot_environment(self, fig_num=1, ax=None, plot_target=True, show=True, plot_normals = False):
        # testing...
        fig = plt.figure(fig_num)
        if ax is None:
            ax = Axes3D(fig)
        for surf in self.perch_regions:
            ax = surf.plot_mesh(ax)
            if plot_normals:
                ax = surf.plot_normal(ax)
        if plot_target:
            ax.scatter3D(self.sampling_points[:, 0], self.sampling_points[:, 1], self.sampling_points[:, 2])

        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_zlim(self.z_lim)
        ax.set_zlabel('Z (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlabel('X (m)')
        ax.set_title('')
        set_axes_equal(ax)
        if show:
            plt.show()

        return fig, ax

    def plot_clust_environment(self, fig_num=1, ax=None, plot_target=True, show=True):
        # testing...
        fig = plt.figure(fig_num)
        if ax is None:
            ax = Axes3D(fig)
        if plot_target:
            ax.scatter3D(self.sampling_points[:, 0], self.sampling_points[:, 1], self.sampling_points[:, 2])

        # Could... SPLIT ENV INTO DIFFERENT COLORED REGIONS AND PLOT SEPARATELY. but its very inefficient representation
        self.cluster_env.plot_mesh(ax)

        set_axes_equal(ax)

        if show:
            plt.show()

    def set_surface_as_perchable(self):
        # drop surfaces which should not be part of the search space
        self.perch_regions = []
        self.perch_area = 0.0
        for s in range(len(self.surfaces)):
            if self.surfaces[s].is_valid:
                # get angle of the surface WRT to upwards direction; by convention, returns on [0,pi]
                surf_angle = np.arccos(np.dot(-self.gravity_direction, self.surfaces[s].mesh_normal))*180/np.pi
                # if angle within angle_threshold of 0, then, floor
                if surf_angle < self.opt_options.angle_threshold:
                    self.surfaces[s].classification = "floor"
                    if self.opt_options.land_on_floor:
                        self.perch_regions.append(self.surfaces[s])
                        self.perch_area += self.surfaces[s].net_area
                    # else:
                    #     self.removed_regions.append(self.surfaces[s].copy())
                # elif angle within angle_threshold of 180, then ceil
                elif 180-surf_angle < self.opt_options.angle_threshold:
                    self.surfaces[s].classification = "ceiling"
                    if self.opt_options.perch_on_ceiling:
                        self.perch_regions.append(self.surfaces[s])
                        self.perch_area += self.surfaces[s].net_area
                    # else:
                    #     self.removed_regions.append(self.surfaces[s].copy())
                # elif angle within angle_threshold of 90, then wall
                elif np.abs(90-surf_angle) < self.opt_options.angle_threshold:
                    self.surfaces[s].classification = "wall"
                    if self.opt_options.perch_on_walls:
                        if self.surfaces[s].is_valid:
                            self.perch_regions.append(self.surfaces[s])
                            self.perch_area += self.surfaces[s].net_area
                    # else:
                    #     self.removed_regions.append(self.surfaces[s].copy())

                # else: unclassified
                elif self.opt_options.perch_on_intermediate_angles:
                    self.surfaces[s].classification = "unclassified"
                    self.perch_regions.append(self.surfaces[s])
                    self.perch_area += self.surfaces[s].net_area
                # else:
                    # self.removed_regions.append(self.surfaces[s].copy())

    def update_perch_area(self):
        self.perch_area = 0.0
        # drop surfaces which should not be part of the search space  # TODO: just use perch regions here...
        s = 0
        while s < len(self.perch_regions):
            if self.perch_regions[s].is_valid:
                # get angle of the surface WRT to upwards direction; by convention, returns on [0,pi]
                surf_angle = np.arccos(np.dot(-self.gravity_direction, self.perch_regions[s].mesh_normal)) * 180 / np.pi
                # if angle within angle_threshold of 0, then, floor
                if surf_angle < self.opt_options.angle_threshold:
                    if self.opt_options.land_on_floor:
                        self.perch_area += self.perch_regions[s].net_area
                # elif angle within angle_threshold of 180, then ceil
                elif 180 - surf_angle < self.opt_options.angle_threshold:
                    if self.opt_options.perch_on_ceiling:
                        self.perch_area += self.perch_regions[s].net_area
                # elif angle within angle_threshold of 90, then wall
                elif np.abs(90 - surf_angle) < self.opt_options.angle_threshold:
                    if self.opt_options.perch_on_walls:
                        self.perch_area += self.perch_regions[s].net_area
                # else: unclassified
                elif self.opt_options.perch_on_intermediate_angles:
                    self.perch_area += self.perch_regions[s].net_area
                s += 1
            else:
                self.perch_regions.pop(s)

    def assign_particles_to_surfaces(self, N_particles, k_neighbors, neighborhood_search=True):
        # rule of thumb for PSO is to use same number of particles as dimensions
        if neighborhood_search:
            min_particles = max(self.opt_options.get_particle_size(), k_neighbors)
        else:
            min_particles = self.opt_options.get_particle_size()

        particle_density = float(N_particles) / self.perch_area

        particles_per_surface = np.zeros(len(self.perch_regions), dtype=int)
        for i in range(len(self.perch_regions)):
            particles_per_surface[i] = np.max([round(self.perch_regions[i].net_area * particle_density), min_particles])

        return particles_per_surface

    def generate_uniform_target_points_rectangular(self, dpp=None, N_target=None):
        # ORIGINALLY DID RECTANGULAR PLACEMENT. TRYING CYLINDRICAL NOW
        if N_target is None:
            N_target = self.n_points

        pts, _ = generate_box_mesh_components(self.target,
                                              d=-self.gravity_direction*self.opt_options.target_volume_height,
                                              z_floor=self.floor.centroid[-1])
        v1 = pts[1] - pts[0]
        v2 = pts[3] - pts[0]
        v3 = pts[4] - pts[0]
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        d3 = np.linalg.norm(v3)

        v1 /= d1
        v2 /= d2
        v3 /= d3

        if dpp is None:
            V_target = d1*d2*d3
            # given desired number of points and target volume, the following point density is required:
            dpp = (V_target/N_target)**(1/3)

        x_steps = np.linspace(0, d1, int(np.ceil(d1/dpp)))
        y_steps = np.linspace(0, d2, int(np.ceil(d2/dpp)))
        z_steps = np.linspace(0, d3, int(np.ceil(d3/dpp)))

        n_pts = len(x_steps) * len(y_steps) * len(z_steps)

        sampling_points = np.zeros([n_pts, 3])

        c = 0
        for i in x_steps:
            for j in y_steps:
                for k in z_steps:
                    sampling_points[c, :] = pts[0] + i*v1 + j*v2 + k*v3
                    c += 1

        return sampling_points

    def generate_uniform_target_points(self, dpp=None, N_target=None):
        # ORIGINALLY DID RECTANGULAR PLACEMENT. TRYING CYLINDRICAL NOW
        O = self.target.centroid
        O[-1] = self.floor.centroid[-1]  # make sure region starts on the floor
        R_max = 0.0
        for pt in self.target.points:
            R = dist(O[:2], pt[:2])  # only want horizontal distance...
            R_max = max(R, R_max)
        R = R_max

        normal = self.floor.mesh_normal/np.linalg.norm(self.floor.mesh_normal)  # USE FLOOR NORMAL ALWAYS
        H = self.opt_options.target_volume_height

        V_target = np.pi * R ** 2 * H
        if dpp is None:
            N = self.n_points
            # given desired number of points and target volume, the following point density is required:
            dpp = (V_target/N)**(1/3)
        elif N_target is not None:
            dpp = (V_target/N_target)**(1/3)

        if np.abs(np.dot(normal, np.array([1, 0, 0]))) < np.abs(np.dot(normal, np.array([0, 1, 0]))):
            test_vector = np.array([1, 0, 0])
        else:
            test_vector = np.array([0, 1, 0])

        dr_a = np.cross(normal, test_vector)
        dr_b = np.cross(normal, dr_a)

        n_pts = len(np.arange(-R, 0, dpp)) * len(np.arange(0, np.pi*2, R*dpp)) * len(np.arange(0, H + dpp, dpp))*10
        sampling_points = np.zeros([n_pts, 3])

        c = 0
        for i in np.arange(-R, 0, dpp):
            dpr = dpp / np.abs(i)
            for th in np.arange(0, np.pi*2, dpr):
                for k in np.arange(0, H + dpp, dpp):
                    th_ = th + dpr/13*k  # get the points to spiral a bit...
                    sampling_points[c, :] = O + i*(dr_a*np.sin(th_) + dr_b*np.cos(th_)) + normal * k
                    c += 1

        return sampling_points[:c]

    def generate_random_target_points(self, N=None):
        if N is None:
            N = self.n_points
        sampling_points = self.target.generate_points_on_mesh(N)
        # add vertical offset randomly to points (which are close enough to centroid...);
        pts_near_centroid = \
            np.abs(np.dot(sampling_points-self.target_centroid,
                          self.gravity_direction)) < self.opt_options.dist_threshold
        sampling_points[pts_near_centroid] -= self.opt_options.target_volume_height * \
            np.random.uniform(0, 1, sampling_points[pts_near_centroid].shape) * self.gravity_direction
        return sampling_points

    def generate_target_mesh(self, n_pts=5, shape="cylinder"):
        if shape == "cylinder":
            O = self.target.centroid
            O[-1] = self.floor.centroid[-1]

            R_max = 0.0
            for pt in self.target.points:
                R = dist(O[:2], pt[:2])
                R_max = max(R, R_max)
            H_vec = self.floor.mesh_normal / \
                    np.linalg.norm(self.floor.mesh_normal) * self.opt_options.target_volume_height
            target_mesh = generate_cylinder_mesh(O=O, R=R_max, H_vec=H_vec, num_pts=n_pts)

        elif shape == "box":
            H_vec = self.floor.mesh_normal / \
                    np.linalg.norm(self.floor.mesh_normal) * self.opt_options.target_volume_height
            target_mesh = generate_box_mesh(self.target, H_vec)

        else:
            target_mesh = generate_extruded_mesh(self.target)

        return target_mesh
