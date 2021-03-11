import rospy
import numpy as np
from sim.environment import Environment, MeshEnvironment
from opt.optimization_options import CameraPlacementOptions
import open3d as o3d
from geom.geometry3d import import_ply, Mesh
import warnings
import glob

# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
import pymesh
import time
import copy


class KDTreeData:
    def __init__(self, tree, colors, clusters, cluster_ids, pts, faces, point_face_map):
        self.tree = tree
        self.colors = colors
        self.clusters = clusters
        self.cluster_ids = cluster_ids
        self.points = pts  # stores a list of points, including duplicates for points shared by multiple faces
        # (which may be in different clusters)
        self.faces = faces  # stores cluster faces
        self.point_face_map = point_face_map


def ply_environment(file_path_prototype, target_path_prototype=None, cluster_env_path=None,
                    optimization_options=CameraPlacementOptions(),
                    reorient_mesh=False, full_env_path=None, N_points=50):  # , surface_number=-1):

    rospy.loginfo(rospy.get_caller_id() + " Importing mesh environment")
    t0 = time.time()

    if cluster_env_path is not None:
        cluster_env = import_ply(cluster_env_path, post_process=False, options=optimization_options, is_plane_mesh=False)

        if optimization_options.nearest_neighbor_restriction:
            # use color codes to identify unique clusters of faces
            colors, clusters, cluster_ids = np.unique(cluster_env.face_colors, axis=0, return_index=True,
                                                      return_inverse=True)

            # make a map between point IDs and faces
            point_face_map = [[] for i in range(cluster_env.num_points)]

            for f in range(cluster_env.num_faces):
                for i in range(3):
                    point_face_map[cluster_env.faces[f, i]].append(f)

            tree = KDTree(cluster_env.points)  # make tree with set of unique points

            kdt_data = KDTreeData(tree, colors, clusters, cluster_ids, cluster_env.points, cluster_env.faces,
                                  point_face_map)
        else:
            kdt_data = None

        t1 = time.time()
        print("Preprocessing the cluster environment takes: " + str(t1-t0))
        t0 = time.time()

        # cluster_env_remeshed = pymesh.form_mesh(cluster_env.points, cluster_env.faces)
        # cluster_env_remeshed, _ = pymesh.split_long_edges(cluster_env_remeshed, optimization_options.dist_threshold)
        # cluster_env_remeshed = Mesh(cluster_env_remeshed.vertices, cluster_env_remeshed.faces,
        #                             options=optimization_options)  # TODO: REMOVE
        cluster_env_remeshed = copy.deepcopy(cluster_env)  # BANDAID SPEEDUP PRIOR TO PROPER REMOVAL.

        t1 = time.time()
        print("Remeshing the cluster environment takes: " + str(t1-t0))
        t0 = time.time()
    else:
        cluster_env = None
        kdt_data = None
        cluster_env_remeshed = None

    surfaces = []
    surf_files = glob.glob(file_path_prototype+"*")
    # print("Surf Files:")
    # print(str(surf_files))
    for fp in range(len(surf_files)):
        # print("Processing: " + str(fp))
        mesh = import_ply(surf_files[fp], post_process=False, options=optimization_options, kdt_data=kdt_data)
        if mesh.faces is None or mesh.points is None:
            print("PROBLEM!")
        if mesh.is_valid:
            surfaces.append(mesh)
        else:
            warnings.warn("Warning: mesh " + str(surf_files[fp]) + " omitted due to invalid geometry.")
    if len(surfaces) == 0:
        surfaces = None

    t1 = time.time()
    print("Loading min area surfaces took: " + str(t1-t0))
    t0 = time.time()

    if target_path_prototype is not None and target_path_prototype != "":
        surf_files = glob.glob(target_path_prototype)
        targets = []
        for fp in surf_files:
            # print("Processing: " + str(fp))
            mesh = import_ply(fp, post_process=False, options=optimization_options)
            if mesh.is_valid:
                targets.append(mesh)
            else:
                warnings.warn("Warning: mesh " + str(fp) + " omitted due to invalid geometry.")
        if len(targets) > 0:
            target = targets[0]
            for t in range(len(targets)):
                if t != 0:
                    target += targets[t]
        else:
            target = None
    else:
        target = None

    t1 = time.time()
    print("Loading target mesh took: " + str(t1 - t0))
    t0 = time.time()

    if full_env_path is not None:
        full_env = import_ply(full_env_path, is_plane_mesh=False)
    else:
        full_env = None

    t1 = time.time()
    print("Loading the full mesh environment took: " + str(t1 - t0))

    env = MeshEnvironment(surfaces=surfaces, n_points=N_points, target=target, obstacles=None, clust_env=cluster_env,
                          full_env=full_env, opt_options=optimization_options,
                          reorient_mesh=reorient_mesh, cluster_env_remeshed=cluster_env_remeshed,
                          full_env_path=full_env_path, clust_env_path=cluster_env_path)
    return env


def environment_1(obstructions=True, target="A", dimension=2, optimization_options=CameraPlacementOptions()):
    height = 2.75

    room_coords = np.array([[20, 10], [20, 20], [10, 20], [10, 15], [5, 15], [5, 10]])

    ceiling_coords = room_coords
    perch_areas = np.array([[[5.5, 10], [10.75, 10]],
                            [[13.5, 10], [18.75, 10]],
                            [[20, 10.25], [20, 18.75]],
                            [[19.75, 20], [10.5, 20]],
                            [[10, 19.25], [10, 15.25]],
                            [[7.5, 15], [5.25, 15]],
                            [[5, 14.75], [5, 10.5]]])

    if target == "A":  # center of main room portion
        target_coords = np.array([[13, 14], [15, 14], [14, 16], [12, 17]])
    elif target == "B":  # center of side annex
        target_coords = np.array([[6, 12], [8, 12], [7, 14], [5.5, 14.5]])
    elif target == "C":  # hugging bottom right corner
        target_coords = np.array([[18, 10.5], [19.5, 11], [19, 11.5], [18.5, 12]])
    else:  # entire room is target
        target_coords = room_coords

    obstruction1 = np.array(
        [[10, 9.98], [20.02, 9.98], [20.02, 20.02], [9.98, 20.02], [9.98, 15.02], [4.98, 15.02],
         [4.98, 9.98]])
    obstruction2 = np.array([[13, 12], [15, 12.5], [15, 13.], [13, 12.75]])
    obstruction3 = np.array([[13.25, 18], [14, 18], [14, 18.5], [13.25, 18.5]])

    if obstructions:
        obstruction_polygons = [obstruction1, obstruction2, obstruction3]
    else:
        obstruction_polygons = [obstruction1]  # room walls only

    using_o3d = False  #

    if dimension == 3:
        # IN 3D, DEFINE EVERYTHING AS OPEN3D POINT CLOUD
        ceiling_coords = add_z_axis(room_coords, height)
        target_coords = add_z_axis(target_coords, 0)
        # convert ceiling to point cloud
        if using_o3d:
            ceiling = o3d.geometry.PointCloud()
            ceiling.points = o3d.utility.Vector3dVector(ceiling_coords)
            ceiling_coords = ceiling

            ceiling_poly = convert_PointCloud_to_Mesh(ceiling_coords)
            # visualize_o3d(ceiling_coords)
            visualize_o3d(ceiling_poly)

            room_coords = convert_2D_poly_list_to_PointCloud(room_coords, height)
            perch_areas = convert_2D_poly_list_to_PointCloud(perch_areas, height)

            obstruction1 = convert_2D_poly_list_to_PointCloud(obstruction1, height)
            obstruction2 = convert_2D_poly_list_to_PointCloud(obstruction2, height)
            obstruction3 = convert_2D_poly_list_to_PointCloud(obstruction3, height)

        else:
            room_coords = convert_2D_Poly_to_3D_Walls(room_coords, height)

            pa = []
            for i in range(len(perch_areas)):
                pa.extend(convert_2D_Poly_to_3D_Walls_PA_ONLY(perch_areas[i, :, :], height))
            perch_areas = pa

            obstruction1 = convert_2D_Poly_to_3D_Walls(obstruction1, height)
            obstruction2 = convert_2D_Poly_to_3D_Walls(obstruction2, height)
            obstruction3 = convert_2D_Poly_to_3D_Walls(obstruction3, height)

        # compile all obstructions into a single list
        if obstructions:
            obstruction2.extend(obstruction3)
            obstruction1.extend(obstruction2)
            obstruction_polygons = obstruction1
        else:
            obstruction_polygons = obstruction1  # room walls only

    env = Environment(walls=room_coords, n_points=100, dimension=dimension, target=target_coords,
                      obstacles=obstruction_polygons, perch_regions=perch_areas,
                      opt_options=optimization_options, ceiling=ceiling_coords)

    return env


def convert_2D_Poly_to_3D_Walls(poly2d, height):
    poly3d = []

    for i in range(len(poly2d)):
        a1 = poly2d[i-1]
        a2 = poly2d[i]

        poly3d.append(np.array([np.append(a1, 0), np.append(a2, 0), np.append(a2, height), np.append(a1, height)]))

    return poly3d


def convert_2D_Poly_to_3D_Walls_PA_ONLY(poly2d, height):
    poly3d = []

    for i in range(len(poly2d)-1):
        a1 = poly2d[i-1]
        a2 = poly2d[i]

        poly3d.append(np.array([np.append(a1, 0), np.append(a2, 0), np.append(a2, height), np.append(a1, height)]))

    return poly3d


def convert_2D_poly_list_to_PointCloud(polygon_list, height):
    point_clouds = []
    for poly in polygon_list:
        poly = convert_2D_Poly_to_3D_Walls(poly, height)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(poly)
        point_clouds.append(pcd)

    return point_clouds


def convert_3D_poly_list_to_PointCloud(polygon_list):
    point_clouds = []
    for poly in polygon_list:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(poly)
        point_clouds.append(pcd)

    return point_clouds


def add_z_axis(poly_2d, z_offset=0.0):
    offset_poly = np.zeros([np.shape(poly_2d)[0], 3])
    offset_poly[:, 0:2] = poly_2d
    offset_poly[:, 2] = z_offset

    return offset_poly


def convert_PointCloud_to_Mesh(point_cloud):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.*avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, o3d.utility.DoubleVector(
        [radius, radius * 2]))
    return bpa_mesh


def visualize_o3d(o3d_geometry):
    o3d.visualization.draw_geometries([o3d_geometry])
