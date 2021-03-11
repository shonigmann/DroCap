import rospy
import rospkg
import copy
import numpy as np
import pyswarms as ps
import multiprocessing
import configparser
from opt.optimization_options import CameraPlacementOptions
from pso.pso_functions import evaluate_swarm, evaluate_particle, convert_particle_to_state, plot_particle_2d, \
    plot_particle_3d
from sim import environment_data as env_data
from sim.cameras import Camera, PlacedCameras
from sim.evaluate import evaluate_camera_covariance, evaluate_arrangement_covariance

from vis.draw3d import draw_room_3d
from matplotlib import pyplot as plt

from test.evaluate_solution import EvaluatePlacement

import time
from datetime import datetime

# TODO: create Ros Launch file to run everything in one go... (run plane partition, perch placement, gazebo, rviz)
# TODO: 13) Add virtual image generation (e.g. through RVIZ) to show views of placed cameras. Can export mesh into
#  gazebo compatible format using trimesh.exchange.dae.export_collada. use (reorient first) full_env_path mesh if
#  possible
# TODO: 1) implement dense reconstruction, mesh cleanup as part of a ROS node
# TODO: 2) import cameras list from config file(s)
# TODO: 4) https://ieeexplore.ieee.org/abstract/document/7892243 suggests you want to minimize overlap to reduce
#  interference between cameras... This likely depends on the camera type. It would be a problem with the Kinect I
#  could include interference data from here: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8436009
# TODO: 8) Implement custom boundary handling for polygonal search spaces
#  https://pyswarms.readthedocs.io/en/latest/api/pyswarms.single.html
# TODO: 9) Could compare separability of environment colors in HSV, LAB, etc color spaces
# TODO: 12) Consider adding a user-defined "preferred direction" to the fitness function
# TODO: 6) consider discretizing the zoom


def initialize(seg_env_prototype, target_prototype, cluster_env_path, full_env_path, load_full_env=False):
    np.random.seed(round(time.time()))

    # # loading configuration
    # config = configparser.ConfigParser()
    # rospack = rospkg.RosPack()
    # config.read(rospack.get_path("perch_placement") + "/src/config/opt.ini")
    # if seg_env_prototype:
    #     rospy.loginfo(rospy.get_caller_id() + seg_env_prototype)
    # if target_prototype:
    #     rospy.loginfo(rospy.get_caller_id() + target_prototype)
    # if cluster_env_path:
    #     rospy.loginfo(rospy.get_caller_id() + cluster_env_path)
    # if full_env_path:
    #     rospy.loginfo(rospy.get_caller_id() + full_env_path)
    #
    # # extract configuration values
    # env_conf = config['ENVIRONMENT']
    # mesh_env = env_conf.getboolean('mesh_env')
    # mesh_dir = env_conf['mesh_dir']
    #
    # if seg_env_prototype is None:
    #     env_file_base_name = env_conf['env_file_base_name']
    #     env_file_suffix = env_conf['env_file_suffix']
    #     seg_env_prototype = mesh_dir + env_file_base_name + "*" + env_file_suffix
    # if target_prototype is None:  # TODO: THIS MIGHT NOT BE THE BEST CHECK (E>G> IF YOU DON:T WANT TO SPECIFY
    #     target_file_base_name = env_conf['target_file_base_name']
    #     target_file_suffix = env_conf['target_file_suffix']
    #     target_prototype = mesh_dir + target_file_base_name + "*" + target_file_suffix
    # if cluster_env_path is None:
    #     cluster_env_path = env_conf['segmented_mesh_path']
    #
    # if full_env_path is None:
    #     full_env_path = env_conf['full_mesh_path']
    #
    # world_frame = np.asarray(np.matrix(env_conf['world_frame'])).squeeze()
    # reorient_mesh = env_conf.getboolean('reorient_mesh')
    # angle_threshold = env_conf.getfloat('angle_threshold')
    # dist_threshold = env_conf.getfloat('dist_threshold')
    # min_room_height = env_conf.getfloat('min_room_height')
    # erosion_raster_density = env_conf.getint('erosion_raster_density')
    # min_obstacle_radius = env_conf.getfloat('min_obstacle_radius')
    # nearest_neighbor_restriction = env_conf.getboolean('enable_nearest_neighbor_restriction')
    # target_volume_height = env_conf.getfloat('target_volume_height')
    #
    # cam_conf = config['CAMERA']
    # variable_pan = cam_conf.getboolean('variable_pan')
    # variable_tilt = cam_conf.getboolean('variable_tilt')
    # gimbal_limit = np.asarray(np.matrix(cam_conf['gimbal_limit'])).squeeze()
    # variable_zoom = cam_conf.getboolean('variable_zoom')
    # cam_fov = np.asarray(np.matrix(cam_conf['cam_fov'])).squeeze()
    # cam_range = np.asarray(np.matrix(cam_conf['cam_range'])).squeeze()
    # cam_alphas = np.asarray(np.matrix(cam_conf['cam_alphas'])).squeeze()
    # max_zoom = np.asarray(np.matrix(cam_conf['max_zoom'])).squeeze()
    # cam_resolution = np.asarray(np.matrix(cam_conf['cam_resolution'])).squeeze()
    # limit_range_by_minimum_feature = cam_conf.getboolean('limit_range_by_minimum_feature')
    # minimum_feature_size_m = cam_conf.getfloat('minimum_feature_size_m')
    # minimum_feature_size_px = cam_conf.getfloat('minimum_feature_size_px')
    #
    # drone_conf = config['DRONE']
    # perch_on_ceiling = drone_conf.getboolean('perch_on_ceiling')
    # perch_on_walls = drone_conf.getboolean('perch_on_walls')
    # land_on_floor = drone_conf.getboolean('land_on_floor')
    # perch_on_intermediate_angles = drone_conf.getboolean('perch_on_intermediate_angles')
    # min_perch_window = np.asarray(np.matrix(drone_conf['min_perch_window'])).squeeze()
    # perch_window_shape = drone_conf['perch_window_shape']
    # min_recovery_height = drone_conf.getfloat('min_recovery_height')
    #
    # variable_height = drone_conf.getboolean('variable_height')
    # frame_rad = drone_conf.getfloat('frame_rad')
    # prop_rad = drone_conf.getfloat('prop_rad')
    # camera_offset = np.asarray(np.matrix(drone_conf['camera_offset'])).squeeze()
    # camera_rotation = np.asarray(np.matrix(drone_conf['camera_rotation'])).squeeze()
    #
    # search_env = config['SEARCH']
    # n_cams = search_env.getint('n_cams')
    # angle_mode = search_env['angle_mode']
    # target_deviation = np.asarray(np.matrix(search_env['target_deviation'])).squeeze()
    # min_vertices = search_env.getint('min_vertices')
    # individual_surface_opt = search_env.getboolean('individual_surface_opt')
    # map_to_flat_surface = search_env.getboolean('map_to_flat_surface')
    # minimum_score_deviation = search_env.getfloat('minimum_score_deviation')
    # minimum_particle_deviation = search_env.getfloat('minimum_particle_deviation')
    # inside_out_search = search_env.getboolean('inside_out_search')
    #
    # if individual_surface_opt:
    #     surface_number = 0  # placeholder for now. will modify in loop below
    # else:
    #     surface_number = -1
    #
    # vary_position_over_face = search_env.getboolean('vary_position_over_face')
    # greedy_search = search_env.getboolean('greedy_search')
    # multi_threading = search_env.getboolean('multi_threading')
    # N_iterations = search_env.getint('N_iterations')
    # N_particles = search_env.getint('N_particles')
    # local_search = search_env.getboolean('local_search')
    # boundary_handling = search_env['boundary_handling']
    # pso_c1 = search_env.getfloat('pso_c1')
    # pso_c2 = search_env.getfloat('pso_c2')
    # pso_w = search_env.getfloat('pso_w')
    # pso_k = search_env.getint('pso_k')
    # pso_p = search_env.getint('pso_p')
    #
    # N_points = search_env.getint('N_points')
    # noise_resistant_particles = search_env.getint('noise_resistant_particles')
    # noise_resistant_sample_size = search_env.getint('noise_resistant_sample_size')
    #
    # # Optimizer Options:
    # optimization_options = CameraPlacementOptions(variable_pan=variable_pan, variable_tilt=variable_tilt,
    #                                               variable_zoom=variable_zoom, perch_on_ceiling=perch_on_ceiling,
    #                                               perch_on_intermediate_angles=perch_on_intermediate_angles,
    #                                               perch_on_walls=perch_on_walls, land_on_floor=land_on_floor,
    #                                               variable_height=variable_height, mesh_env=mesh_env,
    #                                               angle_mode=angle_mode, angle_threshold=angle_threshold,
    #                                               dist_threshold=dist_threshold, min_room_height=min_room_height,
    #                                               target_deviation=target_deviation, min_perch_window=min_perch_window,
    #                                               vary_position_over_face=vary_position_over_face,
    #                                               erosion_raster_density=erosion_raster_density,
    #                                               min_obstacle_radius=min_obstacle_radius,
    #                                               nearest_neighbor_restriction=nearest_neighbor_restriction,
    #                                               target_volume_height=target_volume_height,
    #                                               world_frame=world_frame, min_vertices=min_vertices,
    #                                               surface_number=surface_number,
    #                                               map_to_flat_surface=map_to_flat_surface,
    #                                               perch_window_shape=perch_window_shape,
    #                                               min_recovery_height=min_recovery_height,
    #                                               minimum_score_deviation=minimum_score_deviation,
    #                                               minimum_particle_deviation=minimum_particle_deviation,
    #                                               inside_out_search=inside_out_search,
    #                                               noise_resistant_particles=noise_resistant_particles,
    #                                               noise_resistant_sample_size=noise_resistant_sample_size)
    #
    # # Setup Cameras:
    # camera_list = []
    # for i in range(n_cams):
    #     camera_list.append(Camera(fov=cam_fov, camera_range=cam_range, score_range=cam_alphas, max_zoom=max_zoom,
    #                               camera_offset=camera_offset, camera_rotation=camera_rotation,
    #                               gimbal_limit=gimbal_limit, camera_id=i, prop_rad=prop_rad, frame_rad=frame_rad,
    #                               resolution=cam_resolution))
    #     if limit_range_by_minimum_feature:
    #         camera_list[i].update_range_to_minimum_feature_size(min_diameter_meters=minimum_feature_size_m,
    #                                                             min_diameter_pixel=minimum_feature_size_px)
    #
    # # Setup Environment:
    # if load_full_env:
    #     env = env_data.ply_environment(file_path_prototype=seg_env_prototype,
    #                                    target_path_prototype=target_prototype,
    #                                    cluster_env_path=cluster_env_path, optimization_options=optimization_options,
    #                                    reorient_mesh=reorient_mesh, N_points=N_points, full_env_path=full_env_path)
    # else:
    #     env = env_data.ply_environment(file_path_prototype=seg_env_prototype,
    #                                    target_path_prototype=target_prototype,
    #                                    cluster_env_path=cluster_env_path, optimization_options=optimization_options,
    #                                    reorient_mesh=reorient_mesh, N_points=N_points)
    #
    # pso_options = {
    #                 "pso_c1": pso_c1,
    #                 "pso_c2": pso_c2,
    #                 "pso_w": pso_w,
    #                 "pso_k": pso_k,
    #                 "pso_p": pso_p,
    #                 "boundary_handling": boundary_handling,
    #                 "local_search": local_search,
    #                 "N_particles": N_particles,
    #                 "N_iterations": N_iterations,
    #                 "multi_threading": multi_threading,
    #                 "greedy_search": greedy_search,
    #                 "individual_surface_opt": individual_surface_opt
    #                }
    #
    # # debugging
    # env.generate_target_mesh(shape="box")
    camera_list = initialize_camera_list()
    optimization_options = initialize_opt_options()
    pso_options = initialize_pso_options()
    env = initialize_env(seg_env_prototype, target_prototype, cluster_env_path, full_env_path, optimization_options,
                         load_full_env)

    return env, camera_list, optimization_options, pso_options


def initialize_opt_options():
    np.random.seed(round(time.time()))

    # loading configuration
    config = configparser.ConfigParser()
    rospack = rospkg.RosPack()
    config.read(rospack.get_path("perch_placement") + "/src/config/opt.ini")

    # extract configuration values
    env_conf = config['ENVIRONMENT']
    mesh_env = env_conf.getboolean('mesh_env')
    world_frame = np.asarray(np.matrix(env_conf['world_frame'])).squeeze()
    angle_threshold = env_conf.getfloat('angle_threshold')
    dist_threshold = env_conf.getfloat('dist_threshold')
    min_room_height = env_conf.getfloat('min_room_height')
    erosion_raster_density = env_conf.getint('erosion_raster_density')
    min_obstacle_radius = env_conf.getfloat('min_obstacle_radius')
    nearest_neighbor_restriction = env_conf.getboolean('enable_nearest_neighbor_restriction')
    target_volume_height = env_conf.getfloat('target_volume_height')

    cam_conf = config['CAMERA']
    variable_pan = cam_conf.getboolean('variable_pan')
    variable_tilt = cam_conf.getboolean('variable_tilt')
    variable_zoom = cam_conf.getboolean('variable_zoom')

    drone_conf = config['DRONE']
    perch_on_ceiling = drone_conf.getboolean('perch_on_ceiling')
    perch_on_walls = drone_conf.getboolean('perch_on_walls')
    land_on_floor = drone_conf.getboolean('land_on_floor')
    perch_on_intermediate_angles = drone_conf.getboolean('perch_on_intermediate_angles')
    min_perch_window = np.asarray(np.matrix(drone_conf['min_perch_window'])).squeeze()
    perch_window_shape = drone_conf['perch_window_shape']
    min_recovery_height = drone_conf.getfloat('min_recovery_height')

    variable_height = drone_conf.getboolean('variable_height')

    search_env = config['SEARCH']
    angle_mode = search_env['angle_mode']
    target_deviation = np.asarray(np.matrix(search_env['target_deviation'])).squeeze()
    min_vertices = search_env.getint('min_vertices')
    individual_surface_opt = search_env.getboolean('individual_surface_opt')
    map_to_flat_surface = search_env.getboolean('map_to_flat_surface')
    minimum_score_deviation = search_env.getfloat('minimum_score_deviation')
    minimum_particle_deviation = search_env.getfloat('minimum_particle_deviation')
    inside_out_search = search_env.getboolean('inside_out_search')

    if individual_surface_opt:
        surface_number = 0  # placeholder for now. will modify in loop below
    else:
        surface_number = -1

    vary_position_over_face = search_env.getboolean('vary_position_over_face')
    noise_resistant_particles = search_env.getint('noise_resistant_particles')
    noise_resistant_sample_size = search_env.getint('noise_resistant_sample_size')
    N_points = search_env.getint('N_points')

    # Optimizer Options:
    optimization_options = CameraPlacementOptions(variable_pan=variable_pan, variable_tilt=variable_tilt,
                                                  variable_zoom=variable_zoom, perch_on_ceiling=perch_on_ceiling,
                                                  perch_on_intermediate_angles=perch_on_intermediate_angles,
                                                  perch_on_walls=perch_on_walls, land_on_floor=land_on_floor,
                                                  variable_height=variable_height, mesh_env=mesh_env,
                                                  angle_mode=angle_mode, angle_threshold=angle_threshold,
                                                  dist_threshold=dist_threshold, min_room_height=min_room_height,
                                                  target_deviation=target_deviation, min_perch_window=min_perch_window,
                                                  vary_position_over_face=vary_position_over_face,
                                                  erosion_raster_density=erosion_raster_density,
                                                  min_obstacle_radius=min_obstacle_radius,
                                                  nearest_neighbor_restriction=nearest_neighbor_restriction,
                                                  target_volume_height=target_volume_height,
                                                  world_frame=world_frame, min_vertices=min_vertices,
                                                  surface_number=surface_number,
                                                  map_to_flat_surface=map_to_flat_surface,
                                                  perch_window_shape=perch_window_shape,
                                                  min_recovery_height=min_recovery_height,
                                                  minimum_score_deviation=minimum_score_deviation,
                                                  minimum_particle_deviation=minimum_particle_deviation,
                                                  inside_out_search=inside_out_search,
                                                  noise_resistant_particles=noise_resistant_particles,
                                                  noise_resistant_sample_size=noise_resistant_sample_size,
                                                  n_points=N_points)

    return optimization_options


def initialize_env(seg_env_prototype, target_prototype, cluster_env_path, full_env_path, optimization_options,
                   load_full_env=False):
    # loading configuration
    config = configparser.ConfigParser()
    rospack = rospkg.RosPack()
    config.read(rospack.get_path("perch_placement") + "/src/config/opt.ini")

    if seg_env_prototype:
        rospy.loginfo(rospy.get_caller_id() + seg_env_prototype)
    if target_prototype:
        rospy.loginfo(rospy.get_caller_id() + target_prototype)
    if cluster_env_path:
        rospy.loginfo(rospy.get_caller_id() + cluster_env_path)
    if full_env_path:
        rospy.loginfo(rospy.get_caller_id() + full_env_path)

    # extract configuration values
    env_conf = config['ENVIRONMENT']
    mesh_dir = env_conf['mesh_dir']

    if seg_env_prototype is None:
        env_file_base_name = env_conf['env_file_base_name']
        env_file_suffix = env_conf['env_file_suffix']
        seg_env_prototype = mesh_dir + env_file_base_name + "*" + env_file_suffix
    if target_prototype is None:  # TODO: THIS MIGHT NOT BE THE BEST CHECK (E>G> IF YOU DON:T WANT TO SPECIFY
        target_file_base_name = env_conf['target_file_base_name']
        target_file_suffix = env_conf['target_file_suffix']
        target_prototype = mesh_dir + target_file_base_name + "*" + target_file_suffix
    if cluster_env_path is None:
        cluster_env_path = env_conf['segmented_mesh_path']

    if full_env_path is None:
        full_env_path = env_conf['full_mesh_path']

    reorient_mesh = env_conf.getboolean('reorient_mesh')
    search_env = config['SEARCH']
    N_points = search_env.getint('N_points')

    # Setup Environment:
    if load_full_env:
        env = env_data.ply_environment(file_path_prototype=seg_env_prototype,
                                       target_path_prototype=target_prototype,
                                       cluster_env_path=cluster_env_path, optimization_options=optimization_options,
                                       reorient_mesh=reorient_mesh, N_points=N_points, full_env_path=full_env_path)
    else:
        env = env_data.ply_environment(file_path_prototype=seg_env_prototype,
                                       target_path_prototype=target_prototype,
                                       cluster_env_path=cluster_env_path, optimization_options=optimization_options,
                                       reorient_mesh=reorient_mesh, N_points=N_points)

    return env


def initialize_camera_list():
    # loading configuration
    config = configparser.ConfigParser()
    rospack = rospkg.RosPack()
    config.read(rospack.get_path("perch_placement") + "/src/config/opt.ini")

    cam_conf = config['CAMERA']
    gimbal_limit = np.asarray(np.matrix(cam_conf['gimbal_limit'])).squeeze()
    cam_fov = np.asarray(np.matrix(cam_conf['cam_fov'])).squeeze()
    cam_range = np.asarray(np.matrix(cam_conf['cam_range'])).squeeze()
    cam_alphas = np.asarray(np.matrix(cam_conf['cam_alphas'])).squeeze()
    max_zoom = np.asarray(np.matrix(cam_conf['max_zoom'])).squeeze()
    cam_resolution = np.asarray(np.matrix(cam_conf['cam_resolution'])).squeeze()
    limit_range_by_minimum_feature = cam_conf.getboolean('limit_range_by_minimum_feature')
    minimum_feature_size_m = cam_conf.getfloat('minimum_feature_size_m')
    minimum_feature_size_px = cam_conf.getfloat('minimum_feature_size_px')

    drone_conf = config['DRONE']
    frame_rad = drone_conf.getfloat('frame_rad')
    prop_rad = drone_conf.getfloat('prop_rad')
    camera_offset = np.asarray(np.matrix(drone_conf['camera_offset'])).squeeze()
    camera_rotation = np.asarray(np.matrix(drone_conf['camera_rotation'])).squeeze()

    search_env = config['SEARCH']
    n_cams = search_env.getint('n_cams')
    individual_surface_opt = search_env.getboolean('individual_surface_opt')

    # Setup Cameras:
    camera_list = []
    for i in range(n_cams):
        camera_list.append(Camera(fov=cam_fov, camera_range=cam_range, score_range=cam_alphas, max_zoom=max_zoom,
                                  camera_offset=camera_offset, camera_rotation=camera_rotation,
                                  gimbal_limit=gimbal_limit, camera_id=i, prop_rad=prop_rad, frame_rad=frame_rad,
                                  resolution=cam_resolution))
        if limit_range_by_minimum_feature:
            camera_list[i].update_range_to_minimum_feature_size(min_diameter_meters=minimum_feature_size_m,
                                                                min_diameter_pixel=minimum_feature_size_px)

    return camera_list


def initialize_pso_options():
    np.random.seed(round(time.time()))

    # loading configuration
    config = configparser.ConfigParser()
    rospack = rospkg.RosPack()
    config.read(rospack.get_path("perch_placement") + "/src/config/opt.ini")

    search_env = config['SEARCH']
    individual_surface_opt = search_env.getboolean('individual_surface_opt')
    greedy_search = search_env.getboolean('greedy_search')
    multi_threading = search_env.getboolean('multi_threading')
    N_iterations = search_env.getint('N_iterations')
    N_particles = search_env.getint('N_particles')
    local_search = search_env.getboolean('local_search')
    boundary_handling = search_env['boundary_handling']
    pso_c1 = search_env.getfloat('pso_c1')
    pso_c2 = search_env.getfloat('pso_c2')
    pso_w = search_env.getfloat('pso_w')
    pso_k = search_env.getint('pso_k')
    pso_p = search_env.getint('pso_p')

    pso_options = {
                    "pso_c1": pso_c1,
                    "pso_c2": pso_c2,
                    "pso_w": pso_w,
                    "pso_k": pso_k,
                    "pso_p": pso_p,
                    "boundary_handling": boundary_handling,
                    "local_search": local_search,
                    "N_particles": N_particles,
                    "N_iterations": N_iterations,
                    "multi_threading": multi_threading,
                    "greedy_search": greedy_search,
                    "individual_surface_opt": individual_surface_opt
                   }

    return pso_options
