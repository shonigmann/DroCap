#!/usr/bin/env python3

import copy
import pickle

import numpy as np
import pyswarms as ps
import multiprocessing
import trimesh
import vedo

from geom.geometry3d import rot3d_from_x_vec, rot3d_from_rtp
import ui.surface_confirmation as surface_confirmation
import ui.surface_confirmation_demo as surface_confirmation_demo
from ui.perch_confirmation import confirm_perch_placement

from pso.pso_functions import evaluate_swarm, convert_particle_to_state, PSO_Hyperparameters, run_pso
from sim.cameras import PlacedCameras
from sim.evaluate import evaluate_camera_covariance, evaluate_discrete_coverage
from sim.initialize import initialize, initialize_camera_list, initialize_opt_options, initialize_pso_options

from test.evaluate_solution import EvaluatePlacement

from datetime import datetime


# noinspection PyDeprecation,PyTypeChecker
def optimize(seg_env_prototype=None, target_prototype=None, cluster_env_path=None, full_env_path=None, N_its=1,
             enable_user_confirmation=True, preloaded_vars=None, visualize_with_vedo=False):
    
    if preloaded_vars is None:
        env, camera_list, optimization_options, pso_options = initialize(seg_env_prototype, target_prototype,
                                                                         cluster_env_path, full_env_path,
                                                                         load_full_env=True)

        if enable_user_confirmation:
            # surface_confirmation.confirm_surfaces(environment=env, N_max=10)
            surface_confirmation_demo.confirm_surfaces(environment=env, N_max=10)
        env.post_process_environment()

        preloaded_vars = {'env': copy.deepcopy(env),
                          'camera_list': copy.deepcopy(camera_list),
                          'optimization_options': copy.deepcopy(optimization_options),
                          'pso_options': copy.deepcopy(pso_options)}

        # SAVE THIS!
        pickle_env = open("../test/preloaded_environment_apartment.p", 'wb')
        pickle.dump(preloaded_vars, pickle_env)
        pickle_env.close()

    else:
        # work around to the seg fault issue... load everything in advance, then just pass it in!
        env = preloaded_vars['env']
        camera_list = initialize_camera_list()
        optimization_options = initialize_opt_options()
        pso_options = initialize_pso_options()

        env.vedo_mesh = vedo.mesh.Mesh(env.obs_mesh)
        env.opt_options = optimization_options
        env.correct_normals()
        env.n_points = optimization_options.n_points
        env.generate_integration_points()
        env.perch_regions = []
        env.perch_area = 0
        env.set_surface_as_perchable()
        optimization_options.log_performance = False

    # env.plot_environment()

    # PSO:
    # base dimension for simple 2d problem is 2. scales up depending on selected opt
    camera_particle_dimension = optimization_options.get_particle_size()
    n_cams = len(camera_list)
    N_iterations = pso_options["N_iterations"]
    N_particles = pso_options["N_particles"]

    if pso_options["greedy_search"]:
        particle_dimension = camera_particle_dimension
        num_optimizations = n_cams
    else:
        particle_dimension = n_cams * camera_particle_dimension
        num_optimizations = 1

    # for logging
    pso_keypoints = np.zeros([N_its, num_optimizations, N_iterations, 3])
    optimization_options.search_time = np.zeros([N_its, num_optimizations, N_iterations+1])
    optimization_options.best_fitness = np.zeros([N_its, num_optimizations, N_iterations+1])
    optimization_options.pts_searched = np.zeros([N_its, num_optimizations, N_iterations+1])

    bounds = (np.zeros(particle_dimension), np.ones(particle_dimension))  # particle boundaries

    # for velocity update:
    # 'w' is velocity decay aka inertia,
    # 'c1' is "cognitive parameter", e.g. attraction to particle best,
    # 'c2' is "social parameter", e.g. attraction to local/global best
    # 'k' = number of neighbors to consider
    # 'p' = the Minkowski p-norm. 1 for absolute val dist, 2 for norm dist
    options = {'c1': pso_options["pso_c1"],
               'c2': pso_options["pso_c2"],
               'w': pso_options["pso_w"],
               'k': pso_options["pso_k"],
               'p': pso_options["pso_p"]}

    optimal_cameras = PlacedCameras()

    fig_num = 1

    if pso_options['individual_surface_opt']:
        num_surface_loops = len(env.perch_regions)
        surface_number = range(num_surface_loops)
        N_particles = env.assign_particles_to_surfaces(N_particles, pso_options["pso_k"],
                                                       neighborhood_search=pso_options["local_search"])
    else:
        num_surface_loops = 1
        surface_number = [-1]
        N_particles = [N_particles]

    # STORE FOR ANALYSIS LATER
    if optimization_options.log_performance:
        pso_best_fitnesses = np.zeros(num_optimizations)
        pso_points_searched = np.zeros(num_optimizations)
        pso_search_time = np.zeros(num_optimizations)
        pso_keypoints = []
        for i in range(num_optimizations):
            pso_keypoints.append([[], [], [], []])

    # store, for each optimization, time, num particles searched, fitness, standard deviation
    bh = pso_options["boundary_handling"]

    # for i in range(num_optimizations):
    i = 0
    while i < num_optimizations:
        if optimization_options.log_performance:
            optimization_options.data_index = 0
        start_time = datetime.now()

        if pso_options["greedy_search"]:
            search_cameras_list = [copy.deepcopy(camera_list[i])]
        else:
            search_cameras_list = camera_list

        best_cost = np.finfo(float).max
        best_pos_surf = -1
        best_pos = np.zeros(particle_dimension)

        for j in range(num_surface_loops):
            # Future work: investigate velocity clamping...
            optimization_options.surface_number = j
            # if pso_options["local_search"]:
            #     optimizer = ps.single.LocalBestPSO(n_particles=N_particles[j], dimensions=particle_dimension,
            #                                        options=options, bounds=bounds, bh_strategy=bh)
            # else:
            #     optimizer = ps.single.GlobalBestPSO(n_particles=N_particles[j], dimensions=particle_dimension,
            #                                         options=options, bounds=bounds, bh_strategy=bh)

            optimization_options.surface_number = surface_number[j]
            # this flag gets reset if the current search has too little variance
            optimization_options.continue_searching = True
            optimization_options.stagnant_loops = 0

            if visualize_with_vedo:
                plt1 = vedo.Plotter(title='Confirm Perch Location', pos=[0, 0], interactive=False, sharecam=False)
                plt1.clear()

                # draw wireframe lineset of camera frustum
                # env_mesh = trimesh.load(env.full_env_path)

                env_mesh = trimesh.load('/home/simon/catkin_ws/src/mesh_partition/datasets/' + env.name + '_1m_pt1.ply')

                R = np.zeros([4, 4])
                R[:3, :3] = env.R
                env_mesh.vertices = trimesh.transform_points(env_mesh.vertices, R)
                env_mesh_vedo = vedo.mesh.Mesh(env_mesh)
                target_mesh_pymesh = env.generate_target_mesh(shape='box')
                target_mesh = trimesh.Trimesh(target_mesh_pymesh.vertices, target_mesh_pymesh.faces)
                target_mesh_vedo = vedo.mesh.Mesh(target_mesh)
                target_colors = 0.5 * np.ones([len(target_mesh.faces), 4])
                target_colors[:, 0] *= 0
                target_colors[:, 2] *= 0
                target_mesh_vedo.alpha(0.6)
                target_mesh_vedo.cellIndividualColors(target_colors, alphaPerCell=True)
                env_mesh.visual.face_colors[:, -1] = 255
                env_mesh_vedo.cellIndividualColors(env_mesh.visual.face_colors / 255, alphaPerCell=True)

                geom_list = [env_mesh_vedo, target_mesh_vedo]

                if env.name == 'office3' or env.name == 'apartment':
                    env_mesh2 = trimesh.load(
                        '/home/simon/catkin_ws/src/mesh_partition/datasets/' + env.name + '_1m_pt2.ply')
                    env_mesh_vedo2 = vedo.mesh.Mesh(env_mesh2)

                    env_mesh2.visual.face_colors[:, -1] = 150
                    env_mesh_vedo2.cellIndividualColors(env_mesh2.visual.face_colors / 255, alphaPerCell=True)
                    geom_list.append(env_mesh_vedo2)

                for s in env.perch_regions:
                    surf_mesh = trimesh.Trimesh(vertices=s.points, faces=s.faces)
                    vedo_surf_mesh = vedo.mesh.Mesh(surf_mesh)
                    vedo_surf_mesh.color('g')
                    vedo_surf_mesh.opacity(0.7)
                    geom_list.append(vedo_surf_mesh)

                for i_ in range(len(optimal_cameras.cameras)):
                    quad_mesh = trimesh.load(
                        "/home/simon/catkin_ws/src/perch_placement/src/ui/models/white-red-black_quad2.ply")
                    R = rot3d_from_x_vec(optimal_cameras.cameras[i_].wall_normal)
                    R2 = rot3d_from_rtp(np.array([0, -90, 0]))
                    R_aug = np.zeros([4, 4])
                    R_aug[:3, :3] = R.dot(R2)
                    R_aug[:3, -1] = optimal_cameras.cameras[i_].pose[:3]
                    quad_mesh.vertices = trimesh.transform_points(quad_mesh.vertices, R_aug)
                    quad_mesh_vedo = vedo.mesh.Mesh(quad_mesh)
                    quad_mesh_vedo.cellIndividualColors(quad_mesh.visual.face_colors / 255, alphaPerCell=True)

                    pymesh_frustum = optimal_cameras.cameras[i_].generate_discrete_camera_mesh(degrees_per_step=20,
                                                                                              environment=env)
                    pymesh_verts = pymesh_frustum.vertices.copy()
                    pymesh_verts.flags.writeable = True
                    pymesh_faces = pymesh_frustum.faces.copy()
                    pymesh_faces.flags.writeable = True

                    frustum = trimesh.Trimesh(vertices=pymesh_frustum.vertices.copy(),
                                              faces=pymesh_frustum.faces.copy())
                    vedo_frustum = vedo.mesh.Mesh(frustum)
                    vedo_frustum.alpha(0.3)
                    vedo_frustum.color("b")
                    quad_mesh_vedo.color('o')
                    geom_list.append(quad_mesh_vedo)
                    geom_list.append(vedo_frustum)

                for actor in geom_list:
                    plt1.add(actor)
            else:
                plt1 = None

            # if pso_options["multi_threading"]:
            #     # noinspection PyTypeChecker
            #     surf_best_cost, surf_best_pos = optimizer.optimize(evaluate_swarm, iters=N_iterations, environment=env,
            #                                                        cameras=search_cameras_list,
            #                                                        placed_cameras=optimal_cameras,
            #                                                        opt_options=optimization_options,
            #                                                        n_processes=multiprocessing.cpu_count(),
            #                                                        vedo_plt=plt1)
            # else:
            #     surf_best_cost, surf_best_pos = optimizer.optimize(evaluate_swarm, iters=N_iterations, environment=env,
            #                                                        cameras=search_cameras_list,
            #                                                        placed_cameras=optimal_cameras,
            #                                                        opt_options=optimization_options,
            #                                                        vedo_plt=plt1)

            pso_params = PSO_Hyperparameters(w=pso_options["pso_w"], c1=pso_options["pso_c1"], c2=pso_options["pso_c2"],
                                             lr=1, k=pso_options["pso_k"], p=pso_options["pso_p"],
                                             N_particles=N_particles[j], N_iterations=N_iterations)

            surf_best_cost, surf_best_pos = run_pso(fitness_function=evaluate_swarm, pso_hyper_parameters=pso_params,
                                                    environment=env, cameras=search_cameras_list,
                                                    placed_cameras=optimal_cameras, opt_options=optimization_options,
                                                    local_pso=True)

            if surf_best_cost < best_cost:
                best_cost = copy.deepcopy(surf_best_cost)
                best_pos = copy.deepcopy(surf_best_pos)
                if pso_options["individual_surface_opt"]:
                    best_pos_surf = j
                # print("Surface " + str(j) + " has lowest cost so far.. ")
                # print("Particle: " + str(best_pos))

        if optimization_options.log_performance:
            pso_search_time[i] = (datetime.now() - start_time).total_seconds()
            pso_best_fitnesses[i] = best_cost

        if pso_options["greedy_search"]:
            search_cameras_list_copy = copy.deepcopy(search_cameras_list)
            optimization_options.surface_number = best_pos_surf
            best_cam = convert_particle_to_state(environment=env, particle=best_pos, cameras=search_cameras_list_copy,
                                                 opt_options=optimization_options)[0]
            optimal_cameras.cameras.append(copy.deepcopy(best_cam))

            if enable_user_confirmation:
                if confirm_perch_placement(environment=env, placed_cameras=optimal_cameras.cameras, focus_id=i):
                    best_cam_covariances = evaluate_camera_covariance(environment=env, cameras=[best_cam])
                    optimal_cameras.append_covariances(best_cam_covariances)
                    i += 1
                else:
                    optimal_cameras.cameras.pop()
                    env.remove_rejected_from_perch_space(camera=best_cam, r=0.3)
            else:
                best_cam_covariances = evaluate_camera_covariance(environment=env, cameras=[best_cam])
                optimal_cameras.append_covariances(best_cam_covariances)
                i += 1

    evaluate_discrete_coverage(env.n_points, optimal_cameras, plot=True)

    return optimal_cameras.cameras
