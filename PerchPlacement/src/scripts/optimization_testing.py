import copy
import time

import numpy as np
import pyswarms as ps
import multiprocessing

import vedo

from pso.pso_functions import evaluate_swarm, convert_particle_to_state
from sim.cameras import PlacedCameras
from sim.evaluate import evaluate_camera_covariance
from initialize import initialize, initialize_camera_list, initialize_opt_options, initialize_pso_options

from test.evaluate_solution import EvaluatePlacement

from datetime import datetime
import time


def optimize(seg_env_prototype=None, target_prototype=None, cluster_env_path=None, full_env_path=None, N_its=1,
             preloaded_vars=None):
    if preloaded_vars is None:
        env, camera_list, optimization_options, pso_options = initialize(seg_env_prototype, target_prototype,
                                                                         cluster_env_path, full_env_path)

        env.post_process_environment()
    else:
        # work around to the seg fault issue... load everything in advance, then just pass it in!
        env = preloaded_vars['env']
        camera_list = initialize_camera_list()
        optimization_options = initialize_opt_options()
        pso_options = initialize_pso_options()
        # camera_list = preloaded_vars['camera_list']
        # optimization_options = preloaded_vars['optimization_options']
        # pso_options = preloaded_vars['pso_options']

        env.vedo_mesh = vedo.mesh.Mesh(env.obs_mesh)
        env.opt_options = optimization_options
        env.correct_normals()
        env.n_points = optimization_options.n_points
        env.generate_integration_points()
        env.perch_regions = []
        env.perch_area = 0
        env.set_surface_as_perchable()
        optimization_options.log_performance = True

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

    np.random.seed(round(time.time()))

    # for logging
    pso_keypoints = np.zeros([N_its, num_optimizations, N_iterations + 1, 3], dtype=float)
    optimization_options.search_time = np.zeros([N_its, num_optimizations, N_iterations + 1], dtype=float)
    optimization_options.best_fitness = np.ones([N_its, num_optimizations, N_iterations + 1], dtype=float) * env.n_points
    optimization_options.pts_searched = np.zeros([N_its, num_optimizations, N_iterations + 1], dtype=int)

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
    pso_best_fitnesses = np.ones([N_its, num_optimizations])*env.n_points
    pso_points_searched = np.zeros([N_its, num_optimizations])
    pso_search_time = np.zeros([N_its, num_optimizations])
    pso_best_cameras = []

    # store, for each optimization, time, num particles searched, fitness, standard deviation
    bh = pso_options["boundary_handling"]

    for it in range(N_its):
        print("starting iteration: " + str(it+1))
        optimization_options.iteration = it
        optimal_cameras = PlacedCameras()

        for i in range(num_optimizations):
            if pso_options["greedy_search"]:
                search_cameras_list = [copy.deepcopy(camera_list[i])]
            else:
                search_cameras_list = camera_list

            best_cost = np.finfo(float).max
            best_pos_surf = -1
            best_pos = np.zeros(particle_dimension)

            for j in range(num_surface_loops):

                optimization_options.data_index = 0
                optimization_options.search_time[it, i, 0] = time.time()

                start_time = datetime.now()

                optimization_options.surface_number = j
                # Future work: investigate velocity clamping...
                if pso_options["local_search"]:
                    optimizer = ps.single.LocalBestPSO(n_particles=N_particles[j], dimensions=particle_dimension,
                                                       options=options, bounds=bounds, bh_strategy=bh)
                                                       # velocity_clamp=[-.55, 0.25])
                else:
                    optimizer = ps.single.GlobalBestPSO(n_particles=N_particles[j], dimensions=particle_dimension,
                                                        options=options, bounds=bounds, bh_strategy=bh)
                                                        # velocity_clamp=[-.25, 0.25])

                optimization_options.surface_number = surface_number[j]
                # this flag gets reset if the current search has too little variance
                optimization_options.continue_searching = True
                optimization_options.stagnant_loops = 0
                if pso_options["multi_threading"]:
                    # noinspection PyTypeChecker
                    surf_best_cost, surf_best_pos = optimizer.optimize(evaluate_swarm, iters=N_iterations, environment=env,
                                                                       cameras=search_cameras_list,
                                                                       placed_cameras=optimal_cameras,
                                                                       opt_options=optimization_options,
                                                                       n_processes=multiprocessing.cpu_count())
                else:
                    surf_best_cost, surf_best_pos = optimizer.optimize(evaluate_swarm, iters=N_iterations, environment=env,
                                                                       cameras=search_cameras_list,
                                                                       placed_cameras=optimal_cameras,
                                                                       opt_options=optimization_options)

                if surf_best_cost < best_cost:
                    best_cost = copy.deepcopy(surf_best_cost)
                    best_pos = copy.deepcopy(surf_best_pos)
                    if pso_options["individual_surface_opt"]:
                        best_pos_surf = j

            pso_search_time[it, i] = (datetime.now() - start_time).total_seconds()
            pso_best_fitnesses[it, i] = best_cost

            if pso_options["greedy_search"]:
                search_cameras_list_copy = copy.deepcopy(search_cameras_list)
                optimization_options.surface_number = best_pos_surf
                best_cam = convert_particle_to_state(environment=env, particle=best_pos, cameras=search_cameras_list_copy,
                                                     opt_options=optimization_options)[0]

                # from sim.evaluate import evaluate_arrangement_covariance
                # print("Score should be:" + str(evaluate_arrangement_covariance(env, [best_cam])))

                best_cam_covariance = evaluate_camera_covariance(environment=env, cameras=[best_cam])

                optimal_cameras.append_covariances(copy.deepcopy(best_cam_covariance))
                optimal_cameras.cameras.append(copy.deepcopy(best_cam))

        for i in range(num_optimizations):
            pso_points_searched[it, i] = np.sum(optimization_options.pts_searched[it, i, :])

        pso_best_cameras.append(copy.deepcopy(optimal_cameras))

    pso_keypoints[:, :, :, 0] = copy.deepcopy(optimization_options.search_time)
    pso_keypoints[:, :, :, 1] = copy.deepcopy(optimization_options.pts_searched)
    pso_keypoints[:, :, :, 2] = copy.deepcopy(optimization_options.best_fitness)
    # pso_keypoints[:, :, :, 3] = copy.deepcopy(optimization_options.fitness_deviation)
    for i in range(pso_keypoints.shape[1]):
        pso_keypoints[:, i, :, 0] -= np.reshape(pso_keypoints[:, i, 0, 0], [-1, 1])  # start time at 0s
    pso_keypoints[:, :, :, 1] = np.cumsum(pso_keypoints[:, :, :, 1], axis=2)  # cumsum of points evaluated...
    pso_keypoints[:, :, :, 2] = np.minimum.accumulate(pso_keypoints[:, :, :, 2], axis=2)  # cumulative minimum of best fitnesses

    verts = np.zeros([0, 3])
    faces = np.zeros([0, 3])
    face_colors = np.zeros([0, 4])
    for s in env.perch_regions:
        if s.is_valid:
            faces = np.vstack([faces, s.faces + len(verts)])
            verts = np.vstack([verts, s.points])
            fc = np.ones([len(s.faces), 4])
            fc[:, 3] *= 255
            fc[:, :3] *= s.face_colors[0, :]
            # fc = np.concatenate([s.face_colors, 255*np.ones([s.face_colors.shape[0], 1])], axis=1)
            face_colors = np.vstack([face_colors, fc])

    eval_placement = EvaluatePlacement(environment=env, ground_truth_environment=None,
                                       placed_cameras=pso_best_cameras[0].cameras,
                                       optimization_options=optimization_options,
                                       # sp_linear_density=.2, sp_angular_density=2.5,
                                       sp_linear_density=.2, sp_angular_density=5,
                                       gs_angular_density=20, gs_linear_density=.2,
                                       n_target=pso_options["N_particles"]*N_iterations)

    eval_placement.pso_best_fitnesses = pso_best_fitnesses
    eval_placement.pso_points_searched = pso_points_searched
    eval_placement.pso_search_time = pso_search_time
    eval_placement.pso_keypoints = pso_keypoints

    print("Done running PSO... Running analysis now.")
    eval_placement.run_analysis(N_its=N_its, pso=True, gridsearch=False, randomsearch=False, bruteforce=False)

    return pso_best_cameras[0].cameras
