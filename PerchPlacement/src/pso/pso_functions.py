import colorsys

import numpy as np
import trimesh
import vedo
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geom.geometry2d import vec2eul, min_ang_dist
from geom.geometry3d import dist, rot3d_from_z_vec, rot3d_from_rtp, rot3d_from_x_vec
from opt.optimization_options import CameraPlacementOptions
from sim.evaluate import evaluate_arrangement_covariance
from sim.cameras import PlacedCameras
from vis.draw2d import draw_boundary_2d, place_cameras_2d
from vis.draw3d import draw_room_3d
import time

# from progressbar.bar import ProgressBar
from tqdm import tqdm, trange


class PSO_Hyperparameters:
    def __init__(self, w, c1, c2, lr, k, p, N_iterations, N_particles):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.lr = lr
        self.k = k
        self.p = p
        self.N_iterations = N_iterations
        self.N_particles = N_particles


def run_pso(fitness_function, pso_hyper_parameters, environment, cameras, placed_cameras, opt_options, local_pso=True,
            map_to_2d=False):
    # extract relevant parameters
    N_particles = pso_hyper_parameters.N_particles
    D_particle = opt_options.get_particle_size()

    assert D_particle == pso_hyper_parameters.p_bounds[0].size

    inertia = pso_hyper_parameters.w
    social = pso_hyper_parameters.c1
    cognitive = pso_hyper_parameters.c2
    learning_rate = pso_hyper_parameters.lr
    N_iterations = pso_hyper_parameters.N_iterations
    p_bounds = pso_hyper_parameters.p_bounds

    assert p_bounds.shape == (2, D_particle, 3)

    current = 0
    best = 1
    velocity = 2

    #    Initialize the particle's velocity: vi ~ U(-|bup-blo|, |bup-blo|)
    #    Initialize the particle's position with a uniformly distributed random vector: xi ~ U(blo, bup)
    p = np.random.uniform(p_bounds[0].reshape(1, 1, D_particle, 3), p_bounds[1].reshape(1, 1, D_particle, 3),
                          [N_iterations, N_particles, D_particle, 3])

    # TODO: if map_to_2d, check particle validity. resample as necessary

    #    Initialize the particle's best known position to its initial position: pi ← xi
    p[:, :, :, best] = p[:, :, :, current]

    f = np.zeros([N_iterations, N_particles, 2])

    if local_pso:
        N_neighbors = pso_hyper_parameters.k
        assert N_neighbors <= N_particles
        # get indices for ring topology neighborhoods
        neighbors = (np.linspace([i for i in range(N_neighbors)], [i + N_particles-1 for i in range(N_neighbors)],
                                N_particles) % N_particles).astype(int)
        g = np.zeros([N_iterations, N_particles, D_particle])
        g_best = np.zeros([N_iterations, N_particles])

    else:
        g = np.zeros([N_iterations, D_particle])
        g_best = np.zeros(N_iterations)
        neighbors = None

    optimizing = True
    with trange(N_iterations) as t:
        for i in t:
            t.set_description('Best Fitness %f' % np.max(g_best))

            # TODO: add stop condition
            if not optimizing:
                break

            # evaluate fitness
            f[i, :, current] = fitness_function(p[i, :, :, current], environment, cameras, placed_cameras, opt_options,
                                                maximization=True).copy()
            # (x, environment, cameras, placed_cameras=PlacedCameras(),
            #  opt_options=CameraPlacementOptions(), debug=False, figure_number=0, vedo_plt=None):

            # update local best scores and particles
            i_improved = f[i, :, current] > f[i, :, best]
            f[i, :, best] = f[i-1, :, best].copy()  # copy previous best
            f[i, i_improved, best] = f[i, i_improved, current].copy()  # update new best fitness
            p[i, :, :, best] = p[i-1, :, :, best].copy()  # copy previous best particles
            p[i, i_improved, :, best] = p[i, i_improved, :, current].copy()  # update new best particles

            # Pick random numbers: rp, rg ~ U(0,1)
            rp = np.random.uniform(0, 1, [N_particles])
            rg = np.random.uniform(0, 1, [N_particles])

            # Update the particle's velocity: vi,d ← ω vi,d + φp rp (pi,d-xi,d) + φg rg (gd-xi,d)
            if local_pso:
                # find neighborhood best, then compare it to the value on record
                idx_best = np.argmax(f[i, neighbors, current], axis=1)
                foi = f[i, neighbors, current]
                has_improved = foi[np.arange(foi.shape[0]), idx_best] > g_best[i-1]
                poi = p[i, neighbors, :, best]
                g[i, has_improved] = poi[np.arange(poi.shape[0]), idx_best][has_improved].copy()
                g_best[i] = foi[np.arange(foi.shape[0]), idx_best].copy()

            else:
                # update global best
                idx_best = np.argmax(f[i, i_improved, best])
                if f[i, i_improved, best][idx_best] > g_best:
                    g = p[i, i_improved, best][idx_best].copy()
                    g_best = f[i, i_improved, best][idx_best].copy()

            # update particle velocity
            p[i, :, :, velocity] = inertia * p[i, :, :, velocity] + \
                                np.einsum('i,ik->ik', rp, cognitive * (p[i, :, :, best] - p[i, :, :, current])) + \
                                np.einsum('i,ik->ik', rg, social * (g[i] - p[i, :, :, current]))

            # get next particle positions
            if i < N_iterations - 1:
                # TODO: add boundary handling
                p[i+1, :, :, current] = learning_rate * p[i, :, :, velocity] + p[i, :, :, current]
                # get indices of particles that are invalid

                # TODO: if map_to_2d: check in on surface, correct if not (wrap around? slow? reflect?)


    return g, g_best


def get_invalid_particles(particles, bounds, bound_pos=True, bound_vel=False, map_to_2d=False, surf=None)
    valid_particles = np.ones_like(particles[0, :, :], dtype=bool)

    if bound_pos:
        valid_particles *= (particles[:,:,0] > bounds[0,0]).all(axis=1)
        valid_particles *= (particles[:,:,0] < bounds[1,0]).all(axis=1)

    if bound_vel:
        valid_particles *= (particles[:,:,2] > bounds[0,2]).all(axis=1)
        valid_particles *= (particles[:,:,2] < bounds[1,2]).all(axis=1)

    if map_to_2d:
        valid_particles *= is_on_surface(particles[:, :, 0], surf)

    return valid_particles


def is_on_surface(particles, surface):
    on_surface = np.ones_like(particles[:, 0])
    x = convert_particles_to_surf_positions()


def convert_particles_to_surf_positions(particles, opt_options, environment):

    ppc = opt_options.get_particle_size()  # particles per camera
    n_cameras = particles[0, :] // ppc
    pos3d = np.zeros(n_cameras, 3)

    particle_index = 1  # 1D by default (to select face center on preselected surface)
    surf_index = -1
    if opt_options.surface_number < 0:
        particle_index += 1  # extra term required for surface selection

    elif 0 <= opt_options.surface_number < len(environment.perch_regions):
        surf_index = opt_options.surface_number
    else:
        print("ERROR: SURFACE ID TOO LARGE. THIS SHOULD NOT HAPPEN")
        pos3d *= np.nan

    if opt_options.map_to_flat_surface:
        particle_index += 1
    elif opt_options.vary_position_over_face:
        particle_index += 2

    for i in range(n_cameras):
        surf_points, wall_normals, on_ceiling, surf_id = \
            environment.map_particle_to_surface_vectorized(particles[:, :ppc + particle_index],
                                                surface_index=surf_index, map_to_flat=opt_options.map_to_flat_surface)
    if np.isnan(surf_points).any():
        pos3d *= np.nan

    Rs = rot3d_from_z_vec(wall_normals)
    # cameras[i].pose[0:3] = surf_point + np.dot(R, cameras[i].camera_offset)
    return surf_points, Rs


def map_line_to_edge(environment, x, poly_order="CCW"):
    """
    DEPRECATED
    Maps a particle weight into a position in 2D space, by mapping the distance along the perimeter edge

    :param environment: Environment class instance
    :param x: particle weight
    :param poly_order: polygon point order (clockwise ("CW") or counter clockwise ("CCW"). determines sign of normal
    direction offset
    :return: point position, p, and normal direction euler angles, n
    """
    p = np.array([-np.inf, -np.inf])
    cumulative_dist = [0]

    perimeter = environment.perch_perimeter
    poly_bounds = environment.perch_regions
    x = x % perimeter
    n = -np.inf  # initialize normal direction as inf as a debugging precaution

    while n == -np.inf:
        for pb in poly_bounds:
            d = dist(pb[0, :], pb[1, :])
            cumulative_dist.append(cumulative_dist[-1] + d)

            if cumulative_dist[-1] >= x:
                # find intersection point
                dist_on_edge = x - cumulative_dist[-2]
                norm_dist = dist_on_edge / d
                p[0] = norm_dist * (pb[0, 0] - pb[1, 0]) + pb[1, 0]
                p[1] = norm_dist * (pb[0, 1] - pb[1, 1]) + pb[1, 1]

                # get normal direction as an angle WRT the +x direction
                edge_dir = (np.arctan2(pb[1, 1] - pb[0, 1],
                                       pb[1, 0] - pb[0, 0])) * 180 / np.pi
                if poly_order == "CCW":
                    n = edge_dir - 90  # normal direction to the edge
                else:
                    n = edge_dir + 90

                break

    if n == -np.inf:
        print("Warning: invalid edge normal")
        print(x)
        print(cumulative_dist)

    R = rot3d_from_rtp(np.array([0, 0, n]))
    n = R[:, 0]

    return p, n


def evaluate_swarm(x, environment, cameras, placed_cameras=PlacedCameras(),
                   opt_options=CameraPlacementOptions(), debug=False, figure_number=0, vedo_plt=None,
                   maximization=False):
    """
    The main function used by PySwarms to evaluate particles corresponding with camera placement

    :param x: the list of particles returned by PySwarms for the current iteration
    :param environment: Environment Class instance containing the environment in which to place cameras
    :param cameras: list of cameras, in order of which they are placed
    :param placed_cameras: PlacedCameras class instance
    :param opt_options: CameraPlacementOptions class containing opt opt
    :param debug: flag, if true print and draw relevant information.
    :param figure_number: figure window number on which to plot placements
    :return: particle scores
    """
    particles = np.copy(x)
    n_particles = np.shape(particles)[0]  # extract swarm dimensions

    # pts_searched = 0
    # fitness_deviation = 0
    # max_fitness = np.finfo(float).max

    if not opt_options.continue_searching:
        # print("would stop here")
        return np.ones(n_particles)*np.finfo(float).max/2

    else:
        # Check particle variance
        if opt_options.minimum_particle_deviation is not None and opt_options.minimum_particle_deviation > 0:
            deviations = np.std(particles, axis=0)
            # print("Particle Deviations: " + str(deviations))
            if (deviations < opt_options.minimum_particle_deviation).all():
                # print("Ending search early. Particle deviation too low.")
                opt_options.continue_searching = False
                # print("Would stop here")
                return np.ones(n_particles)*np.finfo(float).max/2

        results = np.ones([n_particles, 1])*np.inf  # start with max possible score...

        # print("There are " + str(len(placed_cameras.cameras)) + "placed cameras")

        for p in range(n_particles):
            results[p] = evaluate_particle(particle=particles[p, :], environment=environment, cameras=cameras,
                                           placed_cameras=placed_cameras, opt_options=opt_options, debug=debug,
                                           vedo_plt=vedo_plt, maximization=maximization)

        # if opt_options.noise_resistant_particles > 0:
        #     n_top = min(opt_options.noise_resistant_particles, n_particles)
        #     top_particles = np.argpartition(np.squeeze(results), -n_top)[-n_top:]
        #     environment.noise_resistant_search = True
        #     for p in top_particles:
        #         results[p] = evaluate_particle(particle=particles[p, :], environment=environment, cameras=cameras,
        #                                        placed_cameras=placed_cameras, opt_options=opt_options, debug=debug,
        #                                        maximization=maximization)
        #     environment.noise_resistant_search = False

        if opt_options.log_performance:
            c = len(placed_cameras.cameras)
            i = opt_options.iteration
            j = opt_options.data_index + 1

            opt_options.search_time[i, c, j] = time.time()
            # opt_options.fitness_deviation[i, c, j] = np.nanstd(results)
            opt_options.best_fitness[i, c, j] = np.nanmin(results)
            opt_options.pts_searched[i, c, j] = n_particles

            opt_options.data_index += 1

        if debug:
            if environment.dimension == 2:
                plot_particles_in_2d_search_space(particles, figure_number)

        # if environment.opt_options.map_to_flat_surface:
        #     plot_particles_on_2D_surface(particles, surf=environment.perch_regions[environment.opt_options.surface_number],
        #                                  figure_num=1, view_time=0.1, persistent_points=False)

        # fig = plt.figure(1)
        # ax = Axes3D(fig)
        # plot_particles_on_3D_surface(particles, environment=environment, surf_id=environment.opt_options.surface_number,
        #                              ax=ax, view_time=0.1, persistent_points=False, ppc=opt_options.get_particle_size())

        if opt_options.minimum_score_deviation is not None and opt_options.minimum_score_deviation > 0:
            deviation = np.std(results)
            # print("Score Deviation: " + str(deviation))
            if (deviation < opt_options.minimum_score_deviation).all():
                opt_options.stagnant_loops += 1
                # print("Score variation too low... (# consecutive stagnant loops = " +
                #       str(opt_options.stagnant_loops) + ")")
                # print("Scores: " + str(results.reshape([-1])))
                if opt_options.stagnant_loops >= opt_options.max_stagnant_loop_count:
                    # print("Ending search early. Score deviation too low.")
                    opt_options.continue_searching = False
            else:
                opt_options.stagnant_loops = 0

        # return np.squeeze(results)
        return results.reshape([-1])

def evaluate_particle(particle, environment, cameras, placed_cameras=PlacedCameras(),
                      opt_options=CameraPlacementOptions(), debug=False, vedo_plt=None, maximization=False):
    """
    This function evaluates a single particle using the heuristic function defined in evaluate_arrangement()

    :param particle: individual particle as np.array
    :param environment: Environment class instance
    :param cameras: a list of Camera objects corresponding to the cameras which have already been placed in an
     environment
    :param placed_cameras: PlacedCameras class instance
    :param opt_options: CameraPlacementOptions class containing opt opt
    :param debug: flag, if true print and draw relevant information.
    :return: score of particle
    """
    # convert particle into camera pose; evaluate edge normal at each camera position
    cameras = convert_particle_to_state(environment=environment, particle=particle, cameras=cameras,
                                        opt_options=opt_options, debug=debug)

    # plot particles to debug...
    # if debug:
    #     if environment.dimension == 2:
    #         plot_particle_2d(particle=np.squeeze(particle), environment=environment, cameras=cameras,
    #                          placed_cameras=placed_cameras, view_time=0.0001, figure_number=figure_number)
    #     else:
    # # if len(placed_cameras.cameras) > 0:
    # plot_particle_3d(particle=particle, environment=environment, cameras=cameras, placed_cameras=placed_cameras,
    #                  view_time=0.0001, figure_number=figure_number)

    score = evaluate_arrangement_covariance(environment=environment, cameras=cameras, placed_cameras=placed_cameras,
                                            debug=debug, maximization=maximization)

    if vedo_plt is not None:
        if len(placed_cameras.cameras) >= 10:
            geom_list = []
            for i in range(len(cameras)):
                pymesh_frustum = cameras[i].generate_discrete_camera_mesh(degrees_per_step=5, environment=environment)
                if len(pymesh_frustum.faces) > 0 and not np.isnan(pymesh_frustum.vertices).any():
                    pymesh_verts = pymesh_frustum.vertices.copy()
                    pymesh_verts.flags.writeable = True
                    pymesh_faces = pymesh_frustum.faces.copy()
                    pymesh_faces.flags.writeable = True
                    frustum = trimesh.Trimesh(vertices=pymesh_frustum.vertices.copy(),
                                              faces=pymesh_frustum.faces.copy())
                    vedo_frustum = vedo.mesh.Mesh(frustum)
                    vedo_frustum.alpha(0.2)
                    vedo_frustum.color("c")
                    quad_mesh = trimesh.load(
                        "/home/simon/catkin_ws/src/perch_placement/src/ui/models/white-red-black_quad2.ply")
                    R = rot3d_from_x_vec(cameras[i].wall_normal)
                    R2 = rot3d_from_rtp(np.array([0, -90, 0]))
                    R_aug = np.zeros([4, 4])
                    R_aug[:3, :3] = R.dot(R2)
                    R_aug[:3, -1] = cameras[i].pose[:3]
                    quad_mesh.vertices = trimesh.transform_points(quad_mesh.vertices, R_aug)
                    quad_mesh_vedo = vedo.mesh.Mesh(quad_mesh)
                    quad_mesh_vedo.cellIndividualColors(quad_mesh.visual.face_colors / 255, alphaPerCell=True)
                    geom_list.append(quad_mesh_vedo)
                    geom_list.append(vedo_frustum)

                    for actor in geom_list:
                        vedo_plt.add(actor)

                    # p_.camera_position = [
                    #     (R * np.cos(t), R * np.sin(t), z),
                    #     (c[0], c[1], c[2]),  # (-0.026929191045848594, 0.5783514020506139, 0.8268966663940324),
                    #     (0, 0, 1),
                    # ]
                    vedo_plt.camera.SetPosition(7*np.cos(-145*np.pi/180.0), 7*np.sin(-145*np.pi/180.0), 6.25)
                    vedo_plt.camera.SetFocalPoint(-0.026929191045848594, 0.5783514020506139, 0.9268966663940324)
                    vedo_plt.camera.SetViewUp(np.array([0, 0, 1]))
                    vedo_plt.camera.SetDistance(7.8)
                    vedo_plt.camera.SetClippingRange([0.25, 30])
                    vedo_plt.camera
                    vedo_plt.show(interactive=False, rate=30, resetcam=False, fullscreen=True)
                    time.sleep(0.5)
                    actors = vedo_plt.actors
                    for i in range(len(cameras)):
                        vedo_plt.remove(actors.pop())
                        vedo_plt.remove(actors.pop())

        # plot_particle_3d(particle=particle, environment=environment, cameras=cameras, placed_cameras=placed_cameras,
        #                  view_time=0.0001, figure_number=0)
    # print("Particle score: " + str(score) + "; Pose: " + str(cameras[0].pose))

    if debug:
        print(score)

    return score


def convert_particle_to_state(environment, particle, cameras, opt_options=CameraPlacementOptions(), debug=False):
    """
    Converts the weights of the particle into the state of the Camera to be evaluated

    :param environment: Environment Class instance containing the environment in which to place cameras
    :param particle: individual particle as np.array
    :param cameras: a list of Camera objects corresponding to the cameras which have already been placed in an
     environment
    :param opt_options: CameraPlacementOptions class containing opt opt
    :param debug: flag, if true print and draw relevant information.
    :return: list of cameras with updated state
    """

    n_cameras = len(cameras)
    ppc = opt_options.get_particle_size()  # particles per camera

    for i in range(n_cameras):

        particle_index = 0

        if opt_options.mesh_env:
            particle_index = 1  # 1D by default (to select face center on preselected surface)
            surf_index = -1
            if opt_options.surface_number < 0:
                particle_index += 1  # extra term required for surface selection

            elif 0 <= opt_options.surface_number < len(environment.perch_regions):
                surf_index = opt_options.surface_number
            else:
                print("ERROR: SURFACE ID TOO LARGE. THIS SHOULD NOT HAPPEN")
                cameras[i].pose *= np.nan
                break

            if opt_options.map_to_flat_surface:
                particle_index += 1
            elif opt_options.vary_position_over_face:
                particle_index += 2

            if opt_options.map_to_flat_surface:
                surf_point, cameras[i].wall_normal, on_ceiling, surf_id = \
                    environment.map_particle_to_flattened_surface(particle[i * ppc:i*ppc+particle_index],
                                                                surface_index=surf_index)
                if np.isnan(surf_point).any():
                    cameras[i].pose[:3] *= np.nan
                    break

            else:
                surf_point, cameras[i].wall_normal, on_ceiling, surf_id = \
                    environment.map_particle_to_surface(particle[i * ppc:i*ppc+particle_index],
                                                        surface_index=surf_index)

            R = rot3d_from_z_vec(cameras[i].wall_normal)
            cameras[i].pose[0:3] = surf_point + np.dot(R, cameras[i].camera_offset)

        else:  # DEPRECATED
            edge_perimeter = environment.perch_perimeter

            if opt_options.perch_on_ceiling:
                cameras[i].pose[0:3], cameras[i].wall_normal, on_ceiling = \
                    environment.map_particle_to_surface(particle=particle, start_index=i*ppc+particle_index)

                particle_index = particle_index + 3
            else:
                cameras[i].pose[0:2], cameras[i].wall_normal = \
                    map_line_to_edge(environment, particle[i*ppc + particle_index] * edge_perimeter)

                on_ceiling = False

            # vertical position is already covered by perch on ceiling if it is active. only include height
            if opt_options.variable_height and not opt_options.perch_on_ceiling:
                cameras[i].pose[2] = convert_particle_to_height(particle[i*ppc + particle_index], on_ceiling,
                                                                environment)
            else:
                cameras[i].pose[2] = 2  # constant height. typical ceiling is ~2.75m high; set in environment.max_height

        # Recall: the camera's axes are X=RIGHT, Y=DOWN, Z=FORWARD; (before was erroneously using X=fwd, Y=left, z=up)
        # therefore camera PAN is about Y, TILT is about X, ROLL is about Z
        if opt_options.variable_pan:
            cameras[i].pose[-2] = -convert_particle_to_angle(particle_angle=particle[i * ppc + particle_index],
                                                            wall_normal=cameras[i].wall_normal, camera=cameras[i],
                                                            axis=2, environment=environment, opt_options=opt_options)
            particle_index = particle_index + 1
        else:
            # passing 0.5 in as particle value and enabling target_center better respects angle mode (either point
            # normal to wall, or towards target) without adding search complexity
            cameras[i].pose[-2] = -convert_particle_to_angle(0.5, wall_normal=cameras[i].wall_normal,
                                                            camera=cameras[i], axis=2, environment=environment,
                                                            opt_options=opt_options, target_center=True)

        if opt_options.variable_tilt:
            cameras[i].pose[-3] = -convert_particle_to_angle(particle_angle=particle[i * ppc + particle_index],
                                                            wall_normal=cameras[i].wall_normal, camera=cameras[i],
                                                            axis=1, environment=environment, opt_options=opt_options)
            particle_index = particle_index + 1
        else:
            # passing 0.5 in as particle value and enabling target_center better respects angle mode (either point
            # normal to wall, or towards target) without adding search complexity
            cameras[i].pose[-3] = -convert_particle_to_angle(0.5, wall_normal=cameras[i].wall_normal,
                                                            camera=cameras[i], axis=1, environment=environment,
                                                            opt_options=opt_options, target_center=True)

        if opt_options.variable_zoom:
            cameras[i].fov = convert_particle_to_zoom(particle[i * ppc + particle_index], cameras[i])
            # particle_index = particle_index + 1  # re-add if more options are added in the future
        else:
            cameras[i].fov = cameras[i].base_fov

        # for the time being, never add any roll. Could consider adding roll as "noise"... might want noise resistance
        cameras[i].pose[-1] = 0

        cameras[i].surf_id = surf_id

        # if debug:
        #     print("Camera: " + str(i))
        #     print("Particle: " + str(particle[i*2:i*2+2]))
        #     print("FOV: " + str(cameras[i].fov[0]))
        #     print("Edge Normal Direction: " + str(cameras[i].wall_normal))
        #     print("Camera Pose: " + str(cameras[i].pose))

    return cameras


def convert_particle_to_angle(particle_angle, wall_normal, camera, environment, axis=2,
                              opt_options=CameraPlacementOptions(), target_center=False):
    """
    Converts a particle weight into a pan or tilt angle

    :param particle_angle: particle weight, on range [0, 1]
    :param wall_normal: normal direction of the surface on which the camera is placed
    :param camera: Camera instance
    :param environment: Environment Class instance containing the environment in which to place cameras
    :param axis: 0 (roll, not used) 1 (tilt/pitch axis), or 2 (pan/yaw axis)
    :param opt_options: used to specify the optimization options which impact how particles are mapped to angles.
     Relevant options include:
     angle_mode
        "TOWARDS_TARGET" - the camera is always pointed to within some angular variation (currently +/- 15 deg) of the
        target centroid
        "WALL_LIMITED" - the camera is always pointed within the internal 180 degrees of the surface which it is
        perching on. i.e. [-90, 90]
        "FOV_LIMITED" - the camera is always pointed within the range of angles which the FOV first contacts the surface
        which it is perching on.
        (i.e. [-90+FOV/2, 90-FOV/2])
        "GIMBAL_LIMITED" - the intersection of the FOV_LIMITED range and the camera's specified gimbal limits
        "GIMBAL_LIMITED_TARGETING" - this mode attempts to point TOWARDS_TARGET, but still conforms with gimbal limits
        All other opt result in a 0 to 360 value to be specified
     target_deviation - a 2d array or list containing the maximum allowable angular deviation away from the centroid of
     the target region
    :param target_center: bool. if true, disregard the particle and point either with 0 gimbal offset (for WALL, FOV, or
     GIMBAL limited options), directly towards the target center, for TOWARDS_TARGET, or as close to target center as
     possible for GIMBAL_LIMITED_TARGETING
    :return: converted angle to assign to camera
    """

    # extract euler angles from 0-direction of camera at surface
    eul = vec2eul(np.dot(camera.camera_rotation, wall_normal))
    norm_offset = eul[axis]  # uses standard convention; eul[0] = roll, eul[1] = pitch/tilt, eul[2] = pan/yaw
    angle_mode = opt_options.angle_mode
    target_deviation = opt_options.target_deviation

    if target_center:
        particle_angle = 0.5

    if angle_mode == "WALL_LIMITED":
        # recenter to -90 to 90 about edge normal
        angle = (particle_angle * 360 - 180) / 2 + norm_offset

    elif angle_mode == "FOV_LIMITED":
        # recenter about edge normal between -(90-fov/2) and (90-fov/2)
        angle = ((particle_angle * 360 - 180) / 2) * ((180 - camera.fov[1-axis]) / 180) + norm_offset
        # using 1-axis b/c FOV is listed as PAN (Horizontal) then TILT (vertical)
        # while you could consider adding different cases for corner points where the view is more or less than 180 deg,
        # it doesn't seem worth the complexity. A camera cannot perch on a corner and wouldn't benefit from increased
        # FOV. A drone also can't perch too near an inner corner, limiting the need for consideration over inner corner
        # occlusions.

    elif angle_mode == "GIMBAL_LIMITED":
        min_angle = np.min([90-camera.fov[1-axis]/2, camera.gimbal_limit[axis-1]])
        angle = (particle_angle * 2 * min_angle - min_angle) + norm_offset

    elif angle_mode == "TOWARDS_TARGET":
        # restrict to +/- X degrees about the center of the target region
        # find vector between camera location and target centroid
        vec_dir = np.array([environment.target_centroid[0] - camera.pose[0],
                            environment.target_centroid[1] - camera.pose[1],
                            environment.target_centroid[2] - camera.pose[2]])

        vec_dir = vec_dir / dist(vec_dir, np.zeros(3))  # normalize to get unit vector
        eul = vec2eul(vec_dir)  # convert to euler angles. use pitch for tilt, use yaw for pan

        angle = (particle_angle * target_deviation[axis-1] - target_deviation[axis-1]/2) + eul[axis]

    elif angle_mode == "GIMBAL_LIMITED_TARGETING":
        # point towards target center, if allowed by gimbal limits. else get as close as possible
        # find vector between camera location and target centroid
        vec_dir = np.array([environment.target_centroid[0] - camera.pose[0],
                            environment.target_centroid[1] - camera.pose[1],
                            environment.target_centroid[2] - camera.pose[2]])

        vec_dir = vec_dir / dist(vec_dir, np.zeros(3))  # normalize to get unit vector
        eul = vec2eul(vec_dir)  # convert to euler angles. use pitch for tilt, use yaw for pan

        # simplify notation a bit here... t=target direction, n=normal direction, g=gimbal limit, d=target deviation
        t = eul[axis]
        n = norm_offset
        g = camera.gimbal_limit[axis-1]
        d = target_deviation[axis-1]

        # min angular distance  between normal and target directions
        delta_angle = min_ang_dist(t, n)

        if np.abs(delta_angle) < g + d:
            # set of deviations about target direction intersects set of angles within gimbal limit, about the normal
            # direction. In this case, want the union of the two sets

            if target_center:  # disregard particle and try to get as close as possible to target direction
                if np.abs(delta_angle) < g:
                    return t  # target is within gimbal limits; aim at target
                else:  # return closest gimbal bound to target
                    if np.abs(min_ang_dist(t, n+g)) < np.abs(min_ang_dist(t, n-g)):
                        return n+g
                    else:
                        return n-g

            bounds = np.array([n + max(-g, delta_angle - d),
                               n + min(g, delta_angle + d)])  # upper and lower bound of resultant set

            # select point within set
            set_range = bounds[1]-bounds[0]
            angle = (particle_angle * set_range) + bounds[0]

        else:
            # target is not within gimbal limits. set angular range as the
            ubd = min_ang_dist(t, n+g)
            lbd = min_ang_dist(t, n-g)
            if np.abs(ubd) < np.abs(lbd):
                angle = n+g
            else:
                angle = n+g

    else:
        # angle_mode == "NO_LIMIT":
        # recenter to -180 to 180
        angle = (particle_angle * 360 - 180)

    return angle


def convert_particle_to_zoom(particle_zoom, camera):
    """
    Converts particle weight to zoom value on range [0, max_zoom]
    :param particle_zoom: particle weight
    :param camera: Camera class instance
    :return: zoom value
    """
    zoom = particle_zoom*camera.max_zoom
    return camera.base_fov/zoom


def convert_particle_to_height(particle_height, on_ceiling, environment):
    """
    converts particle weight on range [0, 1] into a camera height on [0, ceiling_height]
    :param particle_height: particle weight on range [0, 1]
    :param on_ceiling: boolean of whether or not the particle is on the ceiling
    :param environment: Environment Class instance containing the environment in which to place cameras
    :return: height value
    """
    if on_ceiling:
        height = environment.max_height
    else:
        height = particle_height * environment.max_height

    return height


def plot_particles_in_2d_search_space(particles, figure_num, weight_bounds=np.array([[0, 1], [0, 1]]), view_time=0.1,
                                      persistent_points=True):
    """
    Plots particles in 2D search space. for a 2N-D particle, particle is split into N points
    :param particles: np.array of particles
    :param figure_num: number of figure on which to plot points
    :param weight_bounds: limits of x and y axis
    :param view_time: how long the plot should stay open
    :param persistent_points: boolean, whether the plot should be cleared on each iteration to reduce clutter, or if
     particles should be maintained to vis coverage
    :return: none
    """
    dims = np.shape(particles)
    N = dims[0]  # number of particles
    D = dims[1]  # dimension of each particle
    n_cams = D//2

    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    RGBs = list(RGB_tuples)

    plt.figure(figure_num)

    for i in range(N):
        c = RGBs[i]
        x = np.zeros([n_cams])
        y = np.zeros([n_cams])

        for j in range(n_cams):
            x[j] = particles[i, j*2]
            y[j] = particles[i, j*2+1]
            plt.plot(x[j], y[j], 'o', c=c, markersize=3)

    plt.xlim(weight_bounds[0, :])
    plt.ylim(weight_bounds[1, :])

    plt.pause(view_time)
    if not persistent_points:
        plt.clf()


def plot_particles_on_2D_surface(particles, surf, figure_num, view_time=0.1, persistent_points=True):
    """
    Plots particles in 2D search space. for a 2N-D particle, particle is split into N points
    :param particles: np.array of particles
    :param surf: the 2D surface (projected) on which particles are searching
    :param figure_num: number of figure on which to plot points
    :param view_time: how long the plot should stay open
    :param persistent_points: boolean, whether the plot should be cleared on each iteration to reduce clutter, or if
     particles should be maintained to vis coverage
    :return: none
    """
    dims = np.shape(particles)
    N = dims[0]  # number of particles

    prj_lim = np.array([np.min(surf.projected_points, axis=0), np.max(surf.projected_points, axis=0)])
    prj_min = np.min(prj_lim, axis=0)
    prj_range = prj_lim[1, :] - prj_lim[0, :]

    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    RGBs = list(RGB_tuples)

    plt.figure(figure_num)
    for i in range(len(surf.perimeter_loop)):
        draw_boundary_2d(surf.projected_points[surf.perimeter_loop[i]], 'b')
    for i in range(len(surf.inner_loops)):
        draw_boundary_2d(surf.projected_points[surf.inner_loops[i]], 'r')

    for i in range(N):
        point = particles[i][:2] * prj_range + prj_min
        c = RGBs[i]
        plt.plot(point[0], point[1], 'o', c=c, markersize=3)

    plt.xlim(prj_lim[:, 0])
    plt.ylim(prj_lim[:, 1])

    plt.pause(view_time)
    if not persistent_points:
        plt.clf()


def plot_particles_on_3D_surface(particles, environment, surf_id, ppc, ax, view_time=0.1, persistent_points=True,
                                 plot_surface=True):
    """
    Plots particles in 3D search space. for a 2N-D particle, particle is split into N points
    :param particles: np.array of particles
    :param environment:
    :param surf_id: the ID number of the 3D mesh on which particles are searching. if ID is None, plot full environment.
     This is a slow process.
    :param ax: axes3D handle for the target plot
    :param view_time: how long the plot should stay open
    :param persistent_points: boolean, whether the plot should be cleared on each iteration to reduce clutter, or if
     particles should be maintained to vis coverage
    :param plot_surface: boolean, whether or not to plot the surface. Useful to set false to speed up plotting if only
     particles are changing
    :return: none
    """
    dims = np.shape(particles)
    N = dims[0]  # number of particles

    surf = environment.perch_regions[surf_id]

    prj_lim = np.array([np.min(surf.points, axis=0), np.max(surf.points, axis=0)])

    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    RGBs = list(RGB_tuples)

    if plot_surface:
        if surf_id is None:
            environment.plot_environment(ax=ax, show=False)
        elif 0 <= surf_id < len(environment.perch_regions):
            environment.perch_regions[surf_id].plot_mesh(ax)

    for i in range(N):
        point, _, _ = environment.map_particle_to_surface(particles[i][:ppc],
                                                          surface_index=surf_id)
        point = np.squeeze(point)
        c = RGBs[i]
        ax.scatter3D(point[0], point[1], point[2], color=c)

    ax.set_xlim(prj_lim[:, 0])
    ax.set_ylim(prj_lim[:, 1])
    ax.set_zlim(prj_lim[:, 2])

    plt.pause(view_time)
    if not persistent_points:
        plt.clf()


def plot_particle_2d(particle, environment, cameras, placed_cameras=PlacedCameras(), view_time=0.1, figure_number=0,
                     boundary_offset=5):
    """
    Plots the camera arrangement specified by a particle, in a 2D environment

    :param particle: np.array of the particle weights
    :param environment: Environment class instance
    :param cameras: a list of Camera objects corresponding to the cameras which have already been placed in an
        environment
    :param placed_cameras: PlacedCamera class including cameras already placed in the environment
    :param view_time: how long the plot should be visible
    :param figure_number: matplotlib figure number
    :param boundary_offset: x-y offset around walls to use when plotting
    :return: none
    """
    cameras = convert_particle_to_state(environment, particle, cameras)

    plt.figure(figure_number)

    # Draw polygons for wall, target, and obstacles
    draw_boundary_2d(environment.walls)
    draw_boundary_2d(environment.target, "g")
    for o in environment.obstacles:
        draw_boundary_2d(o, "r")

    for p in environment.perch_regions:
        draw_boundary_2d(p, "y")

    # Place cameras for particle (red) and any already placed cameras (magenta)
    place_cameras_2d(cameras, environment)
    if len(placed_cameras.cameras) > 0:
        place_cameras_2d(placed_cameras.cameras, environment, "m")

    # plot randomly sampled points within target
    # plt.plot(environment.sampling_points[:, 0], environment.sampling_points[:, 1], 'ro', markersize=1)

    # keep plot window constant to make visualization less head-ache inducing
    plt.xlim([np.min(environment.walls[:, 0]) - boundary_offset, np.max(environment.walls[:, 0]) + boundary_offset])
    plt.ylim([np.min(environment.walls[:, 1]) - boundary_offset, np.max(environment.walls[:, 1]) + boundary_offset])

    if view_time > 0:
        plt.pause(view_time)
    else:
        plt.waitforbuttonpress()
    plt.clf()


def plot_particle_3d(particle, environment, cameras=None, placed_cameras=None, view_time=0.0, figure_number=0,
                     opt_options=CameraPlacementOptions(), reevaluate_state=False):
    """
    Plots cameras and environment in 3D
    :param particle: particle to be plotted (np.array)
    :param environment: Environment class instance
    :param cameras: a list of Camera objects corresponding to the cameras which have already been placed in an
        environment
    :param placed_cameras: PlacedCameras class instance containing already placed cameras
    :param view_time: how long the plot should be visible
    :param figure_number: figure number on which to draw environment
    :param opt_options: optimization options
    :param reevaluate_state: whether or not to re-evaluate the camera state using the particle input (defaults to false)
    :return: none
    """
    figure = plt.figure(figure_number)
    plt.clf()
    ax = Axes3D(figure)

    # cameras should already have correct state following particle evaluation. only re-evaluate if needed
    if reevaluate_state:
        cameras = convert_particle_to_state(environment, particle, cameras, opt_options)

    draw_room_3d(environment=environment, cameras=cameras, placed_cameras=placed_cameras, fill_alpha=0.1, ax=ax)

    ax.elev = 15
    ax.azim = -140

    # ax.elev = 54
    # ax.azim = -70

    if view_time > 0:
        plt.pause(view_time)  # block for view_time seconds
    elif view_time < 0:
        plt.show()  # block until closed
    else:
        plt.waitforbuttonpress()  # block until any button is pressed
