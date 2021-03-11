import pickle
import vedo
import numpy as np

from environment_data import ply_environment
from sim.initialize import initialize, initialize_camera_list, initialize_opt_options, initialize_pso_options
from pso.pso_functions import evaluate_swarm, convert_particle_to_state
from sim.cameras import PlacedCameras
from ui.perch_confirmation import confirm_perch_placement
from ui.select_target import select_target
from ui.surface_confirmation import confirm_surfaces
from sim.cameras import Camera
import pyvista as pv
import trimesh
import time


p = pv.Plotter(notebook=False)


def print_cam_pos():
    print("-------------------------")
    print(p.camera_position)


def rotate_camera_for_targeting(p_, fp, c):
    p_.open_movie(fp)
    R = 4.5
    th1 = -118 * np.pi / 180.0
    th2 = -170 * np.pi / 180.0
    nframe = 50

    th = np.linspace(th1, th2, nframe)
    for t in th[:nframe // 3]:
        p_.camera_position = [
            (R * np.cos(t), R * np.sin(t), 2.8),
            (c[0], c[1], c[2]),  # (-0.026929191045848594, 0.5783514020506139, 0.8268966663940324),
            (0, 0, 1),
        ]
        time.sleep(0.001)
        p_.write_frame()

    x_ = p_.camera_position
    x = np.array([x_[0][0], x_[0][1], x_[0][2]])
    r = np.linalg.norm(x)
    th1 = np.arctan2(x[2], np.linalg.norm(x[:2]))
    th2 = np.pi / 2 * 1.03
    th = np.linspace(th1, th2, nframe)

    for t in th:
        xy = x[:2]/np.linalg.norm(x[:2]) * r * np.cos(t)
        p_.camera_position = [
            (xy[0], xy[1], r * np.sin(t)),
            (c[0], c[1], c[2]),  # (-0.026929191045848594, 0.5783514020506139, 0.8268966663940324),
            (0, 0, 1),
        ]
        time.sleep(0.001)
        p_.write_frame()


def rotate_camera_for_targeting_inverse(p_, fp, c):
    p_.open_movie(fp)
    R = 4.5
    th1 = -118 * np.pi / 180.0
    th2 = -170 * np.pi / 180.0
    nframe = 50
    th = np.linspace(th2, th1, nframe)

    x_ = np.array([R*np.cos(th[nframe // 3 - 1]), R*np.sin(th[nframe // 3 - 1]), 2.8])
    x = np.array([x_[0], x_[1], x_[2]])
    r = np.linalg.norm(x)
    th1 = np.arctan2(x[2], np.linalg.norm(x[:2]))
    th2 = np.pi / 2 * 1.03
    th_step = (th2-th1)/nframe
    th = np.linspace(th2, th1 - th_step, nframe)

    for t in th:
        xy = x[:2]/np.linalg.norm(x[:2]) * r * np.cos(t)
        p_.camera_position = [
            (xy[0], xy[1], r * np.sin(t)),
            (c[0], c[1], c[2]),  # (-0.026929191045848594, 0.5783514020506139, 0.8268966663940324),
            (0, 0, 1),
        ]
        time.sleep(0.001)
        p_.write_frame()

    x_ = p_.camera_position
    x = np.array([x_[0][0], x_[0][1], x_[0][2]])
    r = np.linalg.norm(x[:2])
    th1 = np.arctan2(x[1], x[0])
    th2 = -170 * np.pi / 180.0
    th = np.linspace(th1, th2, nframe*2//3)
    for t in th:
        p_.camera_position = [
            (r * np.cos(t), r * np.sin(t), x[2]),
            (c[0], c[1], c[2]),  # (-0.026929191045848594, 0.5783514020506139, 0.8268966663940324),
            (0, 0, 1),
        ]
        time.sleep(0.001)
        p_.write_frame()


def rotate_camera(p_, fp, c, R=7.89, z=2.8, nframe=75):
    p_.open_movie(fp)
    th1 = -118*np.pi/180.0
    th2 = -170*np.pi/180.0

    th = np.linspace(th1, th2, nframe*2)
    for t in th:
        p_.camera_position = [
            (R*np.cos(t), R*np.sin(t), z),
            (c[0], c[1], c[2]),  # (-0.026929191045848594, 0.5783514020506139, 0.8268966663940324),
            (0, 0, 1),
        ]
        time.sleep(0.001)
        p_.write_frame()

    # th = np.linspace(th2, th1, nframe)
    # for t in th:
    #     p_.camera_position = [
    #         (R*np.cos(t), R*np.sin(t), z),
    #         (c[0], c[1], c[2]),  # (-0.026929191045848594, 0.5783514020506139, 0.8268966663940324),
    #         (0, 0, 1),
    #     ]
    #     time.sleep(0.001)
    #     p_.write_frame()


def vis_mesh(plotter, mesh1, c1, mesh2, c2, fp,
             c=np.array([-0.026929191045848594, 0.5783514020506139, 0.8268966663940324]),
             R=14.0, z=3.0, n_frame=75):
    plotter.subplot(0, 0)
    p.add_mesh(mesh1, scalars=c1, rgb=True, name="env_mesh", culling=False)
    p.subplot(0, 1)
    p.add_mesh(mesh2, scalars=c2, rgb=True, name="env_mesh",
               culling=False)
    p.link_views()
    p.show(auto_close=False, interactive=False, full_screen=True)
    rotate_camera(p, fp=fp, c=c, z=z, R=R, nframe=n_frame)


def main():
    global p
    clust_env_folder = '/home/simon/catkin_ws/src/mesh_partition/models/'
    full_env_folder = '/home/simon/catkin_ws/src/mesh_partition/datasets/'

    show_full = False
    show_reduced = False
    show_clustered = False
    show_target = False
    show_filtering = True
    show_clustered_filtered = False

    file_base = ["apartment"]

    for i in range(len(file_base)):
        fb = file_base[i]
        seg_env_prototype = clust_env_folder + fb + "_cluster*.ply"
        target_prototype = full_env_folder + fb + "_target.ply"
        cluster_env_path = clust_env_folder + fb + "_c2000.ply"
        full_env_path = full_env_folder + fb + "_filtered.ply"

        pickle_env = open('../test/preloaded_environment_' + fb + '.p', 'rb')
        preloaded_vars = pickle.load(pickle_env)
        pickle_env.close()

        preloaded_vars['env'].name = file_base[i]

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
        optimal_cameras = PlacedCameras()
        optimal_cameras.cameras.append(Camera())
        best_pos = np.random.random(optimization_options.get_particle_size())
        best_cam = convert_particle_to_state(environment=env, particle=best_pos, cameras=optimal_cameras.cameras,
                                             opt_options=optimization_options)[0]
        optimal_cameras.cameras.append(best_cam)

        # START VISUALIZATION
        p = pv.Plotter(notebook=False)

        # 1) Visualize FULL ENV
        mesh = trimesh.load('/home/simon/catkin_ws/src/mesh_partition/datasets/apartment_1m.ply')
        f_colors = np.asarray(mesh.visual.face_colors)
        f_colors = f_colors / 255
        f_colors[:, 3] = 1.0
        pvm = pv.PolyData('/home/simon/catkin_ws/src/mesh_partition/datasets/apartment_1m.ply')
        p.add_mesh(pvm, scalars=f_colors, rgb=True, name="env_mesh", culling=False)  # f_colors[:,:3])

        path = '/home/simon/catkin_ws/src/mesh_partition/datasets/apartment_filtered.ply'
        mesh = trimesh.load(path)
        f_colors_filtered = np.asarray(mesh.visual.face_colors)
        f_colors_filtered = f_colors_filtered / 255
        f_colors_filtered[:, 3] = 1.0
        pvm_filtered = pv.PolyData(path)

        path = '/home/simon/catkin_ws/src/mesh_partition/models/apartment_c2000.ply'
        mesh = trimesh.load(path)
        f_colors_clustered = np.asarray(mesh.visual.face_colors)
        f_colors_clustered = f_colors_clustered / 255
        f_colors_clustered[:, 3] = 1.0
        pvm_clustered = pv.PolyData(path)

        path = '/home/simon/catkin_ws/src/perch_placement/src/results/apartment__pso_allsurf_meshmap_facecenter/apartmentpso_target_mesh.ply'
        pvm_target = pv.PolyData(path)

        path = '/home/simon/catkin_ws/src/mesh_partition/models/apartment_clustered_filtered.ply'
        mesh = trimesh.load(path)
        f_colors_clustered_filtered = np.asarray(mesh.visual.face_colors)
        f_colors_clustered_filtered = f_colors_clustered_filtered / 255
        f_colors_clustered_filtered[:, 3] = 1.0
        pvm_clustered_filtered = pv.PolyData(path)
        p.add_key_event('p', print_cam_pos)

        if show_full:
            p.show(auto_close=False, interactive=False, full_screen=True)
            c = np.array([-0.026929191045848594, 0.5783514020506139, 0.8268966663940324])
            rotate_camera(p, fp='../ui/full_env.mp4', c=c)
        p.close()

        # 2) Target selection  TODO: automate pan/tilt, then select target, then show target volume!

        if show_target:
            p = pv.Plotter(notebook=False)
            p.add_mesh(pvm, scalars=f_colors, rgb=True, name="env_mesh", culling=False)  # f_colors[:,:3])
            p.enable_cell_picking(style='surface', color='r', through=False,
                                  show_message="")  #
            p.add_text(text="Press R to toggle selection tool\n"
                            "Press Y to approve region\n"
                            "Press Q to confirm selection\n",
                       color='w', font_size=18)

            p.show(auto_close=False, interactive=False, full_screen=True)
            c = np.array([-0.026929191045848594, 0.5783514020506139, 0.8268966663940324])
            rotate_camera_for_targeting(p, fp='../ui/target_env1.mp4', c=c)
            pos = p.camera_position
            p.add_mesh(pvm, scalars=f_colors, rgb=True, name="env_mesh", culling=False)  # f_colors[:,:3])
            p.enable_cell_picking(style='surface', color='r', through=False,
                                  show_message="")  #
            p.add_text(text="Press R to toggle selection tool\n"
                            "Press Y to approve region\n"
                            "Press Q to confirm selection\n",
                       color='w', font_size=18)
            p.show(auto_close=False, interactive=True, full_screen=True)
            p.add_mesh(pvm_target, color='g', opacity=0.5)
            rotate_camera_for_targeting_inverse(p, fp='../ui/target_env2.mp4', c=c)
            p.close()

        # 1.2) Visualize FULL (LEFT) and FILTERED (RIGHT)
        if show_reduced:
            p = pv.Plotter(notebook=False, shape=(1, 2))
            vis_mesh(p, pvm, f_colors, pvm_filtered, f_colors_filtered,
                     c=np.array([-0.026929191045848594, 0.5783514020506139, 0.8268966663940324]), R=14,
                     fp='../ui/reduced_env.mp4')
            p.close()

        # 3) Visualize Full and Clustered
        if show_clustered:
            p = pv.Plotter(notebook=False, shape=(1, 2))
            vis_mesh(p, pvm, f_colors, pvm_clustered, f_colors_clustered,
                     c=np.array([-0.026929191045848594, 0.5783514020506139, 0.8268966663940324]), R=14,
                     fp='../ui/clustered_new_env.mp4')
            p.close()

        # 3.2) Visualize Full and Clustered with Area filter
        if show_clustered_filtered:
            p = pv.Plotter(notebook=False, shape=(1, 2))
            vis_mesh(p, pvm_clustered, f_colors_clustered, pvm_clustered_filtered, f_colors_clustered_filtered,
                     c=np.array([-0.026929191045848594, 0.5783514020506139, 0.8268966663940324]), R=14,
                     fp='../ui/filter_env.mp4')
            p.close()

        # TODO: consider redoing manual filtering!
        if show_filtering:
            steps = ['clusters', 'area', 'direction', 'h', 'prox', 'sight', 'window']
            steps_to_vis = [True, True, True, True, True, True, True]
            m1 = pvm_filtered
            mc1 = f_colors_filtered

            optimization_options.land_on_floor = True
            optimization_options.perch_on_walls = True
            optimization_options.perch_on_ceiling = True
            optimization_options.perch_on_intermediate_angles = True
            env_ = ply_environment(file_path_prototype=seg_env_prototype,
                                   target_path_prototype=None,
                                   cluster_env_path=cluster_env_path,
                                   optimization_options=optimization_options,
                                   reorient_mesh=False, N_points=100,
                                   full_env_path=None)
            idx=0
            for f in steps:
                if f == steps[1]:  # minimum area filter (not sure if needed)
                    env_.apply_area_filter(min_area=1.0)
                    n_faces = 0
                elif f == steps[2]:
                    env_.opt_options.land_on_floor = False
                    env_.opt_options.perch_on_walls = True
                    env_.opt_options.perch_on_ceiling = True
                    env_.opt_options.perch_on_intermediate_angles = False
                    env_.estimate_centroid()
                    env_.correct_normals()
                    env_.set_surface_as_perchable()

                elif f == steps[3]:
                    env_.post_process_environment(apply_height_filter=True, apply_proximity_filter=False,
                                                  apply_line_of_sight_filter=False, apply_perch_window_filter=False)
                    # break
                elif f == steps[4]:
                    env_.post_process_environment(apply_height_filter=False, apply_proximity_filter=True,
                                                  apply_line_of_sight_filter=False, apply_perch_window_filter=False)
                    # break
                elif f == steps[5]:
                    env_.post_process_environment(apply_height_filter=False, apply_proximity_filter=False,
                                                  apply_line_of_sight_filter=True, apply_perch_window_filter=False)
                    # break
                elif f == steps[6]:
                    env_.post_process_environment(apply_height_filter=False, apply_proximity_filter=False,
                                                  apply_line_of_sight_filter=False, apply_perch_window_filter=True)
                    # break

                perch_surfs = env_.perch_regions
                mc2 = np.zeros([0, perch_surfs[0].face_colors.shape[1]])
                fcs = np.zeros([0, 3])
                pts = np.zeros([0, 3])
                for ps in perch_surfs:
                    if ps.is_valid:
                        fcs = np.vstack([fcs, ps.faces + len(pts)])
                        pts = np.vstack([pts, ps.points])
                        mc2 = np.vstack([mc2, ps.face_colors[0]*np.ones_like(ps.faces)])  # TODO: make sure face colors is updated properly in filters

                print(f + str(len(fcs)))

                m2 = pv.PolyData(pts, np.hstack([np.ones([len(fcs), 1]) * 3, fcs]).astype(int))
                # mc2 = np.array([mc2[:, 2], mc2[:, 1], mc2[:, 0]]).T
                if (mc2 > 1.0).any():
                    mc2 /= 255.0

                if steps_to_vis[idx]:
                    print('Showing: ' + f)

                    p = pv.Plotter(notebook=False, shape=(1, 2))
                    vis_mesh(p, m1, mc1, m2, mc2, R=14, z=3.5, fp='../ui/'+f+'_filter.mp4', n_frame=100)
                    p.close()

                mc1 = mc2.copy()
                m1 = m2.copy()

                idx += 1

        # 5) Optimization + User Confirmation (show pass cases only?)
        # TODO:

        # if confirm_perch_placement(environment=env, placed_cameras=optimal_cameras.cameras, focus_id=i):
        #     # testing...
        #     print("Selected Camera Locations: ")
        #     for camera in optimal_cameras.cameras:
        #         p = camera.pose
        #         p[3:] *= np.pi / 180.0
        #         p[5] = -p[4]
        #         p[4] = -p[3]
        #         p[3] = 0
        #         print(camera.pose)
        #
        # else:
        #     optimal_cameras.cameras.pop()
        #     env.remove_rejected_from_perch_space(camera=best_cam, r=0.3)


if __name__ == "__main__":
    main()
