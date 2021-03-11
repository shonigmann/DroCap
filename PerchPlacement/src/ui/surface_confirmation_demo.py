import numpy as np
import vedo
import trimesh
import pyvista as pv
import pymesh

# plt1 = vedo.Plotter(title='Confirm Perch Location')
approved = -1
selection = None
p = pv.Plotter(notebook=False)


def print_cam_pos():
    print("-------------------------")
    print(p.camera_position)


def confirm_surfaces(environment, N_max=10):
    global p
    global approved
    global selection

    # plt1.keyPressFunction = surf_keyfunc

    environment.perch_regions = sorted(environment.perch_regions, key=lambda surf: surf.net_area, reverse=True)
    if environment.full_env is not None:
        env_mesh = environment.full_env
        env_path = environment.full_env_path
    else:
        env_mesh = environment.cluster_env
        env_path = environment.clust_env_path

    n_valid = 0
    i = 0

    cam_poses = [
        np.array([[-1.8527394214820874, -8.898351205073983, 2.411918302850694],
         [0.4276506600151187, 1.6632412940849672, 1.372849404608004],
         [0.08345153162687174, 0.07970624666120182, 0.9933190605803921]]),

        np.array([[-3.541336113724442, -7.247799684339455, 0.09890939369780272],
         [0.030319130130572747, 0.8331981745321573, 1.653976252641371],
         [-0.03083634890050513, -0.17571872549645912, 0.9839573410958353]]),

        np.array([[-3.017920987098659, -1.8902417268975296, 2.4341429917739705],
         [0.3864754287145259, 0.40308117747999384, 1.6186678942920476],
         [0.21615180259390215, 0.026182239063852017, 0.9760086519047921]]),

        np.array([[-3.341810904573494, 0.9764946335448804, 2.1066176917486974],
         [0.6223568200020658, -0.17462931351945798, 1.4177545382231624],
         [0.1877395233219364, 0.05983113761919531, 0.9803948726681739]]),

        np.array([[-2.3596467361887523, -4.156909784152686, 2.818280602701666],
         [-1.6428565187548283, 0.7069888719885022, 1.6053061306976804],
         [0.013502088312044948, 0.2400764909047159, 0.9706600703264207]]),

        np.array([[-3.53181091439956, -3.46302900218392, 3.5022299878595198],
         [-0.5018580048071983, 1.2910096461027796, 1.1018465818797472],
         [0.2681214387855507, 0.29256865352041783, 0.9178858736470542]])
    ]

    while n_valid < N_max and i < len(cam_poses):
        p = pv.Plotter(notebook=False)
        p.add_key_event('p', print_cam_pos)
        p.set_background(color='w')
        p.camera_position = cam_poses[i]

        approved = -1

        status, selected = user_approves_surface(environment.perch_regions[i], env_mesh,
                                     full_mesh_path=env_path, R=environment.R, env_path=env_path)
        if status == -1:
            environment.perch_regions[i].is_valid = False
            i += 1
        elif status == 2:
            environment.perch_regions[i].apply_mesh_mask(selection.points, action="remove")
            if not environment.perch_regions[i].is_valid:
                i += 1
        elif status == 3:
            environment.perch_regions[i].apply_mesh_mask(selection.points, action="keep")
            if not environment.perch_regions[i].is_valid:
                i += 1
        else:
            n_valid += 1
            i += 1

    return n_valid


def user_approves_surface(perch_region, full_mesh, full_mesh_path, R, g=np.array([0, 0, -1]), env_path=None):
    #  Find Oriented Bounding Box for surface
    global approved

    # load mesh in open3d
    mesh = trimesh.load('/home/simon/catkin_ws/src/mesh_partition/datasets/apartment_1m.ply')
    R_aug = np.zeros([4, 4])
    R_aug[:3, :3] = R
    mesh.vertices = trimesh.transform_points(mesh.vertices, R_aug)

    # Crop the mesh using face color
    f_colors = np.asarray(mesh.visual.face_colors)
    f_colors = f_colors / 255.0
    # add opacity
    f_colors[:, 3] = 1.0

    # vedo_mesh = vedo.mesh.Mesh(mesh)
    # vedo_mesh.cellIndividualColors(f_colors, alphaPerCell=True)
    # vedo_mesh.frontFaceCulling()
    # plt1.clear()
    # plt1.add(vedo_mesh)
    # plt1.add(vedo.Text2D("Press 'y' to approve surface. Press 'n' to reject. "
    #                      "\nPress 'f' for front culling, 'b' for back culling, 'c' to disable culling "
    #                      "\nPress 'q' when done",
    #                      pos='bottom-right', c='dg', bg='g', font='Godsway'))

    pvm = pv.PolyData('/home/simon/catkin_ws/src/mesh_partition/datasets/apartment_1m.ply')
    n = perch_region.mesh_normal
    pm = pymesh.form_mesh(perch_region.points + n*0.1, perch_region.faces)
    pms, _ = pymesh.split_long_edges(pm, 0.1)
    pm2 = pymesh.form_mesh(perch_region.points - n*0.1, perch_region.faces)
    pms2, _ = pymesh.split_long_edges(pm2, 0.1)
    #
    # surf_colors = np.zeros([len(pms.faces), 4])
    # surf_colors[:, 1] = 0.5  # g
    # surf_colors[:, 3] = 1.0  # alpha
    surf_mesh = pv.PolyData(pms.vertices, np.hstack([np.ones([len(pms.faces), 1])*3, pms.faces]).astype(int))
    surf_mesh2 = pv.PolyData(pms2.vertices, np.hstack([np.ones([len(pms2.faces), 1])*3, pms2.faces]).astype(int))

    # p.add_mesh(surf_mesh, scalars=surf_colors, rgb=True, name="surf_mesh")
    p.add_mesh(surf_mesh, color='g', opacity=.50, name="surf_mesh", culling=False)
    p.add_mesh(surf_mesh2, color='g', opacity=.50, name="surf_mesh2", culling=False)

    p.add_mesh(pvm, scalars=f_colors, rgb=True, name="env_mesh", culling=False)  # f_colors[:,:3])

    p.enable_cell_picking(mesh=surf_mesh, style='wireframe', color='r', through=True,
                          show_message="")
    p.add_text(text="Press R to toggle selection tool\n"
                                       "Press D to remove selected region\n"
                                       "Press A to keep only the selected region\n"
                                       "Press Y to approve region\n"
                                       "Press N to reject region\n"
                                       "Press Q to confirm selection\n",
               color='k', font_size=18)

    p.add_key_event(key='d', callback=pyvista_remove_region)
    p.add_key_event(key='a', callback=pyvista_select_region)
    p.add_key_event(key='y', callback=pyvista_approve_region)
    p.add_key_event(key='n', callback=pyvista_reject_region)

    # pvm.plot(scalars=f_colors, rgb=True)

    p.show(auto_close=True, interactive=True, full_screen=True)

    # plt1.show()
    p.deep_clean()
    p.clear()
    p.close()

    return approved, selection


def pyvista_remove_region():
    global approved
    global selection
    selection = p.picked_cells
    print("removing selected region")
    print(selection)

    approved = 2
    return -1


def pyvista_select_region():
    global approved
    global selection

    selection = p.picked_cells
    print("adding selected region")
    print(selection)
    approved = 3
    return -1


def pyvista_approve_region():
    global approved
    print("approving region")
    approved = 1
    return -1


def pyvista_reject_region():
    global approved
    print("rejecting region")
    approved = -1
    return -1
