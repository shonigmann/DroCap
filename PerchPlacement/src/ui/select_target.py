import numpy as np
import vedo
import trimesh
import pyvista as pv
import pymesh

# plt1 = vedo.Plotter(title='Confirm Perch Location')
approved = -1
selection = None
p = None


def select_target(environment):
    global p
    global approved
    global selection

    environment.perch_regions = sorted(environment.perch_regions, key=lambda surf: surf.net_area, reverse=True)
    if environment.full_env is not None:
        env_path = environment.full_env_path
    else:
        env_path = environment.clust_env_path

    selected = False
    while approved != 1:
        p = pv.Plotter(notebook=False, title="Select Target")
        p.set_background(color='w')
        user_selects_target(full_mesh_path=env_path, R=environment.R)

    return


def user_selects_target(full_mesh_path, R):
    #  Find Oriented Bounding Box for surface
    global approved

    # load mesh in open3d
    mesh = trimesh.load('/home/simon/catkin_ws/src/mesh_partition/datasets/apartment_1m.ply')
    R_aug = np.zeros([4, 4])
    R_aug[:3, :3] = R
    mesh.vertices = trimesh.transform_points(mesh.vertices, R_aug)

    # Crop the mesh using face color
    f_colors = np.asarray(mesh.visual.face_colors)
    f_colors = f_colors / 255
    # add opacity
    f_colors[:, 3] = 1.0

    pvm = pv.PolyData('/home/simon/catkin_ws/src/mesh_partition/datasets/apartment_1m.ply')

    p.add_mesh(pvm, scalars=f_colors, rgb=True, name="env_mesh", culling=False)  # f_colors[:,:3])

    p.enable_cell_picking(mesh=pvm, style='surface', color='r', through=False,
                          show_message="")
    p.add_text(text="Press R to toggle selection tool\n"
                    "Press Y to approve region\n"
                    "Press Q to confirm selection\n",
               color='k', font_size=18)

    p.add_key_event(key='y', callback=pyvista_approve_region)
    p.show(auto_close=True, interactive=True)
    p.deep_clean()
    p.clear()
    p.close()

    return approved


def pyvista_approve_region():
    global approved
    print("approving region")
    approved = 1
    return -1
