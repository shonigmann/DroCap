import numpy as np
import vedo
import trimesh
import pyvista as pv
import pymesh

approved = -1
selection = None
p = pv.Plotter(notebook=False)


def confirm_surfaces(environment, N_max=10):
    global p
    global approved
    global selection

    environment.perch_regions = sorted(environment.perch_regions, key=lambda surf: surf.net_area, reverse=True)
    if environment.full_env is not None:
        env_mesh = environment.full_env
        env_path = environment.full_env_path
    else:
        env_mesh = environment.cluster_env
        env_path = environment.clust_env_path

    n_valid = 0
    i = 0

    while n_valid < N_max and i < len(environment.perch_regions):
        p = pv.Plotter(notebook=False)
        p.set_background(color='w')
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

    # pts = perch_region.principle_axis_projected_points
    # eig_vecs = perch_region.eigen_vectors.copy()  # TODO: might need to flip eig here due to fix elsewhere. revisit
    # if perch_region.classification == "wall":
    #     if np.abs(eig_vecs[0].dot(g)) > np.abs(eig_vecs[1].dot(g)):
    #         tmp = eig_vecs[0].copy()
    #         eig_vecs[0] = eig_vecs[1].copy()
    #         eig_vecs[1] = tmp
    #     if eig_vecs[1].dot(g) < 0:
    #         eig_vecs[1] *= -1
    #     R_ = eig_vecs.astype(np.float64)
    # else:
    #     R_ = eig_vecs.astype(np.float64)

    # obb = np.array([np.min(pts, axis=0), np.max(pts, axis=0)])

    # load mesh in open3d
    mesh = trimesh.load(full_mesh_path)
    R_aug = np.zeros([4, 4])
    R_aug[:3, :3] = R
    mesh.vertices = trimesh.transform_points(mesh.vertices, R_aug)

    # Crop the mesh using face color
    f_colors = np.asarray(mesh.visual.face_colors)
    f_colors = f_colors / 255
    # add opacity
    f_colors[:, 3] = 1.0

    pvm = pv.PolyData(full_mesh_path)
    n = perch_region.mesh_normal
    pm = pymesh.form_mesh(perch_region.points + n*0.1, perch_region.faces)
    pms, _ = pymesh.split_long_edges(pm, 0.1)

    surf_mesh = pv.PolyData(pms.vertices, np.hstack([np.ones([len(pms.faces), 1])*3, pms.faces]).astype(int))

    p.add_mesh(surf_mesh, color='g', opacity=1.0, name="surf_mesh", culling=False)

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

    p.show(auto_close=True, interactive=True)

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
