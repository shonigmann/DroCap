import numpy as np

from geom.geometry3d import rot3d_from_rtp, rot3d_from_z_vec, rot3d_from_x_vec
import vedo
import trimesh
import pyvista as pv
import cv2 as cv


plt1 = vedo.Plotter(title='Confirm Perch Location', pos=[0, 0], interactive=True, sharecam=False)
plt2 = vedo.Plotter(title='Simulated Camera View', sharecam=False, pos=[800, 0], size=[1155.5, 650], interactive=False)

approved = False
user_responded = False


def confirm_perch_placement(environment, placed_cameras, focus_id):
    global approved
    global user_responded

    plt1.keyPressFunction = perch_keyfunc
    plt1.clear()
    approved = False

    plt2.clear()
    dense_env = vedo.load('/home/simon/catkin_ws/src/mesh_partition/datasets/' + environment.name + '_1m.ply')
    pv_dense_env = pv.read('/home/simon/catkin_ws/src/mesh_partition/datasets/' + environment.name + '_1m.ply')

    plt2.add(dense_env)
    plt2.show()

    plt3 = pv.Plotter(notebook=False)
    plt3.add_mesh(pv_dense_env)
    # plt3.show(interactive=False)

    # draw wireframe lineset of camera frustum
    env_mesh = trimesh.load('/home/simon/catkin_ws/src/mesh_partition/datasets/' + environment.name + '_1m.ply')
    # env_mesh = trimesh.load(environment.full_env_path)
    R = np.zeros([4, 4])
    R[:3, :3] = environment.R
    env_mesh.vertices = trimesh.transform_points(env_mesh.vertices, R)
    env_mesh_vedo = vedo.mesh.Mesh(env_mesh)

    target_mesh_pymesh = environment.generate_target_mesh(shape='box')
    target_mesh = trimesh.Trimesh(target_mesh_pymesh.vertices, target_mesh_pymesh.faces)
    target_mesh_vedo = vedo.mesh.Mesh(target_mesh)
    target_colors = 0.5*np.ones([len(target_mesh.faces), 4])
    target_colors[:, 0] *= 0.0
    target_colors[:, 2] *= 0.0
    target_mesh_vedo.alpha(0.6)
    target_mesh_vedo.cellIndividualColors(target_colors, alphaPerCell=True)
    plt2.add(target_mesh_vedo)

    env_mesh.visual.face_colors[:, -1] = 255.0
    env_mesh_vedo.cellIndividualColors(env_mesh.visual.face_colors/255.0, alphaPerCell=True)
    geom_list = [env_mesh_vedo, target_mesh_vedo]

    for s in environment.perch_regions:
        surf_mesh = trimesh.Trimesh(vertices=s.points, faces=s.faces)
        vedo_surf_mesh = vedo.mesh.Mesh(surf_mesh)
        vedo_surf_mesh.color('g')
        vedo_surf_mesh.opacity(0.5)
        geom_list.append(vedo_surf_mesh)

    for i in range(len(placed_cameras)):
        quad_mesh = trimesh.load("/home/simon/catkin_ws/src/perch_placement/src/ui/models/white-red-black_quad2.ply")

        # offset mesh coords to match camera pose
        # eul = placed_cameras[i].pose[3:]  # USE WALL NORMAL, NOT CAMERA POSE
        # R = rot3d_from_rtp(np.array([eul[2], -eul[0], -eul[1]]))
        R = rot3d_from_x_vec(placed_cameras[i].wall_normal)
        R2 = rot3d_from_rtp(np.array([0, -90, 0]))
        R_aug = np.zeros([4, 4])
        R_aug[:3, :3] = R.dot(R2)
        R_aug[:3, -1] = placed_cameras[i].pose[:3]
        quad_mesh.vertices = trimesh.transform_points(quad_mesh.vertices, R_aug)
        quad_mesh_vedo = vedo.mesh.Mesh(quad_mesh)
        quad_mesh_vedo.cellIndividualColors(quad_mesh.visual.face_colors/255, alphaPerCell=True)
        quad_mesh_pv = pv.read(
            "/home/simon/catkin_ws/src/perch_placement/src/ui/models/white-red-black_quad2.ply")

        pymesh_frustum = placed_cameras[i].generate_discrete_camera_mesh(degrees_per_step=10, environment=environment)
        pymesh_verts = pymesh_frustum.vertices.copy()
        pymesh_verts.flags.writeable = True
        pymesh_faces = pymesh_frustum.faces.copy()
        pymesh_faces.flags.writeable = True

        if i == focus_id:
            frustum = trimesh.Trimesh(vertices=pymesh_frustum.vertices.copy(), faces=pymesh_frustum.faces.copy())
            vedo_frustum = vedo.mesh.Mesh(frustum)
            vedo_frustum.alpha(0.3)
            vedo_frustum.color("c")
            # geom_list.append(frustum_lines)
            geom_list.append(quad_mesh_vedo)
            geom_list.append(vedo_frustum)

            print("cam pose: " + str(placed_cameras[i].pose))

            pose = placed_cameras[i].pose
            plt2.camera.SetPosition(pose[0], pose[1], pose[2])
            R = rot3d_from_rtp(np.array([pose[-1], -pose[-3], -pose[-2]]))
            print("R: " + str(R))
            focus = pose[:3] + R[:, 0]
            print("focus: " + str(focus))
            plt2.camera.SetFocalPoint(focus[0], focus[1], focus[2])
            plt2.camera.SetViewUp(R[:, 2])
            plt2.camera.SetDistance(5)
            plt2.camera.SetClippingRange([0.2, 10])
            plt2.camera.SetViewAngle(placed_cameras[i].fov[-1]*1.1)
            plt2.show(resetcam=False)

            plt3.set_position(pose[:3])
            plt3.set_viewup(R[:, 2])
            plt3.set_focus(focus)
            plt3.show(auto_close=False, interactive=False)

        else:
            # vedo_frustum.alpha(0.1)
            # vedo_frustum.color("p")
            quad_mesh_vedo.color('o')
            # geom_list.append(frustum_lines)
            geom_list.append(quad_mesh_vedo)
            # geom_list.append(vedo_frustum)
            plt2.add(quad_mesh_vedo)
            plt3.add_mesh(quad_mesh_pv)

    # testing:
    test = (-plt3.get_image_depth(fill_value=0) / placed_cameras[0].range[1])
    test[test > 1] = 1.0
    test[test < 0] = 0.0
    test = np.round(test * np.iinfo(np.uint16).max)
    test = test.astype(np.uint16)

    # test_cv = cv.normalize(-test / placed_cameras[0].range[1] * 255, 0, 255, cv.NORM_MINMAX)
    cv.imshow('test', test)
    cv.waitKey()

    for actor in geom_list:
        plt1.add(actor)

    plt1.add(vedo.Text2D("Press 'y' to approve placement. Press 'n' to reject. "
                         "\nPress 'f' for front culling, 'b' for back culling, 'c' to disable culling "
                         "\nPress 'q' when done",
                         pos='bottom-right', c='dg', bg='g', font='Godsway'))
    plt1.camera.SetPosition(7.8*np.cos(-145*np.pi/180.0), 7.8*np.sin(-145*np.pi/180.0), 3.)
    plt1.camera.SetFocalPoint(-0.026929191045848594, 0.5783514020506139, 0.8268966663940324)
    plt1.camera.SetViewUp(np.array([0, 0, 1]))
    plt1.camera.SetDistance(7.8)
    plt1.camera.SetClippingRange([0.25, 10])
    plt1.show(resetcam=False)

    return approved


def perch_keyfunc(key):
    global approved
    global user_responded

    actors = plt1.actors
    actors.pop(-1)
    plt1.clear()
    plt1.add(actors)
    if key == 'y':
        approved = True
        plt1.add(vedo.Text2D("Approve placement? pless 'q' to confirm.",
                             pos='bottom-right', c='dg', bg='g', font='Godsway'))
    elif key == 'n':
        approved = False
        plt1.add(vedo.Text2D("Reject placement? Press 'q' to confirm.",
                             pos='bottom-right', c='dg', bg='g', font='Godsway'))
    elif key == 'b':
        actors[0].backFaceCulling(True)
        actors[0].frontFaceCulling(False)
        plt1.add(vedo.Text2D("Press 'y' to approve placement. Press 'n' to reject. "
                             "\nPress 'f' for front culling, 'b' for back culling, 'c' to disable culling "
                             "\nPress 'q' when done",
                             pos='bottom-right', c='dg', bg='g', font='Godsway'))
    elif key == 'f':
        actors[0].backFaceCulling(False)
        actors[0].frontFaceCulling(True)
        plt1.add(vedo.Text2D("Press 'y' to approve placement. Press 'n' to reject. "
                             "\nPress 'f' for front culling, 'b' for back culling, 'c' to disable culling "
                             "\nPress 'q' when done",
                             pos='bottom-right', c='dg', bg='g', font='Godsway'))
    elif key == 'c':
        actors[0].backFaceCulling(False)
        actors[0].frontFaceCulling(False)
        plt1.add(vedo.Text2D("Press 'y' to approve placement. Press 'n' to reject. "
                             "\nPress 'f' for front culling, 'b' for back culling, 'c' to disable culling "
                             "\nPress 'q' when done",
                             pos='bottom-right', c='dg', bg='g', font='Godsway'))
    else:
        plt1.add(vedo.Text2D("Press 'y' to approve placement. Press 'n' to reject. "
                             "\nPress 'f' for front culling, 'b' for back culling, 'c' to disable culling "
                             "\nPress 'q' when done",
                             pos='bottom-right', c='dg', bg='g', font='Godsway'))

    plt1.show()

