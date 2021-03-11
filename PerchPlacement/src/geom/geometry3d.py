import numpy as np
import warnings
import copy
import random

import trimesh
from plyfile import PlyData
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import cv2 as cv
import triangle
import mapbox_earcut
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from opt.optimization_options import CameraPlacementOptions
from tools.tools import set_axes_equal
import pymesh
import vedo


class Mesh:
    def __init__(self, points, faces, face_normals=None, face_colors=None, post_process=False, classification="none",
                 options=CameraPlacementOptions(), kdt_data=None, vertex_colors=None, is_plane_mesh=True):

        self.is_valid = True

        self.num_faces = len(faces)
        self.num_points = len(points)

        self.removed_points = []
        self.removed_faces = []

        self.points = points  # point coordinates in 3D
        self.faces = faces  # face indices
        if points is None or faces is None:
            print("error... mesh has no faces..")
            self.is_valid = False
        self.face_centers = np.zeros([self.num_faces, 3])
        self.face_areas = np.zeros([self.num_faces, 1])
        self.calculate_face_centers()
        self.classification = classification

        self.eig0_range = 0.0
        self.eig1_range = 0.0

        if face_normals is not None:
            self.face_normals = face_normals
        else:
            self.face_normals = np.zeros([self.num_faces, 3])  # temp assignment
            self.estimate_face_normals()

        if face_colors is None:
            self.face_colors = np.ones([len(faces), 4])
        else:
            self.face_colors = face_colors

        if vertex_colors is None:
            self.vertex_colors = np.ones([len(points), 4])
        else:
            self.vertex_colors = vertex_colors

        self.kdt_data = kdt_data
        self.obstacles_near_mesh = []
        self.target_obstructions = []
        self.target_center = np.zeros(3)

        # rospy.loginfo("Post processing imported mesh...")
        if is_plane_mesh:
            pca = PCA(n_components=3)
            self.principle_axis_projected_points = pca.fit_transform(self.points)

            # pca.components has shape n_components x n_features therefore eigenvectors are listed row-wise
            self.mesh_normal = pca.components_[2, :].copy()
            self.mesh_normal /= np.linalg.norm(self.mesh_normal)
            assert np.abs(np.linalg.norm(self.mesh_normal) - 1) <= np.finfo(float).eps
            self.eigen_vectors = pca.components_
            self.pca_mean = pca.mean_
            self.projected_points = self.points.copy()  # not sure if necessary
            # pca = PCA(n_components=2)
            self.projected_points = self.principle_axis_projected_points[:, :2]  # pca.fit_transform(self.projected_points)
            # self.planar_points = pca.inverse_transform(self.projected_points)

        else:
            self.principle_axis_projected_points = None
            self.mesh_normal = None
            self.eigen_vectors = None
            self.pca_mean = None
            self.projected_points = None

        self.net_area = 0
        self.centroid = np.zeros(3)
        self.avg_radius = 0
        self.calculate_centroid()

        self.mesh_loops = []
        self.perimeter_loop = []
        self.inner_loops = []
        self.num_perim_loops = 0

        self.bins = np.zeros(len(self.faces)+1)

        if post_process:
            self.post_process_mesh(options)

        self.update_bins()

    def post_process_mesh(self, options):
        if options.nearest_neighbor_restriction and self.kdt_data is not None:

            # store IDs of partitioned meshes in self.surfaces
            surf_id = self.kdt_data.clusters[np.where(np.all(self.kdt_data.colors == self.face_colors[0, :], axis=1))][0]

            # use KD tree and nearest neighbor search to find distance to nearest obstacle
            idx = self.kdt_data.tree.query_radius(self.points, r=options.min_obstacle_radius)
            # TODO: replace with OBB Tree
            # stack all idx elements into a single vector.
            idx_vec = idx[0]
            for i in range(1, len(idx)):
                idx_vec = np.append(idx_vec, idx[i])

            off_surf_pts = np.unique(idx_vec)  # the set of unique points which are within the min radius

            # want all faces containing the points
            # if i have the id of the point in the full/redundant list, I can get do some clever mod indexing
            obstacle_faces = []
            debug_faces = []
            for p in off_surf_pts:
                for fid in self.kdt_data.point_face_map[p]:
                    if self.kdt_data.clusters[self.kdt_data.cluster_ids[fid]] != surf_id:
                        obstacle_faces.append(fid)
                    debug_faces.append(fid)
            obstacle_faces = np.unique(np.asarray(obstacle_faces))
            for f in obstacle_faces:
                self.obstacles_near_mesh.append(self.kdt_data.points[self.kdt_data.faces[f, :], :])

        self.erode_mesh(landing_window=options.min_perch_window, px_per_m=options.erosion_raster_density,
                        window_shape=options.perch_window_shape, gravity_direction=options.gravity_direction,
                        angle_threshold=options.angle_threshold)
        if self.is_valid:
            self.net_area = 0
            self.centroid = np.zeros(3)
            self.avg_radius = 0
            self.calculate_centroid()

            # # re-orient 3rd eigenvector (normal direction) according to mesh normal
            # if np.dot(self.eigen_vectors[-1], self.mesh_normal) < 0:
            #     self.eigen_vectors[-1] = -self.eigen_vectors[-1]
        self.update_bins()

    def update_bins(self):
        self.bins[1:] = np.cumsum(self.face_areas)

    def find_min_height(self, options, gravity_direction):
        # store IDs of partitioned meshes in self.surfaces
        surf_id = self.kdt_data.clusters[
            np.where(np.all(self.kdt_data.colors == self.face_colors[0, :], axis=1))][0]

        # use KD tree and nearest neighbor search to find distance to nearest obstacle
        idx = self.kdt_data.tree.query_radius(self.points, r=options.min_recovery_height)

        normal = self.eigen_vectors[-1]
        if normal.dot(self.mesh_normal) < 0:
            normal *= -1

        pt_zs = self.points.dot(-gravity_direction)
        h_vec = np.cross(gravity_direction, normal)
        h_vec /= np.linalg.norm(h_vec)
        pt_xs = self.points.dot(h_vec)
        z_max = np.max(pt_zs)
        # debugging
        # ax = Axes3D(plt.figure())
        # self.plot_mesh(ax)
        # ax.scatter3D(self.points[:, 0], self.points[:, 1], self.points[:, 2])

        # check a) if obstructions are below the point
        d_mins = np.nan * np.ones(len(self.points))
        diff_cluster = [np.ones_like(ids, dtype=bool) for ids in idx]

        for i in range(len(idx)):
            if len(idx[i]) > 0:
                for p in range(len(idx[i])):
                    for fid in self.kdt_data.point_face_map[idx[i][p]]:
                        if self.kdt_data.clusters[self.kdt_data.cluster_ids[fid]] == surf_id:
                            diff_cluster[i][p] = False
                            break

                z = (self.kdt_data.points[idx[i]]).dot(-gravity_direction)
                dy = (self.kdt_data.points[idx[i]] - self.points[i]).dot(normal)
                dz = (self.kdt_data.points[idx[i]] - self.points[i]).dot(-gravity_direction)
                dx = (self.kdt_data.points[idx[i]] - self.points[i]).dot(h_vec)
                # keep only points below AND in front AND that aren't in the same cluster
                # pts_of_interest = (pt_zs[i] > z) * (y > 0.0) * diff_cluster[i]
                pts_of_interest = (dy > 0.0) * (dz < options.min_perch_window[1]/2) * \
                                  (np.abs(dx) < options.min_perch_window[0]/2) * diff_cluster[i]
                idx[i] = idx[i][pts_of_interest]
                z = z[pts_of_interest]
                # if below, then keep the closest
                if len(idx[i]) > 0:
                    d_mins[i] = np.max(z)

        # now, order the points based on their horizontal position
        x = np.array([pt_xs, d_mins, self.points[:, 0], self.points[:, 1], self.points[:, 2]]).T
        x = x[x[:, 0].argsort()]
        pt_zs = x[:, 2:].dot(-gravity_direction)
        z_min = np.nanmin(x[:, 1])

        # make sure at least the first point is non-nan
        if np.isnan(x[0, 1]):
            x[0, 1] = z_min

        # then replace all nan points with their left neighbors; interpolation might be better, but this is good enough
        for i in range(len(x)):
            if np.isnan(x[i, 1]):
                x[i, 1] = x[i-1, 1]

        # moving average to smooth things out a bit
        x_copy = x.copy()
        smoothing_range = 8
        for i in range(smoothing_range//2, len(x)-smoothing_range//2):
            x[i, 1] = np.mean(x_copy[i-smoothing_range//2:i+smoothing_range//2+1, 1])

        # convert to 3D coords using pca projection, then return list
        min_heights = self.pca_mean + h_vec*(x[:, 0] - self.pca_mean.dot(h_vec)).reshape(-1, 1) + \
                      -gravity_direction * (options.min_recovery_height +
                                            x[:, 1] - self.pca_mean.dot(-gravity_direction)).reshape([-1, 1])
        # debug = self.pca_mean + h_vec*(x[:, 0].reshape([-1, 1]) - self.centroid.dot(h_vec)) - \
        #               gravity_direction * (pt_zs - self.centroid.dot(gravity_direction)).reshape([-1, 1])
        # debug2 = x[:, 2:] - normal.dot(x[:, 2:].T - self.pca_mean.reshape([-1, 1])).reshape([-1, 1])
        # debug3 = debug2 + (x[:, 1] - pt_zs).reshape([-1, 1]) * gravity_direction.reshape([1, -1])
        # ax.plot3D(min_heights[:, 0], min_heights[:, 1], min_heights[:, 2])
        # ax.plot3D(debug2[:, 0], debug2[:, 1], debug2[:, 2])
        # plt.figure()
        # plt.plot(x[:, 0], x[:, 1])
        # plt.plot(x[:, 0], x[:, 4])
        # ax.plot3D(debug3[:, 0], debug3[:, 1], debug3[:, 2])

        return min_heights


    def calculate_centroid(self):
        net_area = 0
        radius_sum = 0
        net_weighted_center = np.zeros(3)
        fa = np.zeros(self.num_faces)
        # net_weighted_normal = np.zeros(3)

        # net_normal = np.zeros(3)  # for debugging

        for i in range(self.num_faces):
            v1, v2 = self.get_vectors_from_face(i)

            a = 1/2*np.linalg.norm(np.cross(v1, v2))
            c = 1/3*(self.points[self.faces[i, 0], :] + self.points[self.faces[i, 1], :] +
                     self.points[self.faces[i, 2], :])
            fa[i] = a
            net_area += a
            net_weighted_center += a*c

            A = dist(self.points[self.faces[i][0]], self.points[self.faces[i][1]])
            B = dist(self.points[self.faces[i][0]], self.points[self.faces[i][2]])
            C = dist(self.points[self.faces[i][1]], self.points[self.faces[i][2]])

            sides = np.array([A, B, C])
            hyp = np.max(sides)
            if a > 0:
                rad = A*B*C/(4*a)
                radius_sum += min(hyp/2, rad)

        self.face_areas = fa
        self.net_area = net_area
        self.centroid = net_weighted_center/net_area
        self.avg_radius = radius_sum/self.num_faces

    def generate_points_on_mesh(self, n=1, f=None, r1=None, r2=None, weight_by_area=True):
        points = np.zeros([n, 3])
        select_face = False
        generate_vectors = False

        if f is None:
            select_face = True
        if r1 is None or r2 is None:
            generate_vectors = True

        for i in range(n):
            # select random face...
            if select_face:
                if weight_by_area:
                    p_scaled = random.uniform(0, 1) * self.net_area
                    f = 0
                    cumulative_area = self.face_areas[f]
                    while cumulative_area <= p_scaled:
                        f = f + 1
                        cumulative_area = cumulative_area + self.face_areas[f]
                else:
                    f = random.randint(0, self.num_faces-1)

            v1, v2 = self.get_vectors_from_face(f)

            # randomly select over uniform parallelogram
            if generate_vectors:
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)

            # limit to only be within the triangle
            if r1+r2 > 1:
                r1 = 1-r1
                r2 = 1-r2

            # apply rand vars to sample over triangle
            points[i, :] = r1*v1 + r2*v2 + self.points[self.faces[f, :][0], :]

        # # for debug
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        # ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2])
        # plt.show()

        return points

    def get_vectors_from_face(self, index):
        f = self.faces[index, :]
        v1 = self.points[f[1], :] - self.points[f[0], :]
        v2 = self.points[f[2], :] - self.points[f[0], :]
        return v1, v2

    def __add__(self, mesh2):

        self.points = np.append(self.points, mesh2.points, 0)  # point coordinates in 3D
        self.faces = np.append(self.faces, mesh2.faces+self.num_points, 0)  # face indices
        self.face_normals = np.append(self.face_normals, mesh2.face_normals, 0)
        self.face_centers = np.append(self.face_centers, mesh2.faces, 0)
        self.face_areas = np.append(self.face_areas, mesh2.face_areas, 0)

        for loop in mesh2.perimeter_loop:
            self.perimeter_loop.append(list(np.asarray(loop)+self.num_points))
        self.num_perim_loops += mesh2.num_perim_loops
        # for loop in mesh2.inner_loops:
        #     self.inner_loops.append(list(np.asarray(mesh2.inner_loops) + self.num_points))
        # for loop in mesh2.mesh_loops:
        #     self.mesh_loops.append(list(np.asarray(mesh2.mesh_loops) + self.num_points))

        self.num_faces += mesh2.num_faces
        self.num_points += mesh2.num_points

        self.face_colors = np.append(self.face_colors, mesh2.face_colors, 0)

        self.centroid = (self.centroid*self.net_area + mesh2.centroid*mesh2.net_area)/(self.net_area + mesh2.net_area)
        self.mesh_normal = (self.mesh_normal*self.net_area + mesh2.mesh_normal*mesh2.net_area)
        self.mesh_normal /= np.linalg.norm(self.mesh_normal)  # re-normalize
        assert np.abs(np.linalg.norm(self.mesh_normal) - 1) <= np.finfo(float).eps

        self.net_area += mesh2.net_area
        self.update_bins()

        return self

    def reorient_mesh(self, R):
        self.points = np.dot(R, self.points.transpose()).transpose()
        self.face_normals = np.dot(R, self.face_normals.transpose()).transpose()
        self.face_centers = np.dot(R, self.face_centers.transpose()).transpose()

        if self.mesh_normal is not None:
            self.mesh_normal = np.dot(R, self.mesh_normal)
            self.mesh_normal /= np.linalg.norm(self.mesh_normal)  # re-normalize
            assert np.abs(np.linalg.norm(self.mesh_normal) - 1) <= np.finfo(float).eps

        if self.centroid is not None:
            self.centroid = np.dot(R, self.centroid)

        if self.eigen_vectors is not None:
            self.eigen_vectors = np.dot(R, self.eigen_vectors.transpose()).transpose()
        if self.pca_mean is not None:
            self.pca_mean = np.dot(R, self.pca_mean)

    def calculate_face_centers(self):
        fc = np.zeros([len(self.faces), 3])
        for i in range(len(self.faces)):
            f = self.faces[i, :]
            pts = self.points[f, :]
            fc[i, :] = np.mean(pts, 0)
        self.face_centers = fc

    def estimate_face_normals(self):
        fc = np.zeros([len(self.faces), 3])
        invalid_faces = np.zeros_like(fc[:,0], dtype=bool)
        for i in range(len(self.faces)):
            f = self.faces[i, :]
            pts = self.points[f, :]
            v1 = (pts[0, :] - pts[1, :])
            v1 = v1 / np.linalg.norm(v1)
            v2 = (pts[0, :] - pts[2, :])
            v2 = v2 / np.linalg.norm(v2)
            norm = np.cross(v1, v2)*np.linalg.norm(v2)
            if np.linalg.norm(norm) == 0 or np.isnan(np.linalg.norm(norm)).any():
                warnings.warn("PROBLEM! zero area face. this should not happen")
                invalid_faces[i] = True
            else:
                fc[i, :] = norm / np.linalg.norm(norm)
        self.face_normals = fc
        if invalid_faces.any():
            self.remove_invalid_faces(np.logical_not(invalid_faces))

    def remove_invalid_faces(self, idx):
        self.faces = self.faces[idx]
        if len(self.face_colors) == len(idx):
            self.face_colors = self.face_colors[idx]
        self.face_areas = self.face_areas[idx]
        self.face_normals = self.face_normals[idx]
        self.face_centers = self.face_centers[idx]

    def plot_mesh(self, ax=None, color=None, opacity=None):
        if ax is None:
            fig = plt.figure(0)
            ax = Axes3D(fig)
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        if opacity is None:
            opacity = 200.0
        if color is None:
            if len(self.face_colors.shape) > 1 and self.face_colors.shape[-1] < 4:
                if np.max(self.face_colors) > 1.0:
                    color = np.append(self.face_colors[0, :], opacity)/255.0
                else:
                    color = np.append(self.face_colors[0, :], opacity/255.0)
            elif np.max(self.face_colors) > 1.0:
                color = self.face_colors[0, :] / 255.0
        elif len(color.shape) < 4:
            color = np.append(color, opacity)/255

        ax.plot_trisurf(x, y, triangles=self.faces, color=color, Z=z)
        set_axes_equal(ax)
        return ax


    def plot_normal(self, ax=None, color=None, opacity=None):
        if ax is None:
            fig = plt.figure(0)
            ax = Axes3D(fig)
        pts = np.array([self.centroid, self.centroid + self.mesh_normal])
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        if opacity is None:
            opacity = 200.0
        if color is None:
            if len(self.face_colors.shape) > 1 and self.face_colors.shape[-1] < 4:
                if np.max(self.face_colors) > 1.0:
                    color = np.append(self.face_colors[0, :], opacity)/255.0
                else:
                    color = np.append(self.face_colors[0, :], opacity/255.0)
            elif np.max(self.face_colors) > 1.0:
                color = self.face_colors[0, :] / 255.0
        elif len(color.shape) < 4:
            color = np.append(color, opacity)/255

        ax.plot(x, y, z, color=color)
        set_axes_equal(ax)
        return ax

    def plot_multi_colored_mesh(self, ax=None, opacity=None):
        if ax is None:
            fig = plt.figure(0)
            ax = Axes3D(fig)
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        if opacity is None:
            opacity = 220.0

        for p in range(len(x)):
            color = np.append(self.face_colors[p], opacity)/255
            ax.plot_trisurf(x[self.faces[p]], y[self.faces[p]], triangles=np.array([0, 1, 2]), color=color,
                            Z=z[self.faces[p]])
        set_axes_equal(ax)
        plt.show()

    def convert_projection_to_img(self, px_per_m=100):
        if self.projected_points is not None:
            # step one: convert to raster
            # find size of bounding box
            h_max = np.max(self.projected_points[:, 1])
            h_min = np.min(self.projected_points[:, 1])
            h = h_max - h_min

            w_max = np.max(self.projected_points[:, 0])
            w_min = np.min(self.projected_points[:, 0])
            w = w_max - w_min

            px_h = np.ceil(h * px_per_m)
            px_w = np.ceil(w * px_per_m)
            c = np.array([w_min, h_min])

            img = np.zeros(np.array([px_h, px_w]).astype(int), np.uint8)

            # rasterize mesh by filling in loops in the mesh
            if self.perimeter_loop is not None and len(self.perimeter_loop) > 0:
                for pl in self.perimeter_loop:
                    # convert to indices:
                    pts = self.projected_points[pl, :]
                    idx = np.round((pts - c) * px_per_m).astype(int)
                    cv.drawContours(img, [idx], 0, 255, -1)  # use a color of 255 for outer loops
                    # cv.drawContours(debug_img, [idx], 0, [255, 255, 255], -1)
                    # cv.drawContours(debug_img2, [idx], 0, [255, 255, 255], -1)

                for il in self.inner_loops:
                    # convert to indices:
                    pts = self.projected_points[il, :]
                    idx = np.round((pts - c) * px_per_m).astype(int)
                    cv.drawContours(img, [idx], 0, 0, -1)  # use a color of 0 for inner loops
                    # cv.drawContours(debug_img, [idx], 0, [255, 0, 0], -1)
                    # cv.drawContours(debug_img2, [idx], 0, [255, 0, 0], -1)

            else:
                # slower, but more robust. vectorizing draw contours here leads to strange results at overlap
                for f in self.faces:
                    pts = self.projected_points[f, :]
                    idx = np.round((pts - c) * px_per_m).astype(int)
                    cv.drawContours(img, [idx], 0, 255, -1)  # filling in populated triangles therefore use 255

            return img, c

        warnings.warn("Generate planar representation before eroding")
        self.is_valid = False
        return np.nan*np.eye(3), -1

    def remove_regions_from_img(self, img, regions, c, px_per_m=100, project_on_horizon=False):
        # pca = PCA(n_components=3)
        # pca.fit(self.points)
        if project_on_horizon:
            normal = self.eigen_vectors[-1]
            if normal.dot(self.mesh_normal) < 0:
                normal *= -1.0
            eig_vecs = np.array([np.cross(np.array([0, 0, -1]), normal), np.array([0, 0, -1]), normal])
        else:
            eig_vecs = self.eigen_vectors
        for f in regions:
            # Step 1: project 3D points onto plane
            # f is a 3x3 matrix where each row is a point in a triangle; want transpose of this to be able to
            # subtract mean, then multiply with PCA eigenvectors (rotation matrix)
            f_prj = np.dot(f-self.pca_mean, eig_vecs.T)
            # f_prj = np.dot(f-pca.mean_, pca.components_.T)  # after transpose, each row has projected points
            # only want top 2 components...
            f_prj = f_prj[:, :2]
            # Step 2: map projected points into image coordinates
            idx = np.round((f_prj - c) * px_per_m).astype(int)
            # Step 3: draw contour on image
            cv.drawContours(img, [idx], 0, 0, -1)  # remove these from image - use color of 0\
            # cv.drawContours(debug_img, [idx], 0, [0, 0, 255], -1)
        return img

    def keep_only_regions_in_img(self, img, regions, c, px_per_m=100):
        img_mask = np.zeros_like(img)
        for f in regions:
            # Step 1: project 3D points onto plane
            # f is a 3x3 matrix where each row is a point in a triangle; want transpose of this to be able to
            # subtract mean, then multiply with PCA eigenvectors (rotation matrix)
            f_prj = np.dot(f-self.pca_mean, self.eigen_vectors.T)
            # f_prj = np.dot(f-pca.mean_, pca.components_.T)  # after transpose, each row has projected points
            # only want top 2 components...
            f_prj = f_prj[:, :2]
            # Step 2: map projected points into image coordinates
            idx = np.round((f_prj - c) * px_per_m).astype(int)
            # Step 3: draw contour on image
            cv.drawContours(img_mask, [idx], 0, 255, -1)  # remove these from image - use color of 0\
            # cv.drawContours(debug_img, [idx], 0, [0, 0, 255], -1)
        img = img * (img_mask / 255)
        return cv.convertScaleAbs(img)

    def reconstruct_mesh_from_img(self, img, px_per_m, c):
        # convert back to polygon
        ref, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

        # if contours is empty, return 0 (signal to remove mesh from set)
        if len(contours) == 0:
            # erosion removed entire mesh
            self.is_valid = False
            return 0
        else:
            hierarchy = hierarchy.squeeze()
            # NOTE: testing revealed that TC89_KCOS consistently gives the most simplified result, while matching
            # the geometry well

            # reset parameters, then re append
            self.points = None
            self.faces = None
            self.perimeter_loop = []
            self.mesh_loops = []
            self.inner_loops = []
            # segs = []  # tracks all loops

            # hierarchy is reduced to a 1D vector if only one contour exists... Numpy doesn't like giving a
            # 2D index for a single row vector... reshaping here in that case to prevent errors:
            if hierarchy.ndim == 1:
                hierarchy = hierarchy.reshape([1, hierarchy.size])
            is_parent = hierarchy[:, -1] == -1
            parent_ids = []
            parent_reverse_id = np.zeros(len(contours), dtype=int)
            parents = []
            parent_children = []
            # parent_loops = []
            # parents_child_loops = []

            # pull out the parents; create a list for each parent to store their children
            for i in range(len(contours)):
                contours[i] = np.squeeze(contours[i].astype(float) / px_per_m + c)
                n_pts = len(contours[i])
                # seg = []  # keep only current loop indices for later use...

                if len(contours[i]) > 3:
                    if is_parent[i]:
                        parents.append(contours[i])
                        parent_reverse_id[i] = len(parent_ids)
                        parent_ids.append(i)
                        parent_children.append([])
                        # parents_child_loops.append([])

                        # loop = np.vstack([np.arange(0, n_pts), np.arange(1, n_pts + 1)]).T
                        # loop[-1, -1] = 0
                        # parent_loops.append(loop)

            # find all the children for each parent
            for i in range(len(contours)):
                # ignore anything with a hierarchy level lower than a 1st child
                # (no grandkids e.g. islands in a lake in a continent)
                n_pts = len(contours[i])
                if hierarchy[i, -1] > -1 and n_pts > 3:
                    parent_children[parent_reverse_id[hierarchy[i, -1]]].append(contours[i])
                    # loop = np.vstack([np.arange(0, n_pts), np.arange(1, n_pts + 1)]).T
                    # loop[-1, -1] = 0
                    # parents_child_loops[parent_reverse_id[hierarchy[i, -1]]].append(loop)

            vs = np.zeros([0, 2])
            fs = np.zeros([0, 3], dtype=int)

            if len(parents) == 0:
                self.is_valid = False
                return 0

            num_vertices = 0

            for i in range(len(parents)):
                p_loop = np.arange(0, len(parents[i])) + len(vs)
                self.perimeter_loop.append(p_loop)
                self.mesh_loops.append(p_loop)

                # triangulate!
                verts = parents[i]
                rings = np.array([len(verts)])
                for j in range(len(parent_children[i])):
                    c_loop = np.arange(0, len(parent_children[i][j])) + len(vs) + len(verts)
                    self.inner_loops.append(c_loop)
                    self.mesh_loops.append(c_loop)

                    verts = np.vstack([verts, parent_children[i][j]])
                    rings = np.hstack([rings, len(verts)])
                tri_faces = mapbox_earcut.triangulate_float32(verts, rings).reshape([-1, 3])
                fs = np.vstack([fs, tri_faces + len(vs)])
                vs = np.vstack([vs, verts])

            for ml in self.mesh_loops:  # FOR DEBUGGING
                if np.max(ml) >= len(vs):
                    warnings.warn("PROBLEM! Loop index is too high")

            # transform back into 3D
            # multiply by eigenvectors, then add mean
            self.points = np.dot(vs, self.eigen_vectors[:2, :]) + self.pca_mean

            # check for and remove faces with zero area...
            i = 0
            j = len(fs)
            while i < j:
                pts = self.points[fs[i], :]
                v1 = (pts[0, :] - pts[1, :])
                v1 = v1 / np.linalg.norm(v1)
                v2 = (pts[0, :] - pts[2, :])
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v1, v2)
                if np.linalg.norm(v3) == 0 or np.isnan(v3).any():
                    # remove face
                    fs = np.delete(fs, i, axis=0)
                else:
                    i += 1
                j = len(fs)

            if len(fs) < 1:
                self.is_valid = False
                return 0

            self.eig0_range = np.linalg.norm(self.eigen_vectors[0, :]) * \
                              (np.max(vs[:, 0]) -
                               np.min(vs[:, 0]))
            self.eig1_range = np.linalg.norm(self.eigen_vectors[1, :]) * \
                              (np.max(vs[:, 1]) -
                               np.min(vs[:, 1]))
            self.faces = fs
            # set inner and outer loops
            # parent loops will have column 4 == -1... could split into multiple meshes here... not a big
            # benefit though. set everything else to be an inner loop...
            self.projected_points = copy.copy(vs)
            # PCA, or decompose the points using the stored eigenvectors

            self.num_faces = len(self.faces)
            self.num_points = len(self.points)

            self.calculate_face_centers()
            self.estimate_face_normals()
            self.calculate_centroid()

            self.face_colors = self.face_colors[:self.num_faces]  # This assumes the face colors were uniform

            return 1

    def remove_regions_from_mesh(self, regions, px_per_m=100, project_on_horizon=False):
        # self.plot_mesh()
        img, c = self.convert_projection_to_img(px_per_m)
        if np.isnan(img).any():
            return -1
        else:
            img = self.remove_regions_from_img(img, regions, c, px_per_m, project_on_horizon=project_on_horizon)
            code = self.reconstruct_mesh_from_img(img, px_per_m, c)

        # self.plot_mesh()

        return code

    def keep_only_regions_in_mesh(self, regions, px_per_m=100):
        # self.plot_mesh()
        img, c = self.convert_projection_to_img(px_per_m)
        if np.isnan(img).any():
            return -1
        else:
            img = self.keep_only_regions_in_img(img, regions, c, px_per_m)
            code = self.reconstruct_mesh_from_img(img, px_per_m, c)

        # self.plot_mesh()

        return code

    def erode_mesh(self, landing_window=None, px_per_m=100, window_shape="rectangle",
                   gravity_direction=np.array([0, 0, -1]), angle_threshold=20):
        # ensure 2D representation is already found
        if self.projected_points is not None:
            if landing_window is None:
                landing_window_copy = np.array([0.3, 0.3])
            else:
                landing_window_copy = landing_window.copy()

            if np.abs(np.dot(self.mesh_normal, gravity_direction)) > np.cos(np.pi/180*angle_threshold):
                # if floor or ceiling, use most symmetric window consisting of least conservative axis
                landing_window_copy = np.ones(2)*np.min(landing_window_copy)

            # else align landing window such that the vertical matches gravity direction using principal axes
            elif np.abs(np.dot(self.eigen_vectors[1], gravity_direction)) > \
                    np.abs(np.dot(self.eigen_vectors[0], gravity_direction)):
                landing_window_copy = np.array([landing_window_copy[1], landing_window_copy[0]])

            # step one: convert to raster
            # find size of bounding box
            img, c = self.convert_projection_to_img(px_per_m)

            n = self.mesh_normal
            # n = self.eigen_vectors[2]  # plane normal
            for f in self.target_obstructions:
                # Step 1: for each corner, project from the target centroid onto the plane
                w = f - self.centroid  # vector between obstruction vertices and mesh centroid [3x3]
                u = f - self.target_center  # vector direction of project line [3x3]
                s = np.dot(w, -n) / np.dot(u, n)  # [3x1]
                f_intersection = self.target_center + u * s  # [3x3]
                self.obstacles_near_mesh.append(f_intersection)

            img = self.remove_regions_from_img(img, self.obstacles_near_mesh, c, px_per_m)

            # erode image
            landing_window_copy *= px_per_m
            landing_window_copy = np.ceil(landing_window_copy) // 2 * 2 + 1  # ensure the kernel is Odd
            landing_window_copy = landing_window_copy.astype(int)

            if window_shape.lower() == "ellipse":
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (landing_window_copy[0], landing_window_copy[1]))
                img = cv.erode(img, kernel, iterations=1)
                # approximate decomposition is possible here as well... since erosion is a one-off calculation, i'll
                # leave this as a nice-to-have TODO for the future
            else:
                # can decompose rectangular kernels to get significant performance increase
                k1 = np.ones([landing_window_copy[0], 1], np.uint8)
                k2 = np.ones([1, landing_window_copy[1]], np.uint8)
                img = cv.erode(img, k1, iterations=1)
                # debug_img4 = img
                img = cv.erode(img, k2, iterations=1)

            code = self.reconstruct_mesh_from_img(img, px_per_m, c)
            return code

        else:
            warnings.warn("Generate planar representation before eroding")
            self.is_valid = False
            return -1

    def enforce_minimum_height(self, floor_centroid, min_height, gravity_direction=np.array([0, 0, -1]), px_per_m=100):
        if self.points is None or len(self.points) < 1:
            print("PROBLEM")

        h = np.dot(self.points - floor_centroid, -gravity_direction)
        valid_points = h > min_height
        # point_changed = np.zeros_like(valid_points, dtype=bool)
        # valid_faces = np.ones(len(self.faces), dtype=bool)

        # if this function is being called, I know that the surface is already approximately vertical
        if np.logical_not(valid_points).any():
            pts, faces = generate_box_mesh_components(self, -gravity_direction * min_height, z_floor=floor_centroid[2])

            # determine if I want front (8-9) or right face (6-7)
            front_dir = np.abs(np.dot(1/2*(pts[1]+pts[4]) - self.centroid, self.eigen_vectors[2]))
            right_dir = np.abs(np.dot(1/2*(pts[5]+pts[2]) - self.centroid, self.eigen_vectors[2]))
            if front_dir > right_dir:
                # front face is normal to surface
                regions = pts[faces[8:10]]  # make 2 triangles to represent the height cutoff

            else:
                # right face is normal to surface
                regions = pts[faces[6:8]]  # make 2 triangles to represent the height cutoff

            self.remove_regions_from_mesh(regions, px_per_m=px_per_m, project_on_horizon=True)

            if self.is_valid:
                self.calculate_centroid()
                self.calculate_face_centers()

    def enforce_minimum_height2(self, height_cutoff, gravity_direction=np.array([0, 0, -1]), px_per_m=100):
        if self.points is None or len(self.points) < 1:
            print("PROBLEM")
            return

        # find coordinate of lowest point (in gravity direction)
        dp = np.dot(self.points - self.centroid, gravity_direction)
        h_min = np.min(dp)
        h_max = np.max(dp)
        h_range = h_max - h_min

        regions = np.vstack([height_cutoff, np.flip(height_cutoff, axis=0)+gravity_direction*h_range])
        self.remove_regions_from_mesh([regions], px_per_m=px_per_m)

        if self.is_valid:
            self.calculate_centroid()
            self.calculate_face_centers()

    def apply_mesh_mask(self, mask_points, action="remove", px_per_m=100):
        if self.points is None or len(self.points) < 1:
            print("PROBLEM")
            return -1

        # TODO=================================
        pca = PCA(n_components=3)
        pca.fit(mask_points)
        eigs = pca.components_
        pts, faces = pcabox(mask_points, eigs[0], eigs[1], eigs[2])
        regions = pts[faces]  # make 2 triangles to represent the height cutoff
        # TODO=================================

        if action == "remove":
            self.remove_regions_from_mesh(regions, px_per_m=px_per_m)

        elif action == "keep":
            self.keep_only_regions_in_mesh(regions, px_per_m=px_per_m)

        if self.is_valid:
            self.calculate_centroid()
            self.calculate_face_centers()


def import_ply(filepath, post_process=False, options=CameraPlacementOptions(), kdt_data=None, store_vertex_color=False,
               is_plane_mesh=True):

    # Read PLY file
    ply_data = PlyData.read(filepath)
    # extract relevant data into the mesh class
    vertices = np.asarray(np.asarray(ply_data['vertex'].data).tolist())
    if store_vertex_color and ply_data.elements[0].header.__contains__('red'):
        vertex_colors = np.asarray(np.array([ply_data['vertex']['red'], ply_data['vertex']['green'],
                                           ply_data['vertex']['blue']]).tolist()).transpose()
    else:
        vertices = vertices[:, :3]  # only extract position, ignore any color info...
        vertex_colors = None

    # facedata = ply_data['face'].data
    if ply_data.elements[1].header.__contains__('red'):
        face_colors = np.asarray(np.array([ply_data['face']['red'], ply_data['face']['green'],
                                           ply_data['face']['blue']]).tolist()).transpose()
    else:
        face_colors = None
    faces = np.asarray(ply_data['face'].data['vertex_indices'].tolist()).astype(int)

    mesh = Mesh(points=vertices, faces=faces, face_colors=face_colors, post_process=post_process, options=options,
                kdt_data=kdt_data, vertex_colors=vertex_colors, is_plane_mesh=is_plane_mesh)

    return mesh


def dist(p, q=None):
    """
    A simple L2 norm distance function for numpy array inputs

    :param p: vector or point from which to calculate distance
    :param q: (optional) second point. If not specified, the distance from the origin is returned
    :return: distance between p and q
    """

    n = len(p)
    if q is None:
        q = np.zeros(n)

    distance = 0

    for i in range(n):
        distance = distance + np.power(p[i]-q[i], 2)

    return np.sqrt(distance)


def rot3d(eul, unit="degrees"):
    """
    Simple function that returns a 3D rotation matrix for a given angle
    :param eul: euler angles (roll, pitch, yaw)
    :param unit: specifies whether the angles are in degrees or radians
    :return: R, 3x3 rotation matrix for specified angles, assuming Z-Y-X rotation order
    """

    if unit == "degrees":
        eul = eul * np.pi/180

    roll = eul[0]
    pitch = eul[1]
    yaw = eul[2]

    yawMatrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitchMatrix = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    rollMatrix = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # R = np.dot(rollMatrix, np.dot(pitchMatrix, yawMatrix))
    R = np.dot(np.dot(yawMatrix, pitchMatrix), rollMatrix)
    return R


def rot3d_from_rtp(rtp, unit="degrees"):
    """
    Simple function that returns a 3D rotation matrix for a given roll tilt pan angles (roll is assumed to be zero).
    :param rtp: vector containing (roll, tilt, pan) for N points. If multiple points are being computed at once, input
     should be a matrix with dimensions 3xN (see Z1Y2X3 rotation here:
     https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix)
    :param unit: specifies whether the angle is in degrees or radians
    :return: The 3x3 rotation matrix corresponding to a pan, then tilt rotation.
    """
    # generate rotation matrix from camera pan and tilt. Matrix is derived from rotation sequence that is first pan
    # about original z axis, then tilt about new (post-pan rotation) y axis, y'. This is typical of a pan-tilt camera
    th = rtp[0]  # theta is roll
    phi = rtp[1]  # phi is tilt (aka pitch)
    psi = rtp[2]  # psi is pan (aka yaw)

    if unit == "degrees":
        # convert to radians
        th = th*np.pi/180
        phi = phi*np.pi/180
        psi = psi*np.pi/180

    c1 = np.cos(th)
    c2 = np.cos(phi)
    c3 = np.cos(psi)
    s1 = np.sin(th)
    s2 = np.sin(phi)
    s3 = np.sin(psi)

    R = np.array([[c3*c2, c3*s2*s1 - c1*s3, s3*s1 + c3*c1*s2],
                  [c2*s3, c3*c1 + s3*s2*s1, c1*s3*s2 - c3*s1],
                  [-s2,   c2*s1,            c2*c1]])
    if len(R.shape) > 2:
        R = np.moveaxis(R, -1, 0)
    return R


def rot3d_from_x_vec(v):
    """
    Generates a 3x3 Rotation Matrix which represents the orientation of an object, given the global direction of its
    local +Z axis
    :param v: the unit vector representing the direction of the object's z-axis in global space
    :return: R: 3x3 rotation matrix
    """
    if len(v.shape) > 1:
        n = np.linalg.norm(v, axis=1)
        v = np.einsum('ij,i->ij', v, 1/n).transpose()
    else:
        n = np.linalg.norm(v)
        v = v / n  # normalize, just in case

    pitch = np.arcsin(-v[2])  # between -pi/2 and pi/2
    # invalid_vals = np.abs(np.abs(v[0] / np.cos(pitch)) - 1.0) <= np.finfo(float).eps
    # yaw[invalid_vals] = np.arccos(v[0] / np.sign(np.cos(pitch)))
    #
    # else:
    yaw = np.arccos((v[0] / np.cos(pitch)))  # between 0 and pi
    roll = np.zeros_like(yaw)

    # R = np.array([[c3*c2, c3*s2*s1 - c1*s3, s3*s1 + c3*c1*s2],
    #               [c2*s3, c3*c1 + s3*s2*s1, c1*s3*s2 - c3*s1],
    #               [-s2,   c2*s1,            c2*c1]])

    R = rot3d_from_rtp(np.array([roll, pitch, yaw]), unit="radians")
    return R


def rot3d_from_z_vec(v):
    """
    Generates a 3x3 Rotation Matrix which represents the orientation of an object, given the global direction of its
    local +Z axis
    :param v: the unit vector representing the direction of the object's z-axis in global space
    :return: R: 3x3 rotation matrix
    """
    if len(v.shape) > 1:
        n = np.linalg.norm(v, axis=1)
        v = np.einsum('ij,i->ij', v, 1/n).transpose()
    else:
        n = np.linalg.norm(v)
        v = v / n  # normalize, just in case

    pitch = np.arccos(v[2])  # between 0 and pi
    yaw = np.arccos((v[0] / np.sin(pitch)))  # between
    roll = np.zeros_like(pitch)

    R = rot3d_from_rtp(np.array([roll, pitch, yaw]), unit="radians")
    return R


def eul_from_x_vec(v, units="degrees"):
    """
    Generates a 3x3 Rotation Matrix which represents the orientation of an object, given the global direction of its
    local +Z axis
    :param v: the unit vector representing the direction of the object's z-axis in global space
    :return: R: 3x3 rotation matrix
    """
    if len(v.shape) > 1:
        n = np.linalg.norm(v, axis=1)
        v = np.einsum('ij,i->ij', v, 1/n).transpose()
    else:
        n = np.linalg.norm(v)
        v = v / n  # normalize, just in case

    pitch = np.arcsin(-v[2])  # between -pi/2 and pi/2
    if np.abs(np.abs(v[0] / np.cos(pitch)) - 1.0) <= np.finfo(float).eps:
        yaw = np.arccos(v[0] / np.sign(np.cos(pitch)))
        # print("WARNING: INVALID INPUT")
        # print("v = " + str(v))
        # print("Pitch = " + str(pitch))
        # print("Should be less than 1; is: " + str(np.abs(np.abs(v[0] / np.cos(pitch)))))
    else:
        yaw = np.arccos((v[0] / np.cos(pitch)))  # between 0 and pi
    roll = 0

    if units == "degrees":
        pitch *= 180/np.pi
        yaw *= 180/np.pi
        roll *= 180/np.pi

    eul = np.array([roll, pitch, yaw])
    return eul


def eul_from_z_vec(v, units='degrees'):
    """
    Generates a Euler angles which represents the orientation of an object, given the global direction of its
    local +Z axis. Assumes roll=0
    :param v: the unit vector representing the direction of the object's z-axis in global space
    :return: R: 3x3 rotation matrix
    """
    if len(v.shape) > 1:
        n = np.linalg.norm(v, axis=1)
        v = np.einsum('ij,i->ij', v, 1/n).transpose()
    else:
        n = np.linalg.norm(v)
        v = v / n  # normalize, just in case

    pitch = np.arccos(v[2])  # between 0 and pi
    yaw = np.arccos((v[0] / np.sin(pitch)))  # between
    roll = np.zeros_like(pitch)

    if units == "degrees":
        pitch *= 180/np.pi
        yaw *= 180/np.pi
        roll *= 180/np.pi

    eul = np.array([roll, pitch, yaw])

    return eul


def eul_from_rot3d(R, units='degrees'):

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def eul_to_qt(eul, unit="degrees"):
    """
    Converts input euler angles into quaternion form

    :param eul: np.array containing roll, pitch, and yaw angles respectively
    :param unit: string determining whether the euler angles are in degrees or radians
    :return: np array containing the 4 quaternion components representing the rotation
    """

    roll = eul[0]
    pitch = eul[1]
    yaw = eul[2]

    if unit == "degrees":
        roll = roll * np.pi / 180
        pitch = pitch * np.pi / 180
        yaw = yaw * np.pi / 180

    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)

    return np.array([qx, qy, qz, qw])


def get_polygon_perimeter(poly_bounds):
    """
    Computes perimeter of a polygon
    :param poly_bounds: np.matrix containing polygon coordinates
    :return: polygon perimeter
    """
    cumulative_dist = 0
    for i in range(len(poly_bounds)):
        distance = dist(poly_bounds[i, :], poly_bounds[i - 1, :])
        cumulative_dist = cumulative_dist + distance

    return cumulative_dist


def generate_points_on_fov_front(position, R, fov, radius, n_horiz, n_vert):
    # each row is a relative vector from the camera to (in order): near top left -> cw -> nbl, far top left -> cw -> fbl
    if n_horiz >= 2:
        fh = np.linspace(-fov[0]/2, fov[0]/2, n_horiz)
    else:
        fh = np.array([0])

    if n_vert >= 2:
        fv = np.linspace(-fov[1]/2, fov[1]/2, n_vert)
    else:
        fv = np.array([0])
    
    pts = np.zeros([n_horiz, n_vert, 3])
    
    for i in range(n_horiz):
        for j in range(n_vert):
            R_ = rot3d_from_rtp(np.array([0, fv[j], fh[i]]))
            v = R_[:, 0]
            pts[i, j, :] = np.dot(R, v) * radius + position

    return pts


def generate_fov_end_mesh(O, r, R, fov, num_curve_points, normal="in"):
    """
    :param O -
    :param r -
    :param R -
    :param fov -
    :param num_curve_points -
    :param normal -

    :return pts -
    :return vertices -
    :return faces -
    """

    pts = generate_points_on_fov_front(position=O, R=R, fov=fov, radius=r, n_horiz=num_curve_points, 
                                       n_vert=num_curve_points)
    n_h = pts.shape[0]
    n_v = pts.shape[1]

    vs = np.zeros([n_h * n_v, 3])
    fs = []

    # first add points, then faces...?
    for h in range(n_h-1):
        for v in range(n_v-1):
            pid = np.array([[h+v*n_h, h+(v+1)*n_h],
                            [(h+1)+v*n_h, (h+1)+(v+1)*n_h]])
            vs[pid[0, 0], :] = pts[h, v, :]
            vs[pid[1, 0], :] = pts[h+1, v, :]
            vs[pid[0, 1], :] = pts[h, v+1, :]
            vs[pid[1, 1], :] = pts[h+1, v+1, :]  # will duplicate some of these but that's fine...

            if normal == "in":
                fs.append(np.array([pid[0, 0], pid[0, 1], pid[1, 0]]))
                fs.append(np.array([pid[0, 1], pid[1, 1], pid[1, 0]]))
            else:
                fs.append(np.array([pid[0, 1], pid[0, 0], pid[1, 0]]))
                fs.append(np.array([pid[1, 1], pid[0, 1], pid[1, 0]]))

    fs = np.asarray(fs)
    return pts, vs, fs


def generate_circle_pts(O, R, H_vec, num_pts):
    theta = np.linspace(0, np.pi * 2, num_pts + 1)

    x = R * np.cos(theta)
    y = R * np.sin(theta)
    z = np.zeros_like(x)

    pts = np.array([x, y, z])
    pts = np.dot(rot3d_from_z_vec(H_vec), pts).transpose() + O
    pts[-1] = O  # don't need the last point since we're wrapping 2pi to 0 to close the loop. use index to store origin
    return pts


def generate_circle_mesh(O, R, H_vec, num_pts):
    pts = generate_circle_pts(O, R, H_vec, num_pts)
    faces = np.zeros([num_pts, 3], dtype=int)

    for i in range(num_pts):
        faces[i, :] = np.array([(i + 1) % num_pts, i, num_pts])

    return pts, faces


def generate_cylinder_mesh(O, R, H_vec, num_pts):
    H = np.linalg.norm(H_vec)
    H_vec /= H

    pts = generate_circle_pts(O, R, H_vec, num_pts)
    pts_ = pts + H*H_vec

    faces = np.zeros([num_pts, 3])

    for i in range(num_pts):
        faces[i, :] = np.array([(i + 1) % num_pts, i, num_pts])
    faces_ = faces + num_pts + 1
    tmp = faces_[:, 0].copy()
    faces_[:, 0] = faces_[:, 1]
    faces_[:, 1] = tmp

    # loop through and make faces for the side walls:
    sz = num_pts
    side_faces = np.zeros([2 * sz, 3])
    for i in range(sz):
        side_faces[i, :] = np.array([i,
                                     (i + 1) % sz,
                                     i + num_pts + 1])
        side_faces[i + sz, :] = np.array([(i + 1) % sz,
                                          num_pts + 1 + (i + 1) % sz,
                                          i + num_pts + 1])

    pts = np.vstack([pts, pts_])
    faces = np.vstack([faces, faces_, side_faces])

    mesh = pymesh.form_mesh(vertices=pts, faces=faces)
    return mesh


def generate_rectangular_mesh(O, v1, v2):
    pts = np.vstack([O, O+v1, O+v1+v2, O+v2])
    faces = np.array([[0, 1, 2],
                      [0, 2, 3]])
    return pts, faces


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def generate_box_mesh_components(target_mesh, d, z_floor=None):
    flat_pts = target_mesh.points[:, :2]
    if len(flat_pts) > 1000:
        qtr_ids = (np.arange(len(flat_pts)) % 32) == 0
        flat_pts = flat_pts[qtr_ids]

    bbox = minimum_bounding_rectangle(flat_pts)
    z_min = np.min(target_mesh.points[:, 2])

    O1 = np.array([bbox[0, 0], bbox[0, 1], z_min])
    O2 = O1 + d
    O3 = np.array([bbox[1, 0], bbox[1, 1], z_min])
    O4 = np.array([bbox[3, 0], bbox[3, 1], z_min])

    v0 = O3 - O1
    v1 = O4 - O1

    pts = np.vstack([O1, O1 + v0, O1 + v0 + v1, O1 + v1,
                     O2, O2 + v0, O2 + v0 + v1, O2 + v1])

    faces = np.array([[0, 1, 2],  # bottom
                      [0, 2, 3],
                      [4, 6, 5],  # top
                      [4, 7, 6],
                      [0, 3, 4],  # left
                      [3, 7, 4],
                      [1, 5, 2],  # right
                      [2, 5, 6],
                      [0, 4, 1],  # front
                      [1, 4, 5],
                      [3, 2, 6],  # back
                      [3, 6, 7]], dtype=int)

    # DEBUGGING
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2])
    # target_mesh.plot_mesh(ax)
    # plt.waitforbuttonpress()
    return pts, faces


def pcabox(points, Pca1, Pca2, Pca3):
    """ Lo1, Hi1, Lo2, Hi2 = pcabox( Pointcloud, Pca1, Pca2 )
        In: Pointcloud: an N x 3 array of points
        In: Pca1, Pca2: unit vectors at right angles, from PCA
        Out: Lo1, Hi1, Lo2, Hi2: midpoints of the sides of a bounding box
    """
    # check N x 2 --
    assert points.ndim == 2  and  points.shape[1] == 3, points.shape

    C = np.mean(points, axis=0)  # the centre of all the points
    Pointcloud = points - C  # shift the cloud to be centred at [0 0]

    # distances along the long axis t * Pca1 --
    Dist1 = np.dot( Pointcloud, Pca1)
    Lo1 = np.min(Dist1) * Pca1
    Hi1 = np.max(Dist1) * Pca1

    # and along the short axis t * Pca2 --
    Dist2 = np.dot(Pointcloud, Pca2)
    Lo2 = np.min(Dist2) * Pca2
    Hi2 = np.max(Dist2) * Pca2

    # and along the short axis t * Pca3 --
    Dist3 = np.dot(Pointcloud, Pca3)
    Lo3 = np.min(Dist3) * Pca3
    Hi3 = np.max(Dist3) * Pca3

    box = np.array([Lo1, Hi1, Lo2, Hi2]) + C  # makes a cross shape from the center, along the 2 principle axes
    corners = np.vstack([Lo1+Lo2, Hi1+Lo2, Hi1+Hi2, Hi2 + Lo1]) + C
    faces = np.array([[0, 1, 2], [2, 3, 0]])

    return corners, faces  # 4 points


def generate_box_mesh(target_mesh, d):
    pts, faces = generate_box_mesh_components(target_mesh, d)
    mesh = pymesh.form_mesh(vertices=pts, faces=faces)

    #debugging
    # colors = np.vstack([np.linspace(0, 1, 12), np.linspace(1, 0, 12), np.linspace(0, .5, 12)]).T
    # a = trimesh.Trimesh(mesh.vertices, mesh.faces, face_colors=colors)
    return mesh


def generate_extruded_mesh(target_mesh, n, d, xlim=None, ylim=None, zlim=None):
    # bottom
    # TODO complete
    faces = target_mesh.faces
    pts = target_mesh.points

    # top
    top_pts = pts+n*d

    # faces = np.vstack([faces, top_faces])

    mesh = pymesh.form_mesh(vertices=pts, faces=faces)
    return mesh
