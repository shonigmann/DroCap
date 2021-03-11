import numpy as np


class CameraPlacementOptions:
    """
    A class containing several flags and opt to be used during the opt. These opt determine the
    per camera dimensionaility for search algorithms.
    """

    def __init__(self, variable_pan=True, variable_tilt=False, variable_zoom=False, perch_on_ceiling=True,
                 perch_on_walls=True, land_on_floor=True, perch_on_intermediate_angles=True,
                 variable_height=False, angle_mode="TOWARDS_TARGET", target_deviation=np.array([10.0, 30.0]),
                 min_vertices=0, mesh_env=False, surface_number=-1, map_to_flat_surface=False, 
                 vary_position_over_face=True,
                 min_perch_window=np.array([0.3, 0.5]), perch_window_shape="rectangle", angle_threshold=15,
                 dist_threshold=0.3, min_room_height=2.0, erosion_raster_density=100, min_obstacle_radius=0.5,
                 nearest_neighbor_restriction=False, target_volume_height=1.0, world_frame=np.eye(3),
                 min_recovery_height=0.0, minimum_score_deviation=None,
                 minimum_particle_deviation=None, inside_out_search=False, noise_resistant_particles=0,
                 noise_resistant_sample_size=1000, n_points=100):
        """

        :param variable_pan: boolean, whether the camera has variable pan or not
        :param variable_tilt: boolean, whether the camera has variable tilt or not
        :param variable_zoom: boolean, whether the camera has variable zoom or not
        :param perch_on_ceiling: boolean, whether the camera drones can be placed on the ceiling in the optimization
         search
        :param perch_on_walls: boolean, whether the camera drones can be placed on vertical walls in the optimization
         search
        :param land_on_floor: boolean, whether the camera drones can be placed on floor (upward facing horizontal)
         surfaces in the optimization search
        :param perch_on_intermediate_angles: boolean, whether unclassified (non-floor, non-ceil, non-wall) surfaces,
         should be included in the search space
        :param variable_height: boolean, whether the camera can be placed at varying vertical positions
        :param angle_mode: used to specify how pan and tilt angles are assigned to particles. Options include:
         "TOWARDS_TARGET" - the camera is always pointed to within some angular variation (set in target_deviation) of
         the target centroid
         "WALL_LIMITED" - the camera is always pointed within the internal 180 degrees of the surface which it is
         perching on. i.e. [-90, 90]
         "FOV_LIMITED" - the camera is always pointed within the range of angles which the FOV first contacts the
          surface which it is perching on. i.e. [-90+FOV/2, 90-FOV/2]
         "GIMBAL_LIMITED" - the intersection of the FOV_LIMITED range and the camera's specified gimbal limits
         All other opt result in a 0 to 360 value to be specified
        :param target_deviation: if "TOWARDS_TARGET" is selected as the angle mode, this specifies the tilt and pan
         deviations, respectively
        :param min_vertices: The minimum number of vertices of the target polygon which must be visible in order to
         include a point in the perchable_regions search space
        :param mesh_env: whether or not environment geometry is to be represented by mesh (point + face) data. This flag
         has some impact on particle size (spatial placement particle size between 1 and 4 d) and placement, depending
         on the following flags.
        :param surface_number: int, if -1, the optimization will only be run over all possible perching surfaces
         at once. If >=0 and < num_surfaces, then this variable specifies which surface is being optimized over.
        :param map_to_flat_surface: boolean, whether to map particles to the continuous planar approximations of a 
         surface or to map to the discrete faces on the surface 
        :param vary_position_over_face: boolean, if true, particle size increases by 2, and the particle can be placed
         at any point on a specified face. If false, the particle is always place at the face center point.
        :param min_perch_window: 2D array containing the minimum perching window required to successfully land a quad
         on a surface. When landing on horizontal surfaces, the largest of the two entries is used to create a symmetric
         window. When landing on vertical surfaces, the order of values is [horizontal_window, vertical_window]
        :param perch_window_shape: string, either "ellipse" or other; determines whether the erosion structuring
         element is elliptical or rectangular. Defaults to "rectangle".
        :param angle_threshold: float, When floor planes are being merged into a single target region, this parameter
         determines the maximum possible angular deviation between the planes
        :param dist_threshold: float, When floor planes are being merged into a single target region, this parameter
         determines the maximum possible normal distance between each plane's centroid
        :param erosion_raster_density: float, When eroding surfaces to ensure a minimum perch window is satisfied, this
         value sets the number of pixels per meter to be used when rasterizing the surface
        :param min_obstacle_radius: DEPRECATED float, when doing a proximity search to eliminate perching regions which are too
         close to nearby obstacles, this parameter sets the search radius. A radius search is conducted for each point
         on the target surface during the mesh pre-processing phase.
        :param nearest_neighbor_restriction: boolean, whether or not to run the perch region reduction based on
         enforcing a minimum radius to obstacle points. Defaults to true, but can be computationally expensive for large
         or complex mesh environments.
        :param target_volume_height: float, determines the height of the discretized target volume, as measured from the
         floor surface's centroid.
        :param world_frame: 3x3 rotation matrix indicating the orientation of the meshes in the world frame. Applying
         this rotation matrix to the mesh (R*x) files is expected to return point coordinates in Fwd-Left-Up convention.
        :param min_recovery_height: float - the minimum height required for a drone to successfully recover from a
         failed perching attempt. Used to set the minimum wall height for vertical perching surfaces. Only considered if
         set greater than 0. If min_recovery_height <= 0, no search space restriction occurs
        :param minimum_score_deviation: The minimum standard deviation in particle scores to continue searching
        :param minimum_particle_deviation: The minimum standard deviation in particle position to continue searching
        :param inside_out_search: boolean, if true, limits the search area to only regions that are visible from the
         target centroid
        """
        self.variable_pan = variable_pan
        self.variable_tilt = variable_tilt
        self.variable_height = variable_height
        self.variable_zoom = variable_zoom
        self.perch_on_ceiling = perch_on_ceiling
        self.perch_on_walls = perch_on_walls
        self.land_on_floor = land_on_floor
        self.perch_on_intermediate_angles = perch_on_intermediate_angles
        self.minimum_particle_deviation = minimum_particle_deviation
        self.minimum_score_deviation = minimum_score_deviation

        # used in convert_particle_pose() (and potentially to reduce perch area more
        self.angle_mode = angle_mode
        self.target_deviation = target_deviation

        # minimum # of vertices un-occluded from point to be considered a viable perching location
        self.min_vertices = min_vertices

        # new params for mesh environments
        self.mesh_env = mesh_env
        
        self.surface_number = surface_number
        self.map_to_flat_surface = map_to_flat_surface
        self.vary_position_over_face = vary_position_over_face
        
        self.angle_threshold = angle_threshold
        self.dist_threshold = dist_threshold
        self.min_room_height = min_room_height
        self.target_volume_height = target_volume_height

        self.min_perch_window = min_perch_window
        self.min_recovery_height = min_recovery_height
        self.perch_window_shape = perch_window_shape
        self.erosion_raster_density = erosion_raster_density

        self.min_obstacle_radius = min_obstacle_radius
        self.nearest_neighbor_restriction = nearest_neighbor_restriction
        self.noise_resistant_sample_size = noise_resistant_sample_size
        self.noise_resistant_particles = noise_resistant_particles

        self.world_frame = world_frame
        self.gravity_direction = world_frame[2, :]

        self.inside_out_search = inside_out_search

        self.continue_searching = True
        self.max_stagnant_loop_count = 5  # TODO: if this works well, add to opt.ini
        self.stagnant_loops = 0

        self.log_performance = False
        self.search_time = None
        self.fitness_deviation = None
        self.best_fitness = None
        self.pts_searched = None
        self.iteration = 0
        self.data_index = 0

        self.n_points = n_points

    def get_particle_size(self):
        """
        Based on selected opt, this function determines the dimensionality per camera of the problem
        :return: num_dims
        """
        num_dims = 0  # TODO: check other cases...

        variables = [self.variable_tilt, self.variable_pan, self.variable_zoom]
        for var in variables:
            if var:
                num_dims += 1

        if self.mesh_env:
            if self.surface_number == -1:
                # optimizing over all surfaces, therefore one var is needed as the surface selector
                num_dims += 1

            if self.map_to_flat_surface:
                num_dims += 2  # no more need for face selection (-1) but need 2 to map over surface (+2) = (+1)
            else:
                num_dims += 1  # add one dimension as the face selector
                if self.vary_position_over_face:
                    # particle can be placed randomly on face; need 2 params for this. only considered when not mapping flat
                    num_dims += 2

        else:  # for non-mesh environments, it is a bit trickier to place cameras on the cieling as walls and ceiling
            # were treated differently.. These options are deprecated, however.
            if self.variable_height:
                num_dims += 1
            if self.perch_on_ceiling:
                num_dims += 1
            if self.perch_on_ceiling and not self.variable_height:
                num_dims += 1

        return num_dims
