#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "Invalid number of arguments. Usage: ./reconstruction_and_where_to_perch.sh [open3d_config_file_path] [open3d_reconstructed_mesh_path] [clean_mesh_path] [cluster_mesh_path] [top_cluster_prototype]"
    exit 0
fi

roslaunch /home/simon/catkin_ws/src/perch_placement/launch/where_to_perch_start_nodes.launch 

# change to cuda directory... 
echo "Running mesh reconstruction..."
cd ~/Documents/Open3D-Cuda/Open3D/examples/Cuda/ReconstructionSystem
# run dense reconstruction using specified config file
~/Documents/Open3D-Cuda/Open3D/build/bin/examples/RunSystem "$1"

# change to mesh cleanup directory
cd ~/catkin_ws/src/mesh_partition/launch
echo "Running mesh cleanup..."
#meshlabserver -i "$2" -o ~/catkin_ws/src/mesh_partition/models/reduced_mesh.ply -s quadric.mlx -om vc fc
#meshlabserver -i ~/catkin_ws/src/mesh_partition/models/reduced_mesh.ply -o ~/catkin_ws/src/mesh_partition/models/cleaned_mesh.ply -s islands.mlx -om vc fc

meshlabserver -i "$2" -o $3 -s remove_duplicates_islands_smooth_and_decimate.mlx -om

# launch mesh cleanup, partition, and CPO (all in one file)
echo "Starting Where To Perch system..."
roslaunch /home/simon/catkin_ws/src/perch_placement/launch/where_to_perch_publisher.launch input_mesh:=~/catkin_ws/src/mesh_partition/models/cleaned_mesh.ply  # output_mesh:= segmented_mesh_clusters:=


# launch mesh cleanup, partition, and CPO (all in one file)
echo "Starting Where To Perch system..."
roslaunch /home/simon/catkin_ws/src/perch_placement/launch/where_to_perch_publisher.launch input_mesh:=$3  output_mesh:=$4 segmented_mesh_clusters:=$5

