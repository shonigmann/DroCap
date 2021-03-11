#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "Invalid number of arguments. Usage: ./where_to_perch.sh [input_mesh_path] [cleaned_mesh_path] [clustered_mesh_path] [top_cluster_path_prototype]"
    exit 0
fi

echo "Starting WTP nodes"

# roslaunch /home/simon/catkin_ws/src/perch_placement/launch/where_to_perch_start_nodes.launch 

echo "Running mesh cleanup..."
#meshlabserver -i "$2" -o ~/catkin_ws/src/mesh_partition/models/reduced_mesh.ply -s quadric.mlx -om vc fc
#meshlabserver -i ~/catkin_ws/src/mesh_partition/models/reduced_mesh.ply -o ~/catkin_ws/src/mesh_partition/models/cleaned_mesh.ply -s islands.mlx -om vc fc

meshlabserver -i "$1" -o $2 -s remove_duplicates_islands_smooth_and_decimate2.mlx -om


# launch mesh cleanup, partition, and CPO (all in one file)
echo "Publishing cleaned mesh..."
roslaunch /home/simon/catkin_ws/src/perch_placement/launch/where_to_perch.launch input_mesh:=$2  output_mesh:=$3 segmented_mesh_clusters:=$4

