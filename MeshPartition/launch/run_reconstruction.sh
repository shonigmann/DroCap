#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "Invalid number of arguments. Usage: ./run_reconstruction.sh [open3d_config_file_path] [open3d_reconstructed_mesh_path]"
    exit 0
fi

# change to cuda directory... 
cd ~/Documents/Open3D-Cuda/Open3D/examples/Cuda/ReconstructionSystem
echo "Running mesh reconstruction..."
# run dense reconstruction using specified config file
~/Documents/Open3D-Cuda/Open3D/build/bin/examples/RunSystem "$1"

# change to mesh cleanup directory
echo "Running mesh cleanup..."
cd ~/catkin_ws/src/mesh_partition/launch
meshlabserver -i "$2" -o ~/catkin_ws/src/mesh_partition/models/reduced_mesh.ply -s quadric.mlx -om vc fc
meshlabserver -i ~/catkin_ws/src/mesh_partition/models/reduced_mesh.ply -o ~/catkin_ws/src/mesh_partition/models/cleaned_mesh.ply -s islands.mlx -om vc fc

