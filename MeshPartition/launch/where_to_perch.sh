#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "Invalid number of arguments. Usage: ./where_to_perch.sh [cleaned_mesh_path]"
    exit 0
fi

# launch mesh cleanup, partition, and CPO (all in one file)
echo "Starting Where To Perch system..."
roslaunch /home/simon/catkin_ws/src/perch_placement/launch/where_to_perch.launch input_mesh:=$1  # output_mesh:= segmented_mesh_clusters:=

