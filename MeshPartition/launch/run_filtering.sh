#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "Invalid number of arguments. Usage: ./run_filtering.sh [model_folder_path] [in_name] [out_name] [filter_path]"
    exit 0
fi


x=1
while [ $x -le 5 ]
do

	x=$(( $x + 1 ))

	echo "Running mesh cleanup for: $2, [$counter of 10]"

	ts=$(date +%s%N)
	
	meshlabserver -i "$1$2.ply" -o "$1$2_red.ply" -s $4quadric.mlx -om vc fc
	meshlabserver -i "$1$2_red.ply" -o "$1$3.ply" -s $4islands.mlx -om vc fc

	tt=$((($(date +%s%N) - $ts)/1000))
	echo "Time taken: $tt microseconds"

done

