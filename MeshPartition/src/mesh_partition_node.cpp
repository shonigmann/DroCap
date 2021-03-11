
// from ROS tutorial
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>

#include <iostream>
#include "partition.h"
#include "tools.h"
#include <chrono>

#include "mesh_partition/MeshEnvironment.h"
#include "mesh_partition/DenseEnvironment.h"

const bool run_post_processing = true;
const bool run_mesh_simplification = true;
const bool output_mesh_face_color = true;
const double cluster_area_threshold = 0.2;

class MeshPartitionNode
{
public:
  MeshPartitionNode(){
    //Topic you want to publish
    // not sure what type to use... either  mesh_partition::Environment or Environment_ or Environment
    pub_ = n_.advertise<mesh_partition::MeshEnvironment>("/part_mesh_env", 10);

    //Topic you want to subscribe
    sub_ = n_.subscribe("/clean_mesh_env", 1, &MeshPartitionNode::callback, this);
  }

  void callback(const mesh_partition::DenseEnvironment& input){
    using namespace std::chrono;
    // TODO: load parameters currently listed as globals (here and in partition.cpp) from configuration file
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    PRINT_CYAN("Mesh Partition:");
    PRINT_CYAN("Input PLY: %s", input.input_mesh.c_str());
    PRINT_CYAN("Target Number of Clusters: %d", input.target_num_clusters);
    PRINT_CYAN("Output PLY: %s", input.output_mesh.c_str());
    PRINT_CYAN("Output Segmented Meshes: %s", input.segmented_mesh_clusters.c_str());

    string ply_fname = input.input_mesh.c_str();
    int target_cluster_num = input.target_num_clusters;
    string out_ply_fname = input.output_mesh.c_str();
    string out_cluster_fname = out_ply_fname.substr(0, out_ply_fname.length() - 4) + "-cluster" + to_string(target_cluster_num) + ".txt";
    string segmented_mesh_clusters = input.segmented_mesh_clusters.c_str();

    Partition partition;
    PRINT_GREEN("Read ply file: %s", ply_fname.c_str());
    if (!partition.readPLY(ply_fname))
    {
        PRINT_RED("ERROR in reading ply file %s", ply_fname.c_str());
    }
    partition.printModelInfo();

    bool flag_read_cluster_file = false;
    auto start = std::chrono::steady_clock::now();
    bool flag_success = true;

    partition.setTargetClusterNum(target_cluster_num);
    PRINT_GREEN("Run mesh partition ...");
    flag_success = partition.runPartitionPipeline();
    partition.doubleCheckClusters();

    if (run_mesh_simplification)
    {
        PRINT_GREEN("Run mesh simplification...");
        partition.runSimplification();
        // partition.doubleCheckClusters();
    }
    PRINT_GREEN("Final cluster number: %d", partition.getCurrentClusterNum());
    auto end = std::chrono::steady_clock::now();
    double delta = std::chrono::duration_cast<chrono::milliseconds>(end - start).count();
    PRINT_RED("Time: %f ms", delta);
    if (flag_success)
    {
        PRINT_GREEN("Write ply file %s", out_ply_fname.c_str());
        partition.writePLY(out_ply_fname);

        partition.doubleCheckClusters();
        PRINT_GREEN("Final cluster number: %d", partition.getCurrentClusterNum());
        partition.updateClusters();
        partition.writeTopPLYs(segmented_mesh_clusters, cluster_area_threshold);

//        PRINT_GREEN("Write cluster file %s", out_cluster_fname.c_str());
        partition.writeClusterFile(out_cluster_fname);
        PRINT_GREEN("ALL DONE.");
    }

    mesh_partition::MeshEnvironment output;
    output.surf_path_prototype = segmented_mesh_clusters;
//    output.target_path_prototype = "";  // TODO
    output.full_env_path = ply_fname;
    output.clustered_env_path = out_ply_fname;
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    PRINT_RED("ELLAPSED TIME: %f", time_span.count());
    pub_.publish(output);
  }

  private:
    ros::NodeHandle n_;
    ros::Publisher pub_;
    ros::Subscriber sub_;

  private:
    bool MeshClean(){
     //TODO:

    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "mesh_partition_node");
  MeshPartitionNode MPNode;

  ros::spin();

  return 0;
}
