/*!
    Mesh simplification code refers this paper:
    Garland, Michael. Quadric-based polygonal surface simplification. PhD thesis, 1999.
*/

#include "partition.h"
#include <stdlib.h>
//#include "../common/tools.h"
#include "tools.h"
#include <random>
#include <chrono>
#include <queue>
//#include <gflags/gflags.h>
#include <fstream>
#include <algorithm>
#include "pca.h"
#include <math.h>

const double kPI = 3.1415926;

//DEFINE_double(point_plane_dis_threshold, 0.2, "");
//DEFINE_double(normal_angle_threshold, 15.0, "");
//DEFINE_double(center_normal_angle_threshold, 70.0, "");
//DEFINE_double(energy_increase_threshold, 0.1, "");
//DEFINE_double(island_cluster_border_ratio, 0.8, "");
//DEFINE_double(simplification_border_edge_ratio, 0.05, "");
//DEFINE_int32(swapping_loop_num, 300, "0 or a negative value means no swapping at all");
//DEFINE_int32(smallest_connected_component_size, 20, "#faces in smallest connected components");
//DEFINE_int32(smallest_inner_edge_number, 10, "");
//DEFINE_bool(run_post_processing, true, "");
//DEFINE_bool(run_mesh_simplification, true, "");
//DEFINE_bool(output_mesh_face_color, true, "false for no face color in the output PLY mesh");
const double point_plane_dis_threshold = 0.2;
const double normal_angle_threshold = 15.0;
const double center_normal_angle_threshold = 70.0;
const double energy_increase_threshold = 0.1;
const double island_cluster_border_ratio = 0.8;
const double simplification_border_edge_ratio = 0.05;
const int swapping_loop_num = 300;
const int smallest_connected_component_size = 20;
const int smallest_inner_edge_number=10;
const bool run_post_processing = true;
const bool run_mesh_simplification = true;
const bool output_mesh_face_color = true;

const double lab_color_diff_threshold = 10000000.0;  //20.0; // TODO
const double yuv_color_diff_threshold = 20.0; // TODO

Partition::Partition()
{
    vertex_num_ = face_num_ = 0;
    init_cluster_num_ = curr_cluster_num_ = target_cluster_num_ = 0;
    flag_read_cluster_file_ = false;
    flag_new_mesh_ = false;
}

Partition::~Partition()
{
    cout << "Release edges in heap ... " << endl;
    releaseHeap();
}

void Partition::releaseHeap()
{
    // Each edge pointer is stored in the global edge list twice and the heap once,
    // but it can only be deleted once. Here simply delete each edge in the heap.
    while (heap_.size() > 0)
    {
        Edge* edge = (Edge*)heap_.extract();
        delete edge;
        edge = nullptr;
    }
}

//! Read PLY model
/*!
    This function supports PLY model with:
    - both binary and ASCII format;
    - 3 RGB channel vertex color and face color;
    - 3-dim vertex normal (even though doesn't save it);
    - 1-dim vertex quality (even though doesn't save it);
*/
bool Partition::readPLY(const std::string& filename)
{
    FILE* fin;
    if (!(fin = fopen(filename.c_str(), "rb")))
    {
        cout << "ERROR: Unable to open file" << filename << endl;
        return false;
    };

    /************************************************************************/
    /* Read headers */

    // Mode for vertex and face type
    // 1 for vertex (no faces), 2 for vertex and faces,
    // 3 for vertex, vertex colors (no faces), 4 for vertex, vertex colors and faces
    int vertex_mode = 1;
    int ply_mode = 0;  // 0 for ascii, 1 for little-endian (binary)
    size_t vertex_color_channel = 0, face_color_channel = 0;
    size_t vertex_quality_dim = 0;  // vertex quality, any kind of float value per vertex defined by user
    size_t vertex_normal_dim = 0;   // vertex normal dimension
    char seps[] = " ,\t\n\r ";      // separators
    seps[5] = 10;
    int property_num = 0;
    char line[1024];
    while (true)
    {
        if (fgets(line, 1024, fin) == NULL)
            continue;
        char* token = strtok(line, seps);
        if (!strcmp(token, "end_header"))
            break;
        else if (!strcmp(token, "format"))
        {
            token = strtok(NULL, seps);
            if (!strcmp(token, "ascii"))
                ply_mode = 0;
            else if (!strcmp(token, "binary_little_endian"))
                ply_mode = 1;
            else
            {
                cout << "ERROR in Reading PLY model: can not read this type of PLY model: " << string(token) << endl;
                return false;
            }
        }
        else if (!strcmp(token, "element"))
        {
            token = strtok(NULL, seps);
            if (!strcmp(token, "vertex"))
            {
                // vertex count
                token = strtok(NULL, seps);
                sscanf(token, "%d", &vertex_num_);
                vertex_mode = 1;
            }
            else if (!strcmp(token, "face"))
            {
                // Face count
                token = strtok(NULL, seps);
                sscanf(token, "%d", &face_num_);
                vertex_mode++;  // mode with faces is 1 larger than mode without faces
            }
        }
        else if (!strcmp(token, "property"))
        {
            if (vertex_mode % 2 == 1)
            {
                if (property_num >= 3)  // skip property x,y,z
                {
                    token = strtok(NULL, seps);
                    if (!strcmp(token, "uchar"))  // color
                    {
                        token = strtok(NULL, seps);
                        if (!strcmp(token, "red") || !strcmp(token, "green") || !strcmp(token, "blue") ||
                            !strcmp(token, "alpha"))
                            vertex_color_channel++;
                        else
                        {
                            cout << "ERROR in Reading PLY model: cannot read this vertex color type -- " << string(token)
                                 << endl;
                            return false;
                        }
                    }
                    else if (!strcmp(token, "float"))  // vertex quality data
                    {
                        // Currently just count it and skip
                        token = strtok(NULL, seps);
                        if (!strcmp(token, "nx") || !strcmp(token, "ny") || !strcmp(token, "nz"))
                            vertex_normal_dim++;
                        else
                            vertex_quality_dim++;
                    }
                }
                property_num++;
            }
            else if (vertex_mode % 2 == 0)
            {
                token = strtok(NULL, seps);
                bool face_flag = false;
                if (!strcmp(token, "list"))  // face component
                {
                    token = strtok(NULL, seps);
                    if (!strcmp(token, "uint8") || !strcmp(token, "uchar"))
                    {
                        token = strtok(NULL, seps);
                        if (!strcmp(token, "int") || !strcmp(token, "int32"))
                            face_flag = true;
                    }
                    if (!face_flag)
                    {
                        cout << "ERROR in Reading PLY model: the type of 'number of face indices' is not 'unsigned char', or "
                                "the type of 'vertex_index' is not 'int'."
                             << endl;
                        return false;
                    }
                }
                else if (!strcmp(token, "uchar"))  // face color
                {
                    token = strtok(NULL, seps);
                    if (!strcmp(token, "red") || !strcmp(token, "green") || !strcmp(token, "blue") || !strcmp(token, "alpha"))
                        face_color_channel++;
                    else
                    {
                        cout << "ERROR in Reading PLY model: cannot read this face color type -- " << string(token) << endl;
                        return false;
                    }
                }
            }
        }
    }
    if (vertex_color_channel != 0 && vertex_color_channel != 3 && vertex_color_channel != 4)
    {
        cout << "ERROR: Vertex color channel is " << vertex_color_channel << " but it has to be 0, 3, or 4." << endl;
        return false;
    }
    if (face_color_channel != 0 && face_color_channel != 3 && face_color_channel != 4)
    {
        cout << "ERROR: Face color channel is " << face_color_channel << " but it has to be 0, 3, or 4." << endl;
        return false;
    }
    if (vertex_normal_dim != 0 && vertex_normal_dim != 3)
    {
        cout << "ERROR: Vertex normal dimension is " << vertex_normal_dim << " but it has to be 0 or 3." << endl;
        return false;
    }

    /************************************************************************/
    /* Read vertices and faces */
    vertices_.reserve(vertex_num_);
    faces_.reserve(face_num_);
    if (ply_mode == 1)  // binary mode
    {
        for (int i = 0; i < vertex_num_; i++)
        {
            // Vertex data order must be:
            // coordinates -> normal -> color -> others (qualities, radius, curvatures, etc)
            size_t haveread = 0;
            Vertex vtx;
            float vert[3];
            if ((haveread = fread(vert, sizeof(float), 3, fin)) != 3)
            {
                cout << "ERROR in reading PLY vertices in position " << ftell(fin) << endl;
                return false;
            }
            if (vertex_normal_dim)
            {
                float nor[3];
                if ((haveread = fread(nor, sizeof(float), vertex_normal_dim, fin)) != vertex_normal_dim)
                {
                    cout << "ERROR in reading PLY vertex normals in position " << ftell(fin) << endl;
                    return false;
                }
                // NOTE: currently we just abandon the vertex normal
            }
            if (vertex_color_channel)
            {
                unsigned char color[4];
                if ((haveread = fread(color, sizeof(unsigned char), vertex_color_channel, fin)) != vertex_color_channel)
                {
                    cout << "ERROR in reading PLY vertex colors in position " << ftell(fin) << endl;
                    return false;
                } else{
                    vtx.color = rgb2lab(Vector3d((double)color[0], (double)color[1], (double)color[2])); // COLOR CONVERSION HERE
                    // PRINT_RED("Vertex %d Color: [%f, %f, %f]", i, vtx.color[0], vtx.color[1], vtx.color[2]);
                }
            }
            if (vertex_quality_dim)
            {
                float qual[3];  // Currently we just abandon the vertex quality data
                if ((haveread = fread(qual, sizeof(float), vertex_quality_dim, fin)) != vertex_quality_dim)
                {
                    cout << "ERROR in reading PLY vertex qualities in position " << ftell(fin) << endl;
                    return false;
                }
                // NOTE: currently abandon the vertex quality data
            }
            // You can still read other types of data here

            // Save vertex
            vtx.pt = Vector3d(vert[0], vert[1], vert[2]);
            vertices_.push_back(vtx);

        }

        // Face data
        for (int i = 0; i < face_num_; ++i)
        {
            unsigned char channel_num;
            size_t haveread = fread(&channel_num, 1, 1, fin);  // channel number for each face
            Face fa;
            // Face data order must be: face indices -> face color -> others
            if ((haveread = fread(fa.indices, sizeof(int), 3, fin)) != 3)  // currently only support triangular face
            {
                cout << "ERROR in reading PLY face indices: reader position " << ftell(fin) << endl;
                return false;
            }
            if (face_color_channel)
            {
                unsigned char color[4];
                if ((haveread = fread(color, sizeof(unsigned char), face_color_channel, fin)) != face_color_channel)
                {
                    cout << "ERROR in reading PLY face colors: reader position " << ftell(fin) << endl;
                    return false;
                }
                // NOTE: currently we only read face color data but don't save them, since in this program, face color
                // is the same as the corresponding cluster color which is stored and read in the cluster file.
                fa.color = rgb2lab(Vector3d((double)color[0], (double)color[1], (double)color[2])); // COLOR CONVERSION HERE

            }else{
                double c0 = (vertices_[fa.indices[0]].color[0] + vertices_[fa.indices[1]].color[0] + vertices_[fa.indices[2]].color[0])/3.0;
                double c1 = (vertices_[fa.indices[0]].color[1] + vertices_[fa.indices[1]].color[1] + vertices_[fa.indices[2]].color[1])/3.0;
                double c2 = (vertices_[fa.indices[0]].color[2] + vertices_[fa.indices[1]].color[2] + vertices_[fa.indices[2]].color[2])/3.0;
                fa.color = rgb2lab(Vector3d(c0, c1, c2)); // COLOR CONVERSION HERE
            }
            // Read other types of face data here.

            //
            faces_.push_back(fa);
        }
    }
    else  // ASCII mode (face reader is still unfinished)
    {
        // Read vertices (with C functions)
        for (int i = 0; i < vertex_num_; i++)
        {
            // Vertex data order must be:
            // coordinates -> normal -> color -> others (qualities, radius, curvatures, etc)
            if (fgets(line, 1024, fin) == NULL)
                continue;
            char* token = strtok(line, seps);
            // Read 3D point
            Vertex vtx;
            float vert[3];
            for (size_t j = 0; j < 3; ++j)
            {
                token = strtok(NULL, seps);
                sscanf(token, "%f", &(vert[j]));
            }
            // Read vertex normal
            if (vertex_normal_dim)
            {
                float nor[3];
                for (size_t j = 0; j < vertex_normal_dim; ++j)
                {
                    token = strtok(NULL, seps);
                    sscanf(token, "%f", &(nor[j]));
                }
                // Currently we just abandon the vertex normal data
            }
            if (vertex_color_channel)
            {
                unsigned char color[4];
                for (size_t j = 0; j < vertex_quality_dim; ++j)
                {
                    token = strtok(NULL, seps);
                    sscanf(token, "%c", &(color[j]));
                }
                // Abandon vertex color
            }
            if (vertex_quality_dim)
            {
                float qual;
                for (size_t j = 0; j < vertex_quality_dim; ++j)
                {
                    token = strtok(NULL, seps);
                    sscanf(token, "%f", &qual);
                }
                // Currently abandon the vertex quality data
            }
            // Read other types of vertex data here.

            // Save vertex
            vtx.pt = Vector3d(vert[0], vert[1], vert[2]);
            vertices_.push_back(vtx);
        }
        // Read Faces
        for (int i = 0; i < face_num_; i++)
        {
            if (fgets(line, 1024, fin) == NULL)
                continue;
            char* token = strtok(line, seps);
            token = strtok(NULL, seps);
            Face fa;
            for (int j = 0; j < 3; ++j)
            {
                token = strtok(NULL, seps);
                sscanf(token, "%d", &(fa.indices[j]));
            }
            if (face_color_channel)
            {
                unsigned char color[4];
                for (int j = 0; j < 4; ++j)
                {
                    token = strtok(NULL, seps);
                    sscanf(token, "%c", &(color[j]));
                }
                // Currently abandon the face color data
            }
            faces_.push_back(fa);
        }
    }

    // Just in case some vertices or faces are not read correctly
    face_num_ = static_cast<int>(faces_.size());
    vertex_num_ = static_cast<int>(vertices_.size());
    return true;
}

//! Write PLY file
bool Partition::writePLY(const std::string& filename)
{
    // Write PLY
    FILE* fout = NULL;
    fout = fopen(filename.c_str(), "wb");  // write in binary mode
    if (fout == NULL)
    {
        cout << "Unable to create file " << filename << endl;
        return false;
    }
    // Write headers
    fprintf(fout, "ply\n");
    fprintf(fout, "format binary_little_endian 1.0\n");
    int vnum = flag_new_mesh_ ? new_vertex_num_ : vertex_num_;
    fprintf(fout, "element vertex %d\n", vnum);
    fprintf(fout, "property float x\n");
    fprintf(fout, "property float y\n");
    fprintf(fout, "property float z\n");
    /*
      if (output_mesh_face_color)
      {
          fprintf(fout, "property uchar red\n");  // face color
          fprintf(fout, "property uchar green\n");
          fprintf(fout, "property uchar blue\n");
          fprintf(fout, "property uchar alpha\n");
      }
      */

    int fnum = flag_new_mesh_ ? new_face_num_ : face_num_;
    fprintf(fout, "element face %d\n", fnum);
    fprintf(fout, "property list uchar int vertex_indices\n");
    if (output_mesh_face_color)
    {
        fprintf(fout, "property uchar red\n");  // face color
        fprintf(fout, "property uchar green\n");
        fprintf(fout, "property uchar blue\n");
        fprintf(fout, "property uchar alpha\n");
    }
    fprintf(fout, "end_header\n");
    float pt3[3];
    unsigned char kFaceVtxNum = 3;
    unsigned char rgba[4] = {255, 255, 255, 255};
    unsigned char rgb[3] = {255, 255, 255};
    for (int i = 0; i != vertex_num_; ++i)
    {
        if (!flag_new_mesh_ || vertices_[i].is_valid)
        {
            for (int j = 0; j < 3; ++j)
                pt3[j] = float(vertices_[i].pt[j]);
            fwrite(pt3, sizeof(float), 3, fout);
        }
    }
    for (int i = 0; i != face_num_; ++i)
    {
        if (!flag_new_mesh_ || faces_[i].is_valid)
        {
            fwrite(&kFaceVtxNum, sizeof(unsigned char), 1, fout);
            if (flag_new_mesh_)
            {
                for (int j = 0; j < 3; ++j)
                {
                    int vidx = vidx_old2new_[faces_[i].indices[j]];
                    fwrite(&vidx, sizeof(int), 1, fout);
                }
            }
            else
                fwrite(faces_[i].indices, sizeof(int), 3, fout);
            if (output_mesh_face_color)
            {
                int cidx = faces_[i].cluster_id;
                if (cidx == -1)
                {
                    PRINT_RED("ERROR: face %d doesn't belong to any cluster. This shouldn't happen.", i);
                    // Will use default color (0,0,0)
                }
                else
                {
                    for (int j = 0; j < 3; ++j)
                        rgba[j] = static_cast<unsigned char>(clusters_[cidx].color[j] * 255);
                }
                fwrite(rgba, sizeof(unsigned char), 4, fout);
            }
        }
    }
    fclose(fout);
    return true;
}

//! Write PLY file with only top N largest clusters
bool Partition::writePLY(const std::string& filename, double min_area)
{
    // Write PLY
    FILE* fout = NULL;
    fout = fopen(filename.c_str(), "wb");  // write in binary mode
    if (fout == NULL)
    {
        cout << "Unable to create file " << filename << endl;
        return false;
    }
    // Write headers
    fprintf(fout, "ply\n");
    fprintf(fout, "format binary_little_endian 1.0\n");
    int vnum = flag_new_mesh_ ? new_vertex_num_ : vertex_num_;
    fprintf(fout, "element vertex %d\n", vnum);
    fprintf(fout, "property float x\n");
    fprintf(fout, "property float y\n");
    fprintf(fout, "property float z\n");

    int fnum = flag_new_mesh_ ? new_face_num_ : face_num_;
    // TODO: change -----------------------------------------------------------------------------
    fprintf(fout, "element face %d\n", fnum);
    // TODO: change -----------------------------------------------------------------------------
    fprintf(fout, "property list uchar int vertex_indices\n");
    if (output_mesh_face_color)
    {
        fprintf(fout, "property uchar red\n");  // face color
        fprintf(fout, "property uchar green\n");
        fprintf(fout, "property uchar blue\n");
        fprintf(fout, "property uchar alpha\n");
    }
    fprintf(fout, "end_header\n");
    float pt3[3];
    unsigned char kFaceVtxNum = 3;
    unsigned char rgba[4] = {255, 255, 255, 255};
    unsigned char rgb[3] = {255, 255, 255};
    for (int i = 0; i != vertex_num_; ++i)
    {
        if (!flag_new_mesh_ || vertices_[i].is_valid)
        {
            for (int j = 0; j < 3; ++j)
                pt3[j] = float(vertices_[i].pt[j]);
            fwrite(pt3, sizeof(float), 3, fout);
        }
    }

    // TODO: change -----------------------------------------------------------------------------
    for (int i = 0; i != face_num_; ++i)
    {
        if (!flag_new_mesh_ || faces_[i].is_valid)
        {
            fwrite(&kFaceVtxNum, sizeof(unsigned char), 1, fout);
            if (flag_new_mesh_)
            {
                for (int j = 0; j < 3; ++j)
                {
                    int vidx = vidx_old2new_[faces_[i].indices[j]];
                    fwrite(&vidx, sizeof(int), 1, fout);
                }
            }
            else
                fwrite(faces_[i].indices, sizeof(int), 3, fout);
            if (output_mesh_face_color)
            {
                int cidx = faces_[i].cluster_id;
                if (cidx == -1)
                {
                    PRINT_RED("ERROR: face %d doesn't belong to any cluster. This shouldn't happen.", i);
                    // Will use default color (0,0,0)
                }
                else
                {
                    if(clusters_[cidx].cov.area_ < 1)
                    { //trying with area limit instead of number of regions
                        for (int j = 0; j < 3; ++j)
                            rgba[j] = static_cast<unsigned char>(clusters_[cidx].color[j] * 0);
                    }
                    else{
                        for (int j = 0; j < 3; ++j)
                            rgba[j] = static_cast<unsigned char>(clusters_[cidx].color[j] * 255);

                        //PRINT_MAGENTA("(new) Cluster: %d (or %d); Original ID: %d", c, cidx, clusters_[c].original_id);
                    }
                }
                fwrite(rgba, sizeof(unsigned char), 4, fout);
            }
        }
    }
    fclose(fout);
    return true;
}

//! Write a PLY file for each cluster that meets specified criteria
bool Partition::writeTopPLYs(const std::string& basefilename, double area_threshold, Vector3d gravity_direction)//=Vector3d(0,1,0))
{
//    PRINT_RED("test: %s", basefilename.c_str());
    int file_name_len = basefilename.length();
    string basename = basefilename.c_str();
    if(file_name_len >4) {
        if (basefilename.substr(file_name_len - 4, 4).find(".ply")!= std::string::npos ||
            basefilename.substr(file_name_len - 4, 4).find(".PLY")!= std::string::npos) {
            basename = (basefilename.substr(0, file_name_len - 4)).c_str();
//            PRINT_MAGENTA("HERE");
        }
    }

//    PRINT_RED("test: %s", basename.c_str());

//  ofstream plane_text;
//  plane_text.open("planar_reconstruction_test.txt");  // write in binary mode

    int cluster_count = 0;
    // Loop through each cluster and create a PLY file for it
    for(int c = 0; c < clusters_.size(); c++)
    {
        // TODO: figure out best way to bring obstacles into placement optimziation
        int vnum = cluster_vert_num[c].size();
        int fnum = cluster_face_num[c].size();

        // preliminary filter
        if(clusters_[c].cov.area_ < area_threshold || vnum <= 0) //only write file for regions that meet area criteria
            continue;
        //----------------------------------------------------------------
        // Run PCA on cluster components
        Eigen::MatrixXd pca_data_matrix(vnum,3);

        for(int i=0; i < vnum; i++){
            int vidx = cluster_vert_new2old[c][i];
            for(int j=0; j<3; j++){
                pca_data_matrix(i,j) = vertices_[vidx].pt[j];
            }
        }

        try
        {
            pca_t<double> pca;
            pca.set_input(pca_data_matrix);
            pca.compute();

            //reduced reconstruction....
            const auto& T_L = pca.get_centered_matrix() * pca.get_eigen_vectors().leftCols(2);
            const auto& X_L = T_L * pca.get_eigen_vectors().leftCols(2).transpose();
            const auto& X_hat = X_L + pca.get_mean(); // X_Hat is a purely planar reconstruction of the region
            auto error = (pca_data_matrix - X_hat).norm();

            auto max_T_1 = T_L.leftCols(1).maxCoeff();
            auto min_T_1 = T_L.leftCols(1).minCoeff();
            auto max_T_2 = T_L.rightCols(1).maxCoeff();
            auto min_T_2 = T_L.rightCols(1).minCoeff();

            auto range_T_1 = max_T_1-min_T_1;
            auto range_T_2 = max_T_2-min_T_2;

            const std::string& filename = basename + std::to_string(cluster_count) + ".ply";
            cluster_count++;  //keep track of how many clusters meet requirements to update file name.

            // Write PLY
            //----------------------------------------------------------------
            FILE* fout = NULL;
            fout = fopen(filename.c_str(), "wb");  // write in binary mode
//            PRINT_BLUE("WRITING TO: %s", filename.c_str());
            if (fout == NULL) // || fout_hat == NULL)
            {
                cout << "Unable to create file " << filename << endl;
                return false;
            }

            // Write headers
            fprintf(fout, "ply\n");
            fprintf(fout, "format binary_little_endian 1.0\n");
            fprintf(fout, "element vertex %d\n", vnum);
            fprintf(fout, "property float x\n");
            fprintf(fout, "property float y\n");
            fprintf(fout, "property float z\n");
            fprintf(fout, "element face %d\n", fnum);
            fprintf(fout, "property list uchar int vertex_indices\n");
            if (output_mesh_face_color)
            {
                fprintf(fout, "property uchar red\n");  // face color
                fprintf(fout, "property uchar green\n");
                fprintf(fout, "property uchar blue\n");
                fprintf(fout, "property uchar alpha\n");
            }
            fprintf(fout, "end_header\n");


            float pt3[3];
            unsigned char kFaceVtxNum = 3;
            unsigned char rgba[4] = {255, 255, 255, 255};
            unsigned char rgb[3] = {255, 255, 255};

            // only write relevant vertices
            int vcount = 0;
            for(int i=0; i < vnum; i++){
                int vidx = cluster_vert_new2old[c][i];
                if (vertices_[vidx].is_valid)
                {
                    for (int j = 0; j < 3; ++j)
                        pt3[j] = float(vertices_[vidx].pt[j]);
                    fwrite(pt3, sizeof(float), 3, fout);

                    vcount++;
                }
            }
//            PRINT_YELLOW("NUM VERTICES: %d", vcount); // for debugging

            //Add Faces
            int fcount = 0;

            for (int i : cluster_face_num[c])
            {
                if(faces_[i].is_valid)
                {
                    fcount++;
                    fwrite(&kFaceVtxNum, sizeof(unsigned char), 1, fout);
//          fwrite(&kFaceVtxNum, sizeof(unsigned char), 1, fout_hat);
                    for(int j = 0; j < 3; j++){
                        int vidx = cluster_vert_old2new[c][faces_[i].indices[j]];
                        fwrite(&vidx, sizeof(int), 1, fout); //TODO: check validity of & here
                    }

                    if (output_mesh_face_color)
                    {
                        int cidx = faces_[i].cluster_id;
                        if (cidx == -1)
                        {
                            PRINT_RED("ERROR: face doesn't belong to any cluster. This shouldn't happen.");
                            // Will use default color (0,0,0)
                        }
                        else
                        {
                            for (int j = 0; j < 3; ++j)
                                rgba[j] = static_cast<unsigned char>(clusters_[cidx].color[j] * 255);
                        }
                        fwrite(rgba, sizeof(unsigned char), 4, fout);
                    }
                }
            }
//            PRINT_YELLOW("NUM FACES: %d", fcount);

            fclose(fout);
//      fclose(fout_hat);

        }catch(const std::exception& e)
        {
            PRINT_RED("PCA ERROR:");
            std::cout << e.what();
        }
    }
//  plane_text.close();

    return true;
}


//! Write cluster file in binary
void Partition::writeClusterFile(const std::string& filename)
{
    ofstream writeout("test-cluster.txt", std::ios::trunc);
    writeout << curr_cluster_num_ << endl;

    FILE* fout = fopen(filename.c_str(), "wb");

    fwrite(&curr_cluster_num_, sizeof(int), 1, fout);  // #clusters at first
    float color[3];
    int new_cidx = 0, count_faces = 0;
    for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
    {
        if (!isClusterValid(cidx))  // returns true if the list of faces isn't empty
            continue;  // skip one instance of loop
        int num = int(clusters_[cidx].faces.size());
        fwrite(&new_cidx, sizeof(int), 1, fout);  // cluster index
        fwrite(&num, sizeof(int), 1, fout);       // #faces in this cluster
        vector<int> indices(clusters_[cidx].faces.begin(), clusters_[cidx].faces.end());
        fwrite(&indices[0], sizeof(int), num, fout);  // write all face indices at once, this saves time

        // for debug
        writeout << new_cidx << " " << num << ": ";
        for (int fidx : indices)
            writeout << fidx << " ";
        writeout << " Area: " << clusters_[cidx].cov.area_;
        writeout << endl;

        for (int i = 0; i < 3; ++i)
            color[i] = clusters_[cidx].color[i];
        fwrite(&color[0], sizeof(float), 3, fout);  // cluster color
        new_cidx++;
        count_faces += num;
    }
    fclose(fout);
    writeout.close();

    if (!flag_new_mesh_)
    {
        // Sometimes some faces are missing and not in any cluster. This is bad.
        // So here ensure the number of faces in clusters is exactly the same as original face number.
        assert(count_faces == face_num_);
    }
}

//! Read cluster file. Note it can only be used when there are no existing clusters, like call this function
//! just after reading PLY file and haven't run any other partition functions.
bool Partition::readClusterFile(const std::string& filename)
{
    if (!clusters_.empty() || vertex_num_ == 0 || face_num_ == 0)
    {
        cout << "Can ONLY read the cluster file if already reading the mesh and NO existing clusters." << endl;
        return false;
    }
    FILE* fin = fopen(filename.c_str(), "rb");
    if (fin == NULL)
    {
        cout << "ERROR: cannot find cluster file" << filename << endl;
        return false;
    }
    if (fread(&curr_cluster_num_, sizeof(int), 1, fin) != 1)
    {
        cout << "ERROR in reading cluster number in cluster file" << filename << endl;
        return false;
    }
    if (curr_cluster_num_ < 1)
    {
        cout << "ERROR: cluster number is " << curr_cluster_num_ << endl;
        return false;
    }
    clusters_.resize(curr_cluster_num_);
    float color[3];
    int count_faces = 0;
    for (int i = 0; i < curr_cluster_num_; ++i)
    {
        int cidx = -1, cluster_size = -1;
        if (fread(&cidx, sizeof(int), 1, fin) != 1)
            return false;
        if (fread(&cluster_size, sizeof(int), 1, fin) != 1)
            return false;
        assert(cidx >= 0 && cidx < face_num_ && cluster_size >= 0 && cluster_size <= face_num_);
        vector<int> cluster_elems(cluster_size);
        if (fread(&cluster_elems[0], sizeof(int), cluster_size, fin) != size_t(cluster_size))
        {
            cout << "ERROR in reading indices in cluster file " << filename << endl;
            return false;
        }
        count_faces += cluster_size;
        clusters_[i].faces.insert(cluster_elems.begin(), cluster_elems.end());
        assert(clusters_[i].faces.size() == cluster_size);

        // cout << i << " " << cluster_size << ": ";
        // for (int fidx : cluster_elems)
        //     cout << fidx << " ";
        // cout << endl;

        if (fread(&color[0], sizeof(float), 3, fin) != 3)
        {
            cout << "ERROR in reading colors in cluster file " << filename << endl;
            return false;
        }
        for (int j = 0; j < 3; ++j)
            clusters_[i].color[j] = color[j];
    }
    fclose(fin);
    flag_read_cluster_file_ = true;  // a read-cluster flag to skip some steps later
    assert(count_faces == face_num_);
    init_cluster_num_ = curr_cluster_num_;
    return true;
}

bool Partition::runPartitionPipeline()
{
    PRINT_CYAN("Merging neighbor faces.");
    if (!runMerging())
        return false;

    if (swapping_loop_num > 0)
    {
        PRINT_CYAN("(Optional) A further optimization by swapping cluster border faces.");
        runSwapping();
    }
    if (run_post_processing)
    {
        PRINT_CYAN("Post processing: merge neighbor clusters.");
        PRINT_CYAN("#Initial Clusters: %d", curr_cluster_num_);
        runPostProcessing();
        PRINT_CYAN("#Clusters after post processing: %d", curr_cluster_num_);
    }
    createClusterColors();
    return true;
}

bool Partition::runMerging()
{
    initMerging();

    cout << "Merging ..." << endl;
    float progress = 0.0;  // for printing a progress bar
    int cluster_diff = curr_cluster_num_ - target_cluster_num_;
    const int kStep = (cluster_diff < 100) ? 1 : (cluster_diff / 100);
    int count = 0;
    while (curr_cluster_num_ > target_cluster_num_)
    {
        if (count % kStep == 0 || count == cluster_diff - 1)
        {
            progress = (count == cluster_diff - 1) ? 1.0f : static_cast<float>(count) / cluster_diff;
            printProgressBar(progress);
        }
        if (!mergeOnce())
            return false;
        // Special case: sometimes all existing clusters have no neighbors (like floating faces)
        if (heap_.size() == 0)
        {
            PRINT_YELLOW("WARNING: Now heap is empty, but still not reaching the target cluster number. ");
            break;
        }
        count++;
    }
    cout << "Result Cluster Number: " << curr_cluster_num_ << ", Energy: " << getTotalEnergy() << endl;
    return true;
}

void Partition::initMeshConnectivity()
{
    if (!edge_to_face_.empty())
        return;
    cout << "Initialize vertices and faces ... " << endl;
    vector<int> fa(3);
    float progress = 0.0;  // used to print a progress bar
    const int kStep = (face_num_ < 100) ? 1 : (face_num_ / 100);
    for (int fidx = 0; fidx < face_num_; fidx++)
    {
        // Print a progress bar
        if (fidx % kStep == 0 || fidx == face_num_ - 1)
        {
            progress = (fidx == face_num_ - 1) ? 1.0f : static_cast<float>(fidx) / face_num_;
            printProgressBar(progress);
        }

        Face& face = faces_[fidx];
        if (!face.is_valid)
        {
            // Sometimes we remove some faces at first and then recompute mesh connectivity,
            // so it's necessary to skip invalid faces since we dont care about them any more.
            continue;
        }
        // Initialize neighbors of vertices and faces
        for (int i = 0; i < 3; ++i)
            fa[i] = face.indices[i];
        // We use UNDIRECTED edges, so sort the 3 vertices to avoid edge duplication.
        // So each edge now may be included in 1, 2 or even 3 or more (non-manifold mesh) faces.
        std::sort(fa.begin(), fa.end());
        for (int i = 0; i < 3; ++i)
        {
            vertices_[fa[i]].nbr_vertices.insert(fa[(i + 1) % 3]);
            vertices_[fa[i]].nbr_vertices.insert(fa[(i + 2) % 3]);
            vertices_[fa[i]].nbr_faces.insert(fidx);
            long long a = static_cast<long long>((i == 2) ? fa[0] : fa[i]);
            long long b = static_cast<long long>((i == 2) ? fa[i] : fa[i + 1]);
            long long edge = (a << 32) | b;  // fast bit operation
            for (int f : edge_to_face_[edge])
            {
                face.nbr_faces.insert(f);
                faces_[f].nbr_faces.insert(fidx);
            }
            edge_to_face_[edge].push_back(fidx);
        }
    }
}

void Partition::initMerging()
{
    init_cluster_num_ = curr_cluster_num_ = face_num_;
    assert(target_cluster_num_ < init_cluster_num_ && target_cluster_num_ > 0);
    clusters_.resize(init_cluster_num_);
    global_edges_.resize(init_cluster_num_);

    initMeshConnectivity();

    // Initialize face quadratic objects
    for (int fidx = 0; fidx < face_num_; fidx++)
    {
        Face& face = faces_[fidx];
        face.cov = CovObj(vertices_[face.indices[0]].pt, vertices_[face.indices[1]].pt, vertices_[face.indices[2]].pt);
    }

    // Initialize clusters and edges
    cout << "Initialize clusters and edges ... " << endl;
    float progress = 0.0;  // used to print a progress bar
    const int kStep = (init_cluster_num_ < 100) ? 1 : (init_cluster_num_ / 100);
    for (int cidx = 0; cidx < init_cluster_num_; cidx++)
    {
        if (cidx % kStep == 0 || cidx == init_cluster_num_ - 1)
        {
            progress = (cidx == init_cluster_num_ - 1) ? 1.0f : static_cast<float>(cidx) / init_cluster_num_;
            printProgressBar(progress);
        }

        // Initially each face is one single cluster itself
        faces_[cidx].cluster_id = cidx;
        Cluster& cluster = clusters_[cidx];
        cluster.nbr_clusters.insert(faces_[cidx].nbr_faces.begin(), faces_[cidx].nbr_faces.end());
        cluster.cov = faces_[cidx].cov;
        cluster.faces.insert(cidx);
        cluster.avg_color = faces_[cidx].color;

        // Create initial edges between neighbor faces
        for (int nbr : cluster.nbr_clusters)
        {
            if (cidx < nbr)
            {
                Edge* edge = new Edge(cidx, nbr);
                computeEdgeEnergy(edge);
                heap_.insert(edge);
                global_edges_[cidx].push_back(edge);
                global_edges_[nbr].push_back(edge);
            }
        }
    }
}

//! Compute energy of the edge.
/*!
    This function assumes the energy data in each cluster is already the latest, like calling
    `clusters_[cidx].energy = clusters_[cidx].cov.energy()`. This can save some time.
*/
void Partition::computeEdgeEnergy(Edge* edge)
{
    CovObj cov = clusters_[edge->v1].cov;
    cov += clusters_[edge->v2].cov;
    double energy = cov.energy() - clusters_[edge->v1].energy - clusters_[edge->v2].energy;
    edge->heap_key(-energy);  // it's a max heap by default but we need a min heap
}

//! Remove one edge pointer from an Edge list. Note that it doesn't delete/release the pointer.
bool Partition::removeEdgeFromList(Edge* edge, vector<Edge*>& edgelist)
{
    if (edge == nullptr)
        return false;
    bool flag_found_edge = false;
    auto iter = edgelist.begin();
    while (iter != edgelist.end())
    {
        Edge* e = *iter;
        if (e == nullptr || e == edge)
        {
            iter = edgelist.erase(iter);  // iter will be the next position after erasion
            flag_found_edge = true;
        }
        else
        {
            iter++;
        }
    }
    return flag_found_edge;
}

bool Partition::mergeOnce()
{
    Edge* edge = (Edge*)heap_.extract();
    if (!edge)
    {
        cout << endl << "ERROR: No edge exists in the heap!" << endl;
        return false;
    }
    if (isClusterValid(edge->v1) && isClusterValid(edge->v2))
    {
        applyFaceEdgeContraction(edge);
        curr_cluster_num_--;
    }
    else
    {
        cout << "ERROR: This edge does not exist in clusters. Quiting..." << endl;
        return false;
    }
    return true;
}

//! Edge contraction
void Partition::applyFaceEdgeContraction(Edge* edge)
{
    int c1 = edge->v1, c2 = edge->v2;  // two vertices present cluster ids
    mergeClusters(c1, c2);             // merge cluster c2 to c1
    clusters_[c1].cov += clusters_[c2].cov;
    clusters_[c1].energy = clusters_[c1].cov.energy();  // remember to update energy as well
    clusters_[c2].energy = 0;
//    clusters_[c1].avg_color = TODO; // TODO MERGE CLUSTER COLOR HERE

    // NOTE: this function also works but slow, so already computed neighbor clusters in mergeClusters()
    findClusterNeighbors(c1);

    // Remove all old edges next to c1 and c2 from the heap and corresponding edge lists
    for (Edge* e : global_edges_[c1])
    {
        // For each edge (c1, u) or (u, c1), delete this edge and remove it from cluster u's edge list,
        // since cluster c1 and c2's edge list will be cleared later. This is because, each edge pointer is
        // included twice in two clusters, but can only be released once.
        int u = (e->v1 == c1) ? e->v2 : e->v1;
        heap_.remove(e);
        removeEdgeFromList(e, global_edges_[u]);
        delete e;
    }
    for (Edge* e : global_edges_[c2])
    {
        int u = (e->v1 == c2) ? e->v2 : e->v1;
        heap_.remove(e);
        removeEdgeFromList(e, global_edges_[u]);
        delete e;
    }
    global_edges_[c1].clear();
    global_edges_[c2].clear();

    // Add new edges between v1 and all its new neighbors into edge list
    for (int cidx : clusters_[c1].nbr_clusters)
    {
        // if (c1 > cidx)
        //     std::swap(c1, cidx);
        Edge* e = new Edge(c1, cidx);
        computeEdgeEnergy(e);
        heap_.insert(e);
        global_edges_[c1].push_back(e);
        global_edges_[cidx].push_back(e);
    }
}

//! Merge cluster c2 into cluster c1
void Partition::mergeClusters(int c1, int c2)
{
    // Merge all faces from c2 to c1, and update the corresponding cluster index
    for (int fidx : clusters_[c2].faces)
    {
        clusters_[c1].faces.insert(fidx);
//        clusters_[c1].avg_color = TODO; // TODO cluster color
        faces_[fidx].cluster_id = c1;
    }
    clusters_[c2].faces.clear();
}

//! Find neighbor clusters of an input cluster with faces
int Partition::findClusterNeighbors(int cidx, unordered_set<int>& cluster_faces, unordered_set<int>& neighbor_clusters)
{
    neighbor_clusters.clear();
    for (int fidx : cluster_faces)  // brute-force: check all faces in the cluster
    {
        for (int nbr : faces_[fidx].nbr_faces)
        {
            int ncidx = faces_[nbr].cluster_id;
            if (ncidx != cidx)
                neighbor_clusters.insert(ncidx);
        }
    }
    return int(neighbor_clusters.size());
}

//! Overload: find neighbor clusters of an cluster
int Partition::findClusterNeighbors(int cidx)
{
    return findClusterNeighbors(cidx, clusters_[cidx].faces, clusters_[cidx].nbr_clusters);
}

double Partition::getTotalEnergy()
{
    double energy = 0;
    for (int i = 0; i < init_cluster_num_; ++i)
        if (isClusterValid(i))
            energy += clusters_[i].energy;
    return energy;
}

void Partition::createClusterColors()
{
    // srand(time(NULL)); // randomize seed
    for (int i = 0; i < init_cluster_num_; ++i)
        if (isClusterValid(i))  // create a random color
            clusters_[i].color = Vector3f(float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, float(rand()) / RAND_MAX);
}

void Partition::runSwapping()
{
    // Swapping process
    for (int i = 0; i < init_cluster_num_; ++i)
    {
        if (isClusterValid(i))
            last_clusters_in_swap_.insert(i);
    }
    double last_energy = getTotalEnergy();
    double scale = 1e5;  // scale the energy since it is usually too small
    cout << "Energy 0: " << last_energy * scale << " (scaled by " << scale << ")" << endl;
    double curr_energy = 0;
    int iter = 0;
    while (iter++ < swapping_loop_num)
    {
        int count_swap_faces = swapOnce();
        curr_energy = getTotalEnergy();
        bool flag_quit_loop = false;
        if ((last_energy - curr_energy) / last_energy < 1e-10 || count_swap_faces == 0)
            flag_quit_loop = true;
        if (flag_quit_loop || iter % 10 == 0)
            cout << "Energy " << iter << ": " << curr_energy * scale << ", #Swapped faces: " << count_swap_faces << endl;
        if (flag_quit_loop)
            break;
        last_energy = curr_energy;
    }
    // Post process
    processIslandClusters();
    updateCurrentClusterNum();
}

//! Run swapping step for one time, which means swapping some border face to its neighbor cluster once.
//! Return the number of faces got swapped.
int Partition::swapOnce()
{
    clusters_in_swap_.clear();
    for (int cidx : last_clusters_in_swap_)
    {
        clusters_[cidx].faces_to_swap.clear();
        clusters_in_swap_.insert(cidx);
    }
    last_clusters_in_swap_.clear();

    // Find faces to be swapped. Such a face is a cluster border face and
    // must decrease the total energy after swapping to its neighbor cluster.
    // NOTE: parallel computation can accelerate this process, since each face here is independent with
    // the other faces.
    int count_swap_faces = 0;
    for (int cidx : clusters_in_swap_)
    {
        for (int fidx : clusters_[cidx].faces)
        {
            unordered_set<int> visited_clusters;
            double max_delta_energy = 0;
            int max_cidx = -1;
            for (int nidx : faces_[fidx].nbr_faces)
            {
                int ncidx = faces_[nidx].cluster_id;
                // Skip the neighbor clusters visited before
                if (ncidx != cidx && visited_clusters.count(ncidx) == 0)
                {
                    visited_clusters.insert(ncidx);
                    double delta_energy = computeSwapDeltaEnergy(fidx, cidx, ncidx);
                    if (delta_energy > max_delta_energy)
                    {
                        max_cidx = ncidx;
                        max_delta_energy = delta_energy;
                    }
                }
            }
            if (max_cidx != -1)
            {
                SwapFace sf(fidx, cidx, max_cidx);
                clusters_[cidx].faces_to_swap.push_back(sf);
                count_swap_faces++;
                last_clusters_in_swap_.insert(cidx);
                last_clusters_in_swap_.insert(max_cidx);
            }
        }
    }
    // Now swap faces
    for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
    {
        for (SwapFace& sf : clusters_[cidx].faces_to_swap)
        {
            int from = sf.from;
            int to = sf.to;
            int fidx = sf.face_id;
            faces_[fidx].cluster_id = to;
            clusters_[to].cov += faces_[fidx].cov;
            clusters_[from].cov -= faces_[fidx].cov;
            clusters_[from].faces.erase(fidx);
            clusters_[to].faces.insert(fidx);
//
//            // TODO: update cluster average colors due to face swap
//            clusters_[to].avg_color = ;
//            clusters_[from].avg_color = ;
        }
    }
    // Remember to update the energy each time after updating the covariance object
    for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
        if (isClusterValid(cidx))
            clusters_[cidx].energy = clusters_[cidx].cov.energy();
    return count_swap_faces;
}

// Compute delta energy (energy changes) by swapping face 'fidx' from cluster 'from' to a neighbor cluster 'to'
double Partition::computeSwapDeltaEnergy(int fidx, int from, int to)
{
    double energy0 = clusters_[from].energy + clusters_[to].energy;
    CovObj cov_from = clusters_[from].cov, cov_to = clusters_[to].cov;
    cov_from -= faces_[fidx].cov;
    cov_to += faces_[fidx].cov;
    double energy1 = cov_from.energy() + cov_to.energy();
    return energy0 - energy1;
}

//! After swapping step, some clusters may be split into unconnected component 'islands' (like
//! one component totally located inside another cluster). This is not good. Here we split each
//! of these 'island' clusters into different connected components based on cluster distribution,
//! and merge these island components to their neighbor clusters.
void Partition::processIslandClusters()
{
    for (int i = 0; i < face_num_; ++i)
        faces_[i].is_visited = false;
    int count_split_clusters = 0;
    int last_valid_cidx = 0;
    for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
    {
        if (!isClusterValid(cidx))
            continue;
        vector<unordered_set<int>> connected_components;
        auto cmp = [](unordered_set<int>& a, unordered_set<int>& b) { return a.size() > b.size(); };
        if (splitCluster(cidx, connected_components) > 1)
        {
            std::sort(connected_components.begin(), connected_components.end(), cmp);
            // Firstly merge the 'island' components (each with only 1 neighbor cluster) into their neighbor clusters.
            mergeIslandComponentsInCluster(cidx, connected_components);
            count_split_clusters++;

            // For the rest components, create a new cluster for each component
            if (connected_components.size() > 1)
            {
                for (size_t i = 1; i < connected_components.size(); ++i)  // leave the largest (first) component where it is
                {
                    // Find a valid cluster index to 'hold' the component as a new cluster and create new cluster there
                    int pos = last_valid_cidx;
                    while (pos < init_cluster_num_ && isClusterValid(pos))
                        pos++;
                    assert(pos < init_cluster_num_);
                    last_valid_cidx = pos;
                    clusters_[pos].faces.clear();
                    clusters_[pos].cov.clearCov();
                    for (int fidx : connected_components[i])
                    {
                        faces_[fidx].cluster_id = pos;
                        clusters_[pos].cov += faces_[fidx].cov;
                        clusters_[pos].faces.insert(fidx);
                        clusters_[cidx].faces.erase(fidx);
//
//                        clusters_[pos].avg_color += ;  // TODO cluster color
//                        clusters_[cidx].avg_color += ;  // TODO cluster color
                    }
                    clusters_[pos].energy = clusters_[pos].cov.energy();
                }
            }
        }
    }
    cout << "#Split clusters: " << count_split_clusters << endl;
}

//! Split one cluster into separate connected components by breath-first search. Return the component number.
int Partition::splitCluster(int cidx, vector<unordered_set<int>>& connected_components)
{
    int count_left_faces = int(clusters_[cidx].faces.size());
    connected_components.push_back(unordered_set<int>());
    for (int fidx : clusters_[cidx].faces)
    {
        int count = traverseFaceBFS(fidx, cidx, connected_components.back());
        if (count == 0)  // visited face
            continue;
        count_left_faces -= count;
        if (count_left_faces == 0)
            break;  // quit until all faces in the cluster are visited
        connected_components.push_back(unordered_set<int>());
    }
    return int(connected_components.size());
}

//! Traverse (unvisited) faces from one same cluster with BFS. Return the number of visited faces.
int Partition::traverseFaceBFS(int start_fidx, int start_cidx, unordered_set<int>& component)
{
    if (faces_[start_fidx].is_visited)
        return 0;
    faces_[start_fidx].is_visited = true;
    queue<int> qe;  // typical BFS with a queue
    qe.push(start_fidx);
    while (!qe.empty())
    {
        int fidx = qe.front();
        qe.pop();
        component.insert(fidx);
        for (int nbr : faces_[fidx].nbr_faces)
        {
            if (faces_[nbr].is_visited || faces_[nbr].cluster_id != start_cidx)
                continue;
            faces_[nbr].is_visited = true;
            qe.push(nbr);
        }
    }
    return int(component.size());
}

//! Merge island components from one cluster to its neighbor cluster.
/*!
    \param connected_components are separate connected components split from one cluster.
    NOTE: 'island' component is a connected component with only 1 neighbor cluster.
*/
void Partition::mergeIslandComponentsInCluster(int original_cidx, vector<unordered_set<int>>& connected_components)
{
    if (connected_components.size() <= 1)
        return;
    for (auto iter = connected_components.begin(); iter != connected_components.end();)
    {
        unordered_set<int> neighbors;
        unordered_set<int>& component = *iter;
        int neighbor_num = findClusterNeighbors(original_cidx, component, neighbors);
        if (neighbor_num != 1)
        {  // Only merge component with 1 neighbor cluster (means this is an island cluster)
            ++iter;
            continue;
        }
        // Merge this component into its neighbor cluster by fault without additional check.
        int target_cidx = *neighbors.begin();
        for (int fidx : component)
        {
            clusters_[target_cidx].cov += faces_[fidx].cov;
            clusters_[original_cidx].cov -= faces_[fidx].cov;
            clusters_[target_cidx].faces.insert(fidx);
            clusters_[original_cidx].faces.erase(fidx);
            faces_[fidx].cluster_id = target_cidx;
            faces_[fidx].is_visited = false;  // do NOT forget this
        }
        clusters_[original_cidx].energy = clusters_[original_cidx].cov.energy();
        clusters_[target_cidx].energy = clusters_[target_cidx].cov.energy();
        iter = connected_components.erase(iter);
    }
}

//! Compute maximum distance between points from cluster c2 to plane c1.
/*!
    \param flag_use_projection Use projected vertices (projections on the plane) instead of original vertices
*/
double Partition::computeMaxDisBetweenTwoPlanes(int c1, int c2, bool flag_use_projection)
{
    const Vector3d& ctr1 = clusters_[c1].cov.center_;
    const Vector3d& n1 = clusters_[c1].cov.normal_;  // NOTE: here assume the normal is the latest
    const Vector3d& ctr2 = clusters_[c2].cov.center_;
    const Vector3d& n2 = clusters_[c2].cov.normal_;  // NOTE: here assume the normal is the latest
    double max_dis = 0;
    for (int fidx : clusters_[c2].faces)
    {
        Vector3d vtx = faces_[fidx].cov.center_;
        if (flag_use_projection)
        {
            // Project the vertex on its plane
            vtx = vtx - ((vtx - ctr2).dot(n2)) * n2;
        }
        double dis = fabs((vtx - ctr1).dot(n1));
        max_dis = std::max(max_dis, dis);
    }
    return max_dis;
}

//! Average distance between points in cluster c2 and plane c1
double Partition::computeAvgDisBtwTwoPlanes(int c1, int c2)
{
    double dis = 0;
    const Vector3d& n1 = clusters_[c1].cov.normal_;
    const Vector3d& ctr1 = clusters_[c1].cov.center_;
    for (int fidx : clusters_[c2].faces)
    {
        Vector3d pt = faces_[fidx].cov.center_;
        dis += fabs(n1.dot(pt - ctr1));
    }
    return dis / int(clusters_[c2].faces.size());
}

void Partition::updateCurrentClusterNum()
{
    curr_cluster_num_ = 0;
    for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
        if (isClusterValid(cidx))
            curr_cluster_num_++;
}

//! Post processing step, including merging adjacent planes together, and merging island planes to neighbors.
/*!
    NOTE: You can read PLY mesh and cluster file, and then run this function without running the mesh partiton
    step which takes a very long time.
*/
void Partition::runPostProcessing()
{
    // If reading cluster data from input file, then need to initialize relevant cluster data at first
    if (flag_read_cluster_file_)
    {
        initMeshConnectivity();
        // Initialize face quadratic objects
        for (int fidx = 0; fidx < face_num_; fidx++)
        {
            Face& face = faces_[fidx];
            face.cov = CovObj(vertices_[face.indices[0]].pt, vertices_[face.indices[1]].pt, vertices_[face.indices[2]].pt);
        }
        // Initialize clusters
        init_cluster_num_ = curr_cluster_num_;  // no redundant empty clusters now
        for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
        {
            for (int fidx : clusters_[cidx].faces)
            {
                faces_[fidx].cluster_id = cidx;
                CovObj Q(vertices_[faces_[fidx].indices[0]].pt, vertices_[faces_[fidx].indices[1]].pt,
                         vertices_[faces_[fidx].indices[2]].pt);
                faces_[fidx].cov = Q;
                clusters_[cidx].cov += Q;
            }
            clusters_[cidx].energy = clusters_[cidx].cov.energy();
        }
    }

    // Faces are not merged well enough because of noisy data. So some adjacent planes can be
    // further merged together.
    mergeAdjacentPlanes();

    // Remove some small island clusters totally located inside other clusters.
    mergeIslandClusters();

    // (Optional) Clean small clusters which are independent connected components (like small floating pieces).
    // NOTE: this may cause problems since the threshold to determine a small cluster is hard to set. So it may be better
    // not use it.
    // removeSmallClusters();
    // // Only update vertex/face indices if faces are removed in 'removeSmallClusters()'
    // if (flag_new_mesh_ && !run_mesh_simplification)
    // {
    //     // If running mesh simplification later, the following two steps will also be run later.
    //     updateNewMeshIndices();
    //     updateCurrentClusterNum();
    // }
    updateCurrentClusterNum();
}

//! This function is for double checking if the mesh partition process is correct through checking each
//! cluster's faces.
void Partition::doubleCheckClusters()
{
    map<int, vector<int>> cluster_face_num;
    for (int fidx = 0; fidx < face_num_; fidx++)
    {
        Face& face = faces_[fidx];
        if (!face.is_valid)
            continue;
        int cidx = face.cluster_id;
        if (cidx == -1 || cidx >= init_cluster_num_)
        {
            PRINT_RED("ERROR: face %d, cluster %d, init_cluster_num: %d", fidx, cidx, init_cluster_num_);
            continue;
        }
        cluster_face_num[cidx].push_back(fidx);
    }
    int count = 0;
    for (auto& it : cluster_face_num)
    {
        int cidx = it.first;
        int count_faces = it.second.size();
        int cluster_faces = clusters_[cidx].faces.size();
        if (count_faces < cluster_faces)
        {
            PRINT_YELLOW("Cluster %d: #faces %d, #faces counted %d", cidx, cluster_faces, count_faces);
        }
        else if (count_faces > cluster_faces)
        {
            PRINT_GREEN("Cluster %d: #faces %d, #faces counted %d", cidx, cluster_faces, count_faces);
        }
        count += count_faces;
    }
    if (count != face_num_)
    {
        PRINT_RED("#Count faces: %d, #faces: %d", count, face_num_);
    }
}

//! Merge adjacent planes together by starting from one plane and spread to its neighbors as long as
//! they satisfy the merging conditions.
void Partition::mergeAdjacentPlanes()
{
    cout << "Start merging adjacent planes ... " << endl;
    updateCurrentClusterNum();
    cout << "Cluster number (before merging): " << curr_cluster_num_ << endl;

    for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
    {
        if (!isClusterValid(cidx))
            continue;
        clusters_[cidx].cov.computePlaneNormal();
        findClusterNeighbors(cidx);
    }
    const double kNormalAngleThrsd = cos(kPI * normal_angle_threshold / 180);
    const double kCenterNormalAngleThrsd = cos(kPI * center_normal_angle_threshold / 180);
    float progress = 0.0f;  // for printing a progress bar
    const int kStep = (init_cluster_num_ < 100) ? 1 : (init_cluster_num_ / 100);
    for (int c1 = 0; c1 < init_cluster_num_; ++c1)
    {
        if (c1 % kStep == 0 || c1 == init_cluster_num_ - 1)
        {
            progress = (c1 == init_cluster_num_ - 1) ? 1.0f : static_cast<float>(c1) / init_cluster_num_;
            printProgressBar(progress);
        }
        if (!isClusterValid(c1))
            continue;
        clusters_[c1].cov.computePlaneNormal();

        findClusterNeighbors(c1);
        const Vector3d& n1 = clusters_[c1].cov.normal_;
        while (!clusters_[c1].nbr_clusters.empty())
        {
            int c2 = *clusters_[c1].nbr_clusters.begin();
            clusters_[c1].nbr_clusters.erase(c2);  // this ensures quitting the loop at last
            if (!isClusterValid(c2))
                continue;
            if (clusters_[c1].cov.area_ < clusters_[c2].cov.area_)
                continue;  // always merge small plane c2 to large plane c1
            const Vector3d& n2 = clusters_[c2].cov.normal_;
            if (fabs(n1.dot(n2)) < kNormalAngleThrsd)
                continue;  // skip the neighbor plane with large normal direction difference
            Vector3d dir = (clusters_[c1].cov.center_ - clusters_[c2].cov.center_).normalized();
            if (fabs(dir.dot(n1)) > kCenterNormalAngleThrsd || fabs(dir.dot(n2)) > kCenterNormalAngleThrsd)
                continue;  // skip plane pair if the direction of two centers is close to one plane normal

            // if (computeMaxDisBetweenTwoPlanes(c1, c2, true) > point_plane_dis_threshold ||
            //     computeMaxDisBetweenTwoPlanes(c2, c1, true) > point_plane_dis_threshold)
            //     continue;
            // if (computeAvgDisBtwTwoPlanes(c1, c2) > point_plane_dis_threshold ||
            //     computeAvgDisBtwTwoPlanes(c2, c1) > point_plane_dis_threshold)
            //     continue;  // skip plane pair if their distance is too far
            if (computeMaxDisBetweenTwoPlanes(c1, c2, true) > point_plane_dis_threshold)
                continue;
            updateClusterColor(c1); // TODO: cluster color
            updateClusterColor(c2); // TODO: cluster color
            double dcolor = computeColorDiffBetweenPlanes(c1, c2);
//            PRINT_MAGENTA("COMPARING COLORS: %f", dcolor);
            if (dcolor > lab_color_diff_threshold)
                continue;  // TODO... could add some randomness here. might help...

            // If passing all the above checks, merge cluster/plane c2 to c1
            mergeClusters(c1, c2);
            clusters_[c1].cov += clusters_[c2].cov;
            clusters_[c2].cov.clearCov();
            clusters_[c1].cov.computePlaneNormal();

            double fc1 = clusters_[c1].faces.size();
            double fc2 = clusters_[c2].faces.size();
            clusters_[c1].avg_color = (clusters_[c1].avg_color*fc1 + clusters_[c2].avg_color*fc2)/(fc1 + fc2);

            findClusterNeighbors(c1);
        }
    }

    updateCurrentClusterNum();
    cout << "Cluster number (after merging): " << curr_cluster_num_ << endl;
}

/*! A cluster is an 'island' cluster if MOST of its cluster border faces have a same neighbor cluster.
    Usually this kind of cluster is entirely located inside some other larger cluster, such as some
    small parts of a bumpy floor. We will merge these islands to their 'best' neighbor clusters.
*/
void Partition::mergeIslandClusters()
{
    cout << "Remove island clusters ... " << endl;
    float progress = 0.0f;
    const int kStep = (init_cluster_num_ < 100) ? 1 : (init_cluster_num_ / 100);
    for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
    {
        if (cidx % kStep == 0 || cidx == init_cluster_num_ - 1)
        {
            progress = (cidx == init_cluster_num_ - 1) ? 1.0f : static_cast<float>(cidx) / init_cluster_num_;
            printProgressBar(progress);
        }
        if (!isClusterValid(cidx))
            continue;
        unordered_map<int, int> plane_neighbor_clusters;  // neighbor cluster id -> #border faces
        int count_border_faces = 0, count_cluster_border_faces = 0;
        for (int fidx : clusters_[cidx].faces)
        {
            bool flag_is_border = false, flag_is_cluster_border = false;
            if (faces_[fidx].nbr_faces.size() < 3)
                flag_is_border = true;
            unordered_set<int> face_nbr_clusters;
            for (int nbr : faces_[fidx].nbr_faces)
            {
                int ncidx = faces_[nbr].cluster_id;
                if (ncidx != cidx)
                    face_nbr_clusters.insert(ncidx);
            }
            if (face_nbr_clusters.size() > 0)
            {
                for (int ncidx : face_nbr_clusters)
                    plane_neighbor_clusters[ncidx]++;
                flag_is_cluster_border = true;
            }
            if (flag_is_cluster_border)
                count_cluster_border_faces++;  // cluster border faces only
            if (flag_is_border || flag_is_cluster_border)
                count_border_faces++;  // mesh border and cluster border faces together
        }
        // Experiential only: merge clusters with 1, 2 or 3 neighbor clusters
        if (plane_neighbor_clusters.size() < 1 || plane_neighbor_clusters.size() > 3)
            continue;

        double cluster_border_ratio = static_cast<double>(count_cluster_border_faces) / count_border_faces;
        if (cluster_border_ratio < island_cluster_border_ratio)
            continue;  // skip planes with a large part of mesh border faces
        // Find the neighbor cluster corresponding to maximum border faces
        int target_nbr = -1, max_faces = 0;
        for (auto it : plane_neighbor_clusters)
        {
            if (it.second > max_faces)
            {
                max_faces = it.second;  // # border faces
                target_nbr = it.first;  // neighbor cluster id
            }
        }
        double prior_cluster_ratio = static_cast<double>(max_faces) / count_cluster_border_faces;
        if (max_faces < island_cluster_border_ratio)
            continue;  // only merge island cluster while MOST of its border faces has a same neighbor cluster

        mergeClusters(target_nbr, cidx);
    }

    updateCurrentClusterNum();
    cout << "Cluster number (after removing islands): " << curr_cluster_num_ << endl;
}

//! Remove clusters which are small connected components. These clusters are independent components in the mesh,
//! like small floating pieces. Therefore, the removal will not influence other processing.
void Partition::removeSmallClusters()
{
    cout << "Remove small independent clusters ... " << endl;
    for (int i = 0; i < face_num_; ++i)
    {  // Reset flags just in case they are used before
        faces_[i].is_valid = true;
        faces_[i].is_visited = false;
    }

    unordered_set<int> small_clusters;
    for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
    {
        if (!isClusterValid(cidx) || clusters_[cidx].faces.size() > smallest_connected_component_size ||
            small_clusters.find(cidx) != small_clusters.end())
            continue;  // skip invalid or large or visited clusters

        // A cluster is always one connected component (since we create each cluster by spreading
        // one face to its neighbors). So we only need to check if each cluster is a small connected
        // component not adjacent to other clusters.
        // Start from one face and spread to neighbors until no new neighbors, or the number of visited
        // faces is large enough.
        int fidx = *clusters_[cidx].faces.begin();
        queue<int> qu;
        qu.push(fidx);
        faces_[fidx].is_visited = true;
        bool flag_small_component = true;
        unordered_set<int> visited_clusters;
        visited_clusters.insert(cidx);
        vector<int> fset;
        while (!qu.empty())
        {
            int f = qu.front();
            qu.pop();
            fset.push_back(f);
            // Sometimes a small cluster is adjacent to another cluster, but their total size
            // is still small. Save them both and remove them later.
            if (faces_[f].cluster_id != cidx)
                visited_clusters.insert(faces_[f].cluster_id);
            if (fset.size() > smallest_connected_component_size)
            {
                // Here we use the number of faces as the criteria to determine a small component.
                // Obviously this only works on the initial mesh which is usually created by some 3D
                // reconstruction system that each face has almost the same area.
                flag_small_component = false;
                break;
            }
            for (int nf : faces_[f].nbr_faces)
            {
                if (!faces_[nf].is_visited)
                {
                    faces_[nf].is_visited = true;
                    qu.push(nf);
                }
            }
        }
        if (flag_small_component)
        {
            small_clusters.insert(visited_clusters.begin(), visited_clusters.end());
            // Reset 'visited' flags of all relevant faces
            for (int fidx : fset)
                faces_[fidx].is_visited = false;
            while (!qu.empty())
            {
                int f = qu.front();
                qu.pop();
                faces_[f].is_visited = false;
            }
        }
    }
    // By removing these small clusters, we set the flags of all their faces as invalid for now
    // and skip them in other steps.
    if (!small_clusters.empty())
    {
        for (int cidx : small_clusters)
        {
            for (int fidx : clusters_[cidx].faces)
                faces_[fidx].is_valid = false;
            clusters_[cidx].faces.clear();
        }
        flag_new_mesh_ = true;  // Remember to set a flag for creating a new mesh
    }
    cout << "Small clusters removed: " << small_clusters.size() << endl;
    updateCurrentClusterNum();
    cout << "Cluster number (after removing): " << curr_cluster_num_ << endl;
}

//! If removing some faces or vertices, vertex/face indices and cluster data in the new mesh
//! will be different from original mesh. So here we update the indices for the new mesh.
//! This function is usually called in the last step before writing PLY mesh and cluster
//! file.
void Partition::updateNewMeshIndices()
{
    cout << "Original #Vertices: " << vertex_num_ << ", #Faces: " << face_num_ << endl;
    for (int i = 0; i < vertex_num_; ++i)
        vertices_[i].is_valid = false;
    new_face_num_ = 0;
    if (run_mesh_simplification)
    {  // Have to update faces in clusters after mesh simplification, since most faces are removed.
        for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
        {
            if (isClusterValid(cidx))
                clusters_[cidx].faces.clear();
        }
    }
    for (int fidx = 0; fidx < face_num_; ++fidx)
    {
        if (faces_[fidx].is_valid)
        {
            fidx_old2new_[fidx] = new_face_num_++;
            for (int j = 0; j < 3; ++j)
                vertices_[faces_[fidx].indices[j]].is_valid = true;
            if (run_mesh_simplification)
            {
                // Note to add new face index instead of original ones. Note that each face's cluster-id is still the same.
                clusters_[faces_[fidx].cluster_id].faces.insert(fidx_old2new_[fidx]);
            }
        }
    }
    new_vertex_num_ = 0;
    for (int i = 0; i < vertex_num_; ++i)
    {
        if (vertices_[i].is_valid)
            vidx_old2new_[i] = new_vertex_num_++;
    }
    flag_new_mesh_ = true;  // just in case forgot to set this flag before

    updateCurrentClusterNum();
    cout << "New #Vertices: " << new_vertex_num_ << ", #Faces: " << new_face_num_ << ", #Clusters: " << curr_cluster_num_
         << endl;
}

void Partition::runSimplification()
{
    initSimplification();

    // Two-step simplification
    simplifyInnerEdges();
    simplifyBorderEdges();

    // Update new mesh vertex and face indices since some vertices and faces are removed
    updateNewMeshIndices();
}

void Partition::initSimplification()
{
    cout << "Initialize simplification ... " << endl;
    // Reset each face's cluster-id just in case not set before
    for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
    {
        if (isClusterValid(cidx))
        {
            for (int fidx : clusters_[cidx].faces)
                faces_[fidx].cluster_id = cidx;
        }
    }
    // This function will skip if it is already called before
    initMeshConnectivity();

    // Can only call this after calling 'initMeshConnectivity()'
    findInnerAndBorderEdges();

    // Must release the heap before
    initInnerEdgeQuadrics();
}

//! Find cluster border edges and vertices
void Partition::findInnerAndBorderEdges()
{
    // Get all cluster border and mesh border edges
    for (auto it : edge_to_face_)  // edge -> face list contains only valid faces
    {
        long long edge = it.first;
        int v1, v2;
        getEdge(edge, v1, v2);
        int edge_fnum = static_cast<int>(it.second.size());
        bool flag_border_edge = false;
        if (edge_fnum == 0)
        {
            PRINT_RED("Edge (%d,%d) is not in any face. This shouldn't happen.", v1, v2);
        }
        else if (edge_fnum == 1)
            flag_border_edge = true;  // mesh border
        else
        {
            int init_cluster_id = faces_[it.second[0]].cluster_id;
            assert(init_cluster_id != -1);
            bool flag_same_cluster = true;
            for (size_t i = 1; i < it.second.size(); ++i)
            {
                if (faces_[it.second[i]].cluster_id != init_cluster_id)
                {
                    flag_same_cluster = false;
                    break;
                }
            }
            if (flag_same_cluster)  // all faces of this edge are in the same cluster
                vertices_[v1].cluster_id = vertices_[v2].cluster_id = init_cluster_id;
            else
                flag_border_edge = true;  // cluster border
        }
        if (flag_border_edge)  // border edges include both mesh border and cluster border edges
            border_edges_.insert(edge);
    }
    // Reset border vertices' flags
    for (long long edge : border_edges_)
    {
        int v1, v2;
        getEdge(edge, v1, v2);
        vertices_[v1].cluster_id = vertices_[v2].cluster_id = -1;  // border vertices' cluster-id is always -1
    }
    // Get inner-cluster edges. An inner-cluster edge is an edge whose two endpoints' ALL adjacent
    // faces belong to a same cluster (i.e., the edge is surrounded by one cluster faces).
    int count_inner_edges = 0;
    for (auto it : edge_to_face_)
    {
        long long edge = it.first;
        int v1, v2;
        getEdge(edge, v1, v2);
        if (vertices_[v1].cluster_id == -1 || vertices_[v2].cluster_id == -1 ||
            vertices_[v1].cluster_id != vertices_[v2].cluster_id)
            continue;
        cluster_inner_edges_[vertices_[v1].cluster_id].push_back(edge);
        count_inner_edges++;
    }
    cout << "#Border edges: " << border_edges_.size() << ", #Inner Edges: " << count_inner_edges << endl;
}

//! Initialize quadratic objects for inner vertices. See QEM reference for details.
void Partition::initInnerEdgeQuadrics()
{
    // face quadrics for all vertices
    const double kConstCoeff = 4.0 * sqrt(3.0);  // coefficient for triangle compactness (from the QEM paper)
    for (int i = 0; i < face_num_; ++i)
    {
        if (!faces_[i].is_valid)
            continue;
        int v0 = faces_[i].indices[0], v1 = faces_[i].indices[1], v2 = faces_[i].indices[2];

        // Initial triangle quadratic
        QEMQuadrics Q(vertices_[v0].pt, vertices_[v1].pt, vertices_[v2].pt);
        Q *= kFaceCoefficient;

        // Compute a measure of triangle compactness to give large weight to a triangle close
        // to equilateral, and small weight to a triangle whose vertices are close to colinear.
        // From Eq. (3.20) in QEM reference paper "Quadric-based polygonal surface simplification"
        // by M. Garland.
        Vector3d e0 = vertices_[v1].pt - vertices_[v0].pt;
        Vector3d e1 = vertices_[v2].pt - vertices_[v1].pt;
        Vector3d e2 = vertices_[v2].pt - vertices_[v0].pt;
        double area = (e0.cross(e2)).norm() / 2.0;
        double coef = kConstCoeff * area / (e0.squaredNorm() + e1.squaredNorm() + e2.squaredNorm());
        Q *= coef;

        vertices_[v0].Q += Q;
        vertices_[v1].Q += Q;
        vertices_[v2].Q += Q;
    }
    // Scale triangle quadrics for each vertex
    const double kFaceFactor = 1.0 / 3;  // normalizing factor (from the paper)
    for (int i = 0; i < vertex_num_; ++i)
        vertices_[i].Q *= 1.0 / (3.0 * vertices_[i].nbr_faces.size());

    // Point quadrics for all vertices
    for (int i = 0; i < vertex_num_; ++i)
    {
        QEMQuadrics Q(vertices_[i].pt);
        Q *= kPointCoefficient;
        vertices_[i].Q += Q;
    }
    // Line quadrics for border edges
    for (long long key : border_edges_)
    {
        int v1, v2;
        getEdge(key, v1, v2);
        QEMQuadrics Q(vertices_[v1].pt, vertices_[v2].pt);
        vertices_[v1].Q += Q;
        vertices_[v2].Q += Q;
    }
}

void Partition::simplifyInnerEdges()
{
    cout << "Simplify inner edges ... " << endl;

    // Simplify inner edges in each cluster.
    // Note: this step can be run in parallel, since inner-edge simplification in each cluster
    // is independent with others. Here is simple serial step cluster by cluster.
    float progress = 0.0;  // used to print a progress bar
    const int kStep = (init_cluster_num_ < 100) ? 1 : (init_cluster_num_ / 100);
    for (int cidx = 0; cidx < init_cluster_num_; ++cidx)
    {
        if (cidx % kStep == 0 || cidx == init_cluster_num_ - 1)
        {
            progress = (cidx == init_cluster_num_ - 1) ? 1.0f : static_cast<float>(cidx) / init_cluster_num_;
            printProgressBar(progress);
        }
        if (!isClusterValid(cidx) || cluster_inner_edges_.find(cidx) == cluster_inner_edges_.end() ||
            cluster_inner_edges_[cidx].size() < smallest_inner_edge_number)
            continue;

        releaseHeap();

        // Add candidate inner-cluster edges into the heap
        for (long long key : cluster_inner_edges_[cidx])
        {
            int v1, v2;
            getEdge(key, v1, v2);
            Edge* e = new Edge(v1, v2);
            if (checkEdgeContraction(e))
            {
                heap_.insert(e);
                vertices_[v1].nbr_edges.push_back(e);
                vertices_[v2].nbr_edges.push_back(e);
            }
            else
            {
                delete e;
                e = nullptr;
            }
        }

        // Simplify inner edges
        curr_edge_num_ = heap_.size();
        while (curr_edge_num_ > smallest_inner_edge_number)
        {
            Edge* edge = (Edge*)heap_.extract();
            if (!edge)
            {
                cout << "WARNING: No edges in the heap. " << endl;
                break;
            }
            // NOTE: Even though we have already checked the validness of each edge's contraction
            // before adding it into the heap, we still need to check it again before contracting
            // it. The reason is that, we may change the connectivity of the edge's neighborhood
            // after inserting it into the heap. Therefore, a valid edge in the heap may become
            // invalid again.
            if (checkEdgeContraction(edge))
                applyVtxEdgeContraction(edge, cidx);
            else
            {
                // If this edge cannot be contracted, we still remove it from the heap to ensure
                // the loop will end at last.
                curr_edge_num_--;
                removeEdgeFromList(edge, vertices_[edge->v1].nbr_edges);
                removeEdgeFromList(edge, vertices_[edge->v2].nbr_edges);
                delete edge;
                edge = nullptr;
            }
        }
        assert(curr_edge_num_ <= smallest_inner_edge_number);
    }
}

//! Apply edge contraction for all types of edges including inner-cluster edge, border-cluster edge,
//! and mesh border edge. These cases are very similar except some details.
void Partition::applyVtxEdgeContraction(Edge* edge, int cluster_idx)
{
    int v1 = edge->v1, v2 = edge->v2;
    vertices_[v1].Q += vertices_[v2].Q;
    vertices_[v2].Q.reset();
    vertices_[v2].is_valid = false;
    double energy = 0;

    // After edge contraction, v1 will hold the new vertex position and also
    // inherit the connectivity of both original v1 and v2.
    if (!vertices_[v1].Q.optimize(vertices_[v1].pt, energy))
    {
        // If cannot find optimized vertex position, still use the endpoint with
        // lower energy result. This is the same as done in checkEdgeContraction().
        double energy1 = vertices_[v1].Q(vertices_[v1].pt);
        double energy2 = vertices_[v1].Q(vertices_[v2].pt);
        if (energy1 > energy2)
            vertices_[v1].pt = vertices_[v2].pt;
    }

    // Merge adjacent vertices of v2 to v1
    for (int vidx : vertices_[v2].nbr_vertices)
    {
        vertices_[vidx].nbr_vertices.erase(v2);
        if (vidx != v1)
        {
            vertices_[v1].nbr_vertices.insert(vidx);
            vertices_[vidx].nbr_vertices.insert(v1);
        }
    }
    // Update adjacent faces of v2 by v1
    for (int fidx : vertices_[v2].nbr_faces)
    {
        if (checkFaceContainsVertices(fidx, v1, v2))
        {  // remove faces containing both v1 and v2
            faces_[fidx].is_valid = false;
            for (int i = 0; i < 3; ++i)
            {
                int vidx = faces_[fidx].indices[i];
                if (vidx != v2)
                    vertices_[vidx].nbr_faces.erase(fidx);
            }
        }
        else
        {  // replace v2 in v2's adjacent face by v1
            for (int i = 0; i < 3; ++i)
            {
                if (faces_[fidx].indices[i] == v2)
                {
                    faces_[fidx].indices[i] = v1;
                    break;
                }
            }
            vertices_[v1].nbr_faces.insert(fidx);
        }
    }

    // Remove all old edges of v1 and v2
    // NOTE: the input edge is also deleted here, since v1 is one of its endpoint.
    // cout << "     Edges for v1" << endl;
    for (Edge* e : vertices_[v1].nbr_edges)
    {
        int u = (e->v1 == v1) ? e->v2 : e->v1;
        heap_.remove(e);
        // Like before, remove this edge in the list of the other endpoint than v1,
        // since each edge pointer is stored twice but can only be deleted once.
        removeEdgeFromList(e, vertices_[u].nbr_edges);
        delete e;
        e = nullptr;
        curr_edge_num_--;
    }
    // cout << "     Edges for v2" << endl;
    for (Edge* e : vertices_[v2].nbr_edges)
    {
        int u = (e->v1 == v2) ? e->v2 : e->v1;
        heap_.remove(e);
        removeEdgeFromList(e, vertices_[u].nbr_edges);
        delete e;
        e = nullptr;
        curr_edge_num_--;
    }
    vertices_[v1].nbr_edges.clear();
    vertices_[v2].nbr_edges.clear();

    // Add new edges for v1
    for (int vidx : vertices_[v1].nbr_vertices)
    {
        // NOTE: Border vertex's cluster id is -1.
        if (vertices_[vidx].cluster_id != cluster_idx)
            continue;
        Edge* e = (v1 < vidx) ? (new Edge(v1, vidx)) : (new Edge(vidx, v1));
        if (checkEdgeContraction(e))
        {
            heap_.insert(e);
            vertices_[v1].nbr_edges.push_back(e);
            vertices_[vidx].nbr_edges.push_back(e);
            curr_edge_num_++;
        }
        else
        {
            delete e;
            e = nullptr;
        }
    }
}

// If edge can be contracted, return true and update the edge energy, otherwise return false.
bool Partition::checkEdgeContraction(Edge* edge)
{
    int v1 = edge->v1, v2 = edge->v2;

    // Preserve non-manifold topology or not? Your choice.
    // if (getCommonNeighborNum(v1, v2) > 2)
    //     return false;

    QEMQuadrics Q = vertices_[v1].Q;
    Q += vertices_[v2].Q;
    double energy = 0;
    Vector3d vtx;
    if (!Q.optimize(vtx, energy))
    {
        // If the planes represented by the two endpoints of the edge is nearly parallel, then
        // this optimize() will fail and we cannot find the optimal contracted vertex.
        // If so, use two simple strategies to determine the vertex. They are both intuitive.

        // Strategy 1: Use edge center as the target vertex
        // vtx = (vertices_[v1].pt + vertices_[v2].pt) / 2;
        // energy = Q.evaluate(vtx);

        // Strategy 2: Use endpoint with smaller energy as the target vertex.
        // Since energy actually is the distance between a point to the plane represented by the quadrics,
        // so these two energies are usually very close here. So it's like selecting an arbitrary one
        // endpoint from the two.
        double energy1 = Q(vertices_[v1].pt);
        double energy2 = Q(vertices_[v2].pt);
        if (energy1 < energy2)
        {
            energy = energy1;
            vtx = vertices_[v1].pt;
        }
        else
        {
            energy = energy2;
            vtx = vertices_[v2].pt;
        }
    }
    // Skip an edge which may cause face flipping/inversion. This is very important.
    if (!checkFlippedFaces(edge, v1, vtx) || !checkFlippedFaces(edge, v2, vtx))
        return false;

    edge->heap_key(-energy);  // Still, it is a max-heap by default but we need a min-heap
    return true;
}

//! Check if the contracted vertex after edge contraction causes flipped faces (simplex inversion)
//! around a vertex's adjacent faces.
//! Ref: Chapter 3.7 'Consistency check' in QEM reference paper
bool Partition::checkFlippedFaces(Edge* edge, int endpoint, const Vector3d& contracted_vtx)
{
    assert(endpoint == edge->v1 || endpoint == edge->v2);
    int v1 = endpoint, v2 = (edge->v1 == endpoint) ? edge->v2 : edge->v1;
    int p1 = v1;
    for (int fidx : vertices_[v1].nbr_faces)
    {
        if (vertices_[v2].nbr_faces.find(fidx) != vertices_[v2].nbr_faces.end())
            continue;  // skip faces shared by this edge, since they will be removed later

        // Find v1's index inside this face
        int vidx = -1;
        for (int i = 0; i < 3; ++i)
        {
            if (faces_[fidx].indices[i] == v1)
            {
                vidx = i;
                break;
            }
        }
        assert(vidx != -1);  // v1's adjacent face MUST contain v1

        // See Figure 3.15 in QEM paper for details about meaning of variables in this check
        int p2 = faces_[fidx].indices[(vidx + 1) % 3], p3 = faces_[fidx].indices[(vidx + 2) % 3];  // other two vertices
        Vector3d e1 = vertices_[p1].pt - vertices_[p2].pt;
        Vector3d e2 = (vertices_[p3].pt - vertices_[p2].pt).normalized();
        // This is the normal of the plane perpendicular to this face and going through edge (p2, p3)
        Vector3d n = (e1 - (e1.dot(e2)) * e2).normalized();
        // Ensure the distance between contracted vertex and this plane (with normal n and points
        // p2, p3 on it) is >= 0.
        if (n.dot(contracted_vtx - vertices_[p2].pt) <= 1e-5)
            return false;
    }
    return true;
}

int Partition::getCommonNeighborNum(int v1, int v2)
{
    int count = 0;
    for (int vidx : vertices_[v1].nbr_vertices)
    {
        if (vertices_[v2].nbr_vertices.find(vidx) != vertices_[v2].nbr_vertices.end())
            count++;
    }
    return count;
}

void Partition::simplifyBorderEdges()
{
    cout << "Simplify border edges ... " << endl;

    releaseHeap();

    // Add all border edges into heap.
    // NOTE: here we add all border edges including both mesh border and cluster border.
    // However, in some situation, you may want to keep the topology of mesh border. So
    // you can separate mesh border and cluster border edges during initialization, and
    // simplify them separately.
    for (long long key : border_edges_)
    {
        int v1, v2;
        getEdge(key, v1, v2);
        Edge* e = new Edge(v1, v2);
        if (checkEdgeContraction(e))
        {
            heap_.insert(e);
            vertices_[v1].nbr_edges.push_back(e);
            vertices_[v2].nbr_edges.push_back(e);
        }
        else
        {
            delete e;
            e = nullptr;
        }
    }

    // Simplify all border edges. This step is the same as simplifying inner-cluster edges.
    curr_edge_num_ = heap_.size();
    const int kTargetEdgeNum =
            max(static_cast<int>(simplification_border_edge_ratio * curr_edge_num_), smallest_inner_edge_number);
    while (curr_edge_num_ > kTargetEdgeNum)
    {
        Edge* edge = (Edge*)heap_.extract();
        if (!edge)
        {
            cout << "WARNING: No edges in the heap. " << endl;
            break;
        }
        if (checkEdgeContraction(edge))
            applyVtxEdgeContraction(edge, -1);  // border vertex's cluster id is -1
        else
        {
            // If this edge cannot be contracted, we still remove it from the heap to ensure
            // the loop will end at last.
            curr_edge_num_--;
            removeEdgeFromList(edge, vertices_[edge->v1].nbr_edges);
            removeEdgeFromList(edge, vertices_[edge->v2].nbr_edges);
            delete edge;
            edge = nullptr;
        }
    }
    assert(curr_edge_num_ <= kTargetEdgeNum);
}

double Partition::computeFaceArea(int f){

    int a = faces_[f].indices[0];
    int b = faces_[f].indices[1];
    int c = faces_[f].indices[2];

    double area = 0.0;

    //this check doesn't seem to be necessary as Im already checking validity later...
    if (vertices_[a].is_valid && vertices_[b].is_valid && vertices_[c].is_valid){

        Vector3d A = vertices_[a].pt;
        Vector3d B = vertices_[b].pt;
        Vector3d C = vertices_[c].pt;

        Vector3d AB = B-A;
        Vector3d AC = C-A;

        area = (AB.cross(AC)).norm() / 2.0;

    }

    return area;
}

void Partition::computeAllFaceAreas(){
    double area;
    double min_area= 999999999999999.9;
    double max_area = 0.0;
    int max_index = 0;
    for (int i=0; i<faces_.size(); i++)
    {
        if(faces_[i].is_valid)
        {
            faces_[i].area = computeFaceArea(i);
            area = faces_[i].area;
        }
        // if(i<10)
        //   PRINT_GREEN("FACE %d HAS AREA: %f", i, area);
        if(area < min_area)
            min_area = area;
        if(area > max_area)
        {
            max_area = area;
            max_index = i;
        }

    }
    PRINT_GREEN("min_area: %f", min_area);
    PRINT_GREEN("max_index: %d; max_area: %f", max_index, max_area);
}

void Partition::updateClusterInfo(){

    PRINT_YELLOW("Computing Face Areas");
    computeAllFaceAreas();

    PRINT_YELLOW("Computing Cluster Areas");
    double area;
    int face_count;

    PRINT_MAGENTA("CURR_CLUSTER_NUM: %d", curr_cluster_num_);
    PRINT_MAGENTA("CLUSTERS_.SIZE(): %lu",  clusters_.size());

    int filled_clusters = 0;

    for(int c=0; c < clusters_.size(); c++){
        clusters_[c].original_id = c;

        area = 0.0;
        face_count = 0;

        //print every 5 to keep things a bit cleaner
        if(filled_clusters % 5 == 0){
            PRINT_BLUE("FILLED_CLUSTERS = %d", filled_clusters);
        }

        if(filled_clusters==548 || filled_clusters==547){
            PRINT_YELLOW("Cluster %d. Area: %f; Face count: %d", filled_clusters, area, face_count);
        }

        for (int fidx : clusters_[c].faces){
            // add areas
            if(faces_[fidx].is_valid){
                if(filled_clusters==548 || filled_clusters==547){
                    PRINT_RED("Area: %f; Face count: %d", area, face_count);
                }
                area += faces_[fidx].area;
                face_count++;
            }
        }

        if(face_count>0){
            filled_clusters++;
        }

        clusters_[c].num_faces = face_count;
        clusters_[c].area = area;
//    if(face_count > 0)
//      PRINT_CYAN("Cluster %d: %d faces; %f area", c, face_count, area);
    }

    sortClusters(true);
}

void Partition::sortClusters(bool byArea){
    PRINT_YELLOW("SORTING CLUSTERS BY SIZE");
    if(byArea){
        orderClustersByArea();
        PRINT_YELLOW("(debug) CLUSTER AREAS ARE:");
        for(int i = 0; i < 10; i++){
            PRINT_YELLOW(" %d: %f", i, clusters_[i].area);
        }
    }
    else{
        orderClustersByFaceCount();
        PRINT_YELLOW("(debug) CLUSTER FACE COUNTS ARE:");
        for(int i = 0; i < 10; i++){
            PRINT_YELLOW(" %d: %d", i, clusters_[i].num_faces);
        }
    }
}

void Partition::orderClustersByArea(){
    std::sort(clusters_.begin(), clusters_.end(), compareByArea);
}

void Partition::orderClustersByFaceCount(){
    std::sort(clusters_.begin(), clusters_.end(), compareByNumFaces);
}

bool Partition::faceInTopNClusters(int face_num, int n_clusters){
    for(int i = 0; i < n_clusters; i++){
        if(clusters_[i].faces.find(face_num) != clusters_[i].faces.end()){
            return true;
        }
    }
    return false;
}

void Partition::updateClusters()
{

    // for each face, update it's vertex indices, if applicable. compute face area. add face to cluster map;
    for (int fidx = 0; fidx < face_num_; fidx++)
    {
        Face& face = faces_[fidx];
        if (!face.is_valid)
            continue;

        // faces_[fidx].area = computeFaceArea(fidx); //TODO: area is already included in face.cov.area. use this to reduce redundant calcs

        int cidx = face.cluster_id;
        if (cidx == -1 || cidx >= init_cluster_num_)
        {
            PRINT_RED("ERROR: face %d, cluster %d, init_cluster_num: %d", fidx, cidx, init_cluster_num_);
            continue;
        }
        cluster_face_num[cidx].push_back(fidx);
        // clusters_[cidx].area += faces_[fidx].area;
    }

    //update mesh centroid
    computeMeshCentroid(0.0); // input of 0.0 means all (valid) faces are weighted

    int count = 0;
    // for each cluster in the map, check the face counts, add the vertices to the cluster map
    for (auto& it : cluster_face_num)
    {
        int vidx = 0;

        int cidx = it.first;
        int count_faces = it.second.size();

        //for each face in the cluster add the vertices to the cluster set
        for(int fidx : it.second){
            for(int i=0; i<3; i++){
                if(cluster_vert_num[cidx].count(faces_[fidx].indices[i])==0){
                    cluster_vert_num[cidx].insert(faces_[fidx].indices[i]); //store the index from the original point list
                    cluster_vert_old2new[cidx][faces_[fidx].indices[i]] = vidx; //and the index in the context of the cluster
                    cluster_vert_new2old[cidx][vidx] = faces_[fidx].indices[i];
                    vidx++;
                }
            }
        }

        //update plane normals
        clusters_[cidx].cov.computePlaneNormal();

        //adjust normal directions based on a few simple rules
        Vector3f grav_dir; //not used for now...
        //changeClusterNormalDirection(cidx, grav_dir);

        clusters_[cidx].num_faces = count_faces;
//        if(clusters_[cidx].cov.area_ > 1.0){
//            PRINT_CYAN("Cluster %d: #faces %d, #cov_area %f, #face_normal [%f, %f, %f]", cidx, count_faces, clusters_[cidx].cov.area_, clusters_[cidx].cov.normal_[0], clusters_[cidx].cov.normal_[1], clusters_[cidx].cov.normal_[2]);
//        }
        int cluster_faces = clusters_[cidx].faces.size();
        if (count_faces < cluster_faces)
        {
            PRINT_YELLOW("Cluster %d: #faces %d, #faces counted %d", cidx, cluster_faces, count_faces);
        }
        else if (count_faces > cluster_faces)
        {
            PRINT_GREEN("Cluster %d: #faces %d, #faces counted %d", cidx, cluster_faces, count_faces);
        }
        count += count_faces;
    }
    if (count != face_num_)
    {
        PRINT_RED("#Count faces: %d, #faces: %d", count, face_num_);
    }
}

Vector3d Partition::computeMeshCentroid(double min_cluster_area){
    Vector3d weighted_areas(0,0,0);
    Vector3d centroid(0,0,0);
    double area=0;

    for(Face f : faces_){
        if(f.is_valid){
            if(clusters_[f.cluster_id].cov.area_ >= min_cluster_area){
                area+= f.cov.area_;
                weighted_areas+= f.cov.center_*f.cov.area_;
            }
        }
    }

    centroid = weighted_areas/area;
    mesh_centroid_ = centroid;
    return centroid;
}

double Partition::computeColorDiffBetweenPlanes(int c1, int c2){

    double diff = sqrt(pow(clusters_[c1].avg_color[0]-clusters_[c2].avg_color[0], 2.0)+
                  pow(clusters_[c1].avg_color[1]-clusters_[c2].avg_color[1], 2.0)+
                  pow(clusters_[c1].avg_color[2]-clusters_[c2].avg_color[2], 2.0));
    return diff;
}

void Partition::updateClusterColor(int c){
    int i = 0;
    Vector3d color(0.0,0.0,0.0);

    for(int fidx : clusters_[c].faces){
        color += faces_[fidx].color;
//        PRINT_RED("ADDING FACE WITH COLOR: %f", faces_[fidx].color[0]);
        i++;
    }
//    PRINT_RED("Average Cluster Color: %f, %f, %f", color[0]/((double)i), color[1]/((double)i), color[2]/((double)i));

    clusters_[c].avg_color = color/((double)i);
}