#ifndef PARTITION_H
#define PARTITION_H

#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <eigen3/Eigen/Eigen>
//#include "../common/covariance.h"
#include "covariance.h"
#include "MxHeap.h"
#include "qemquadrics.h"

using namespace std;
using namespace Eigen;

class Partition
{
public:
    struct Edge : public MxHeapable
    {
        int v1, v2;
        Edge(int a, int b) : v1(a), v2(b) {}
    };

    struct SwapFace
    {
        int face_id, from, to;
        SwapFace(int v, int f, int t)
        {
            face_id = v;
            from = f;
            to = t;
        }
    };

    struct Vertex
    {
        bool is_valid;  // false if it is removed (all its adjacent faces are removed)
        int cluster_id;

        Vector3d color; //TODO: implement
        Vector3d pt;
        unordered_set<int> nbr_vertices, nbr_faces;
        vector<Edge*> nbr_edges;
        QEMQuadrics Q;
        Vertex() : is_valid(false), cluster_id(-1) {}
    };

    struct Face
    {
        int cluster_id;
        bool is_visited;  // used in Breadth-first search to get connected components in clusters
        bool is_valid;    // false if this face is removed.
        int indices[3];
        Vector3d color; //TODO: implement
        double area;
        CovObj cov;
        unordered_set<int> nbr_faces;
        Face() : cluster_id(-1), area(0), is_visited(false), is_valid(true) {}
    };

    struct Cluster
    {
        int original_id;
        int num_faces;
        int num_vertices;
        double energy;             // to save some computation time of calling CovObj::energy() too frequently
        double area;               // used to calculate, store, and sort by mesh cluster area
        bool is_visited;           // used in Breath-first search to remove small floating clusters
        unordered_set<int> faces;  // faces each cluster contains
        unordered_set<int> nbr_clusters;
        vector<SwapFace> faces_to_swap;
        Vector3f color;
        Vector3d avg_color;
        CovObj cov;
        Cluster() : energy(0), area(0.0) {}
    };

public:
    Partition();
    ~Partition();
    bool readPLY(const std::string& filename);
    bool writePLY(const std::string& filename);
    bool writePLY(const std::string& filename, double min_area);
    bool writeTopPLYs(const std::string& basefilename, double min_area, Vector3d gravity_direction=Vector3d(0,1,0));

    bool runPartitionPipeline();
    void writeClusterFile(const std::string& filename);
    bool readClusterFile(const std::string& filename);
    void setTargetClusterNum(int num) { target_cluster_num_ = num; }
    int getCurrentClusterNum() { return curr_cluster_num_; }
    void printModelInfo() { cout << "#Vertices: " << vertices_.size() << ", #Faces: " << faces_.size() << endl; }
    void runPostProcessing();
    void runSimplification();
    void doubleCheckClusters();
    void updateClusterInfo();
    void updateClusters();

private:
    /* Merging */
    bool runMerging();
    void initMerging();
    void initMeshConnectivity();
    void computeEdgeEnergy(Edge* edge);
    bool removeEdgeFromList(Edge* edge, vector<Edge*>& edgelist);
    bool isClusterValid(int cidx) { return !clusters_[cidx].faces.empty(); }
    bool mergeOnce();
    void applyFaceEdgeContraction(Edge* edge);
    void mergeClusters(int c1, int c2);
    int findClusterNeighbors(int cidx);
    int findClusterNeighbors(int cidx, unordered_set<int>& cluster_faces, unordered_set<int>& neighbor_clusters);
    double getTotalEnergy();
    void createClusterColors();
    void updateCurrentClusterNum();
    void releaseHeap();

    /* Swap */
    void runSwapping();
    int swapOnce();
    double computeSwapDeltaEnergy(int fidx, int from, int to);
    void processIslandClusters();
    int splitCluster(int cidx, vector<unordered_set<int>>& connected_components);
    int traverseFaceBFS(int start_fidx, int start_cidx, unordered_set<int>& component);
    void mergeIslandComponentsInCluster(int original_cidx, vector<unordered_set<int>>& connected_components);

    /* Post processing */
    double computeMaxDisBetweenTwoPlanes(int c1, int c2, bool flag_use_projection = false);
    double computeAvgDisBtwTwoPlanes(int c1, int c2);
    void removeSmallClusters();
    void updateNewMeshIndices();
    void mergeAdjacentPlanes();
    void mergeIslandClusters();

    /* Simplification */
    void initSimplification();
    void findInnerAndBorderEdges();
    void initInnerEdgeQuadrics();
    void initBorderEdges();
    void simplifyInnerEdges();
    void simplifyBorderEdges();
    bool checkEdgeContraction(Edge* edge);
    int getCommonNeighborNum(int v1, int v2);
    bool checkFlippedFaces(Edge* edge, int endpoint, const Vector3d& contracted_vtx);
    void applyVtxEdgeContraction(Edge* edge, int cluster_idx);

    /* Geometric Functions */
    double computeFaceArea(int f);
    static bool compareByArea(const Cluster& a, const Cluster& b){
      return a.area > b.area;
    }
    static bool compareByNumFaces(const Cluster& a, const Cluster& b){
      return a.num_faces > b.num_faces;
    }
    bool faceInTopNClusters(int face_num, int n_clusters);    
    void computeAllFaceAreas();
    void orderClustersByArea();
    void orderClustersByFaceCount();
    void sortClusters(bool byArea); //TODO: remove unused functions
    Vector3d computeMeshCentroid(double min_cluster_area);
    Vector3d computeClusterCentroid(int c){
      return clusters_[c].cov.center_;
    }

    /* Color Management */
    double computeColorDiffBetweenPlanes(int c1, int c2);
    void updateClusterColor(int c);
    static Vector3d rgb2xyz(Vector3d rgb){

        double var_R = ( rgb[0] / 255.0 );
        double var_G = ( rgb[1] / 255.0 );
        double var_B = ( rgb[2] / 255.0 );

        if ( var_R > 0.04045 ){
            var_R = pow( ( var_R + 0.055 ) / 1.055, 2.4);
        }
        else{
            var_R /= 12.92;
        }
        if ( var_G > 0.04045 ){
            var_G =  pow(( var_G + 0.055 ) / 1.055, 2.4);
        }
        else{
            var_G /= 12.92;
        }
        if ( var_B > 0.04045 ){
            var_B = pow(( var_B + 0.055 ) / 1.055, 2.4);
        }
        else{
            var_B /= 12.92;
        }

        var_R *= 100.0;
        var_G *= 100.0;
        var_B *= 100.0;

        double X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
        double Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
        double Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

        Vector3d xyz(X, Y, Z);
        return {xyz};
    }
    static Vector3d xyz2lab(Vector3d xyz){
        double var_X = xyz[0] / ReferenceX;
        double var_Y = xyz[1] / ReferenceY;
        double var_Z = xyz[2] / ReferenceZ;

        if ( var_X > 0.008856 )
            var_X = pow(var_X, 1.0/3.0);
        else
            var_X = ( 7.787 * var_X ) + ( 16.0 / 116.0 );
        if ( var_Y > 0.008856 )
            var_Y = pow(var_Y, 1.0/3.0);
        else
            var_Y = ( 7.787 * var_Y ) + ( 16.0 / 116.0 );
        if ( var_Z > 0.008856 )
            var_Z = pow(var_Z, 1.0/3.0);
        else
            var_Z = ( 7.787 * var_Z ) + ( 16.0 / 116.0 );

        double L = ( 116.0 * var_Y ) - 16;
        double A = 500.0 * ( var_X - var_Y );
        double B = 200.0 * ( var_Y - var_Z );
        Vector3d lab(L, A, B);
        return {lab};
    }
    static Vector3d rgb2lab(Vector3d rgb){
        return xyz2lab(rgb2xyz(std::move(rgb)));
    }

    static Vector3d lab2xyz(Vector3d lab){
        double var_Y = ( lab[0] + 16.0 ) / 116.0;
        double var_X = lab[1] / 500.0 + var_Y;
        double var_Z = var_Y - lab[2] / 200.0;

        if ( pow(var_Y, 3.0)  > 0.008856 )
            var_Y = pow(var_Y, 3.0);
        else
            var_Y = ( var_Y - 16.0 / 116.0 ) / 7.787;
        if ( pow(var_X,3)  > 0.008856 )
            var_X = pow(var_X, 3.0);
        else
            var_X = ( var_X - 16.0 / 116.0 ) / 7.787;
        if ( pow(var_Z, 3.0)  > 0.008856 )
            var_Z = pow(var_Z, 3.0);
        else
            var_Z = ( var_Z - 16.0 / 116.0 ) / 7.787;

        double X = var_X * ReferenceX;
        double Y = var_Y * ReferenceY;
        double Z = var_Z * ReferenceZ;
        Vector3d xyz(X, Y, Z);
        return {xyz};
    }
    static Vector3d xyz2rgb(Vector3d xyz){
        double var_X = xyz[0] / 100.0;
        double var_Y = xyz[1] / 100.0;
        double var_Z = xyz[2] / 100.0;

        double var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986;
        double var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415;
        double var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570;

        if ( var_R > 0.0031308 )
            var_R = 1.055 * pow(var_R, ( 1.0 / 2.4 )) - 0.055;
        else
            var_R = 12.92 * var_R;
        if ( var_G > 0.0031308 )
            var_G = 1.055 * pow(var_G, ( 1.0 / 2.4 )) - 0.055;
        else
            var_G = 12.92 * var_G;
        if ( var_B > 0.0031308 )
            var_B = 1.055 * pow(var_B, ( 1.0 / 2.4 )) - 0.055;
        else
            var_B = 12.92 * var_B;

        double R = var_R * 255.0;
        double G = var_G * 255.0;
        double B = var_B * 255.0;
        Vector3d rgb(R, G, B);
        return {rgb};
    }
    static Vector3d lab2rgb(Vector3d lab){
        return xyz2rgb(lab2xyz(std::move(lab)));
    }

    static Vector3d rgb2yuv(Vector3d rgb){
        double y = Wr*rgb[0] + Wg*rgb[1] + Wb*rgb[2];  // Y'
        double u = U_max*(rgb[2]-y)/(1-Wb);
        double v = V_max*(rgb[0]-y)/(1-Wr);
        Vector3d yuv(y, u, v);
        return {yuv};
    }
    static Vector3d yuv2rgb(Vector3d yuv){
        double R = yuv[0] + yuv[2]*(1-Wr)/V_max;
        double G = yuv[0] + yuv[1]*Wb*(1-Wb)/(U_max*Wg)-yuv[2]*Wr*(1-Wr)/(V_max*Wg);
        double B = yuv[0] + yuv[1]*(1-Wb)/U_max;
        Vector3d rgb(R, G, B);
        return {rgb};
    }

    /* Small functions */
    template <typename T> int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

    //! Check if a face contains two vertices
    inline bool checkFaceContainsVertices(int fidx, int v1, int v2)
    {
        return checkFaceContainsVertices(fidx, v1) && checkFaceContainsVertices(fidx, v2);
    }
    //! Check if a face contains one vertex
    inline bool checkFaceContainsVertices(int fidx, int v1)
    {
        return faces_[fidx].indices[0] == v1 || faces_[fidx].indices[1] == v1 || faces_[fidx].indices[2] == v1;
    }
    //! Convert an long long edge type to two endpoints
    inline void getEdge(const long long& key, int& v1, int& v2)
    {
        v2 = int(key & 0xffffffffLL);  // v2 is lower 32 bits of the 64-bit edge integer
        v1 = int(key >> 32);           // v1 is higher 32 bits
    }

private:

    // store the final maps for each cluster to make PLYs from each cluster as needed
    map<int, vector<int>> cluster_face_num;
    map<int, unordered_set<int>> cluster_vert_num;
    map<int, unordered_map<int, int>> cluster_vert_old2new;
    map<int, unordered_map<int, int>> cluster_vert_new2old;
    Vector3d mesh_centroid_;

    int vertex_num_, face_num_;
    int init_cluster_num_, curr_cluster_num_, target_cluster_num_;
    bool flag_read_cluster_file_;
    vector<Vertex> vertices_;
    vector<Face> faces_;
    vector<Cluster> clusters_;
    vector<Cluster> ordered_clusters_;
    vector<vector<Edge*>> global_edges_;
    MxHeap heap_;
    double total_energy_;
    unordered_set<int> clusters_in_swap_, last_clusters_in_swap_;
    unordered_map<long long, vector<int>> edge_to_face_;         // edge (represented by two int32 endpoints) -> face id
    unordered_map<int, vector<long long>> cluster_inner_edges_;  // edges inside each cluster
    unordered_set<long long> border_edges_;                      // mesh border and cluster border edges
    unordered_map<int, int> vidx_old2new_;  // original vertex indices -> new mesh indices (after removing some faces)
    unordered_map<int, int> fidx_old2new_;  // original vertex indices -> new mesh indices (after removing some faces)
    int new_vertex_num_, new_face_num_;
    bool flag_new_mesh_;  // true if removing some faces/vertices/clusters; false by default

    // These are used to balance the importance of point and triangle quadrics, respectively.
    // However, equal values work well in experiments.
    const double kFaceCoefficient = 1.0, kPointCoefficient = 1.0;
    int curr_edge_num_;

    constexpr static const double Wr = 0.299;  // red weight
    constexpr static const double Wg = 0.587;  // green weight
    constexpr static const double Wb = 0.114;  // blue weight
    constexpr static const double U_max = 0.436;
    constexpr static const double V_max = 0.615;
    constexpr static const double ReferenceX = 100.0;  // equal energy reference (http://www.easyrgb.com/en/math.php)
    constexpr static const double ReferenceY = 100.0;
    constexpr static const double ReferenceZ = 100.0;
};

#endif  // !PARTITION_H
