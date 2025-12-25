#ifndef OBSTACLE_DETECTOR_H
#define OBSTACLE_DETECTOR_H

#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <numeric> // For std::accumulate, std::sqrt
#include <Eigen/Dense> // For Eigen matrices and vectors

// Original PointXYZIRT definition for reading the PCD file and ROS message
struct PointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (float, time, time))

// Local PointXYZ structure to hold additional information for PatchWork++ processing
struct PointXYZ_Local {
  float x;
  float y;
  float z;
  float intensity;
  uint16_t ring;
  int original_idx; // To map back to the original cloud_raw

  PointXYZ_Local(float _x, float _y, float _z, float _intensity, uint16_t _ring, int _idx = -1)
      : x(_x), y(_y), z(_z), intensity(_intensity), ring(_ring), original_idx(_idx) {}
};

struct BoundingBox {
    pcl::PointXYZ min_point;
    pcl::PointXYZ max_point;
    pcl::PointXYZ center;
    float width, height, depth;
};

class RangeImageObstacleDetector {
public:
    // Concentric Zone Model (CZM) typedefs moved to public
    typedef std::vector<PointXYZ_Local> RingSector;
    typedef std::vector<RingSector> RingPatches;
    typedef std::vector<RingPatches> ZonePatches;

    RangeImageObstacleDetector(int num_rings = 16, int num_sectors = 1800, 
                               float max_distance = 10.0f, float min_cluster_z_difference = 0.1f);
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr detectObstacles(
        pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw);
    
private:
    int num_rings;
    int num_sectors;
    float max_distance;
    float min_cluster_z_difference_;
    
    // Members for Range Image (retained for saveNormalsToPCD)
    cv::Mat range_image_;
    cv::Mat x_image_;
    cv::Mat y_image_;
    cv::Mat z_image_;
    cv::Mat valid_mask_;
    
    // PatchWork++ inspired parameters
    bool verbose_ = false;
    // enable reflected_noise_removal, false, Wei
    bool enable_RNR_ = false;
    bool enable_RVPF_ = true; // Region-wise Vertical Plane Fitting
    
    int num_iter_ = 3;
    int num_lpr_ = 20;
    int num_min_pts_ = 10;
    int num_zones_ = 4;
    int num_rings_of_interest_ = 4;

    double RNR_ver_angle_thr_ = -15.0;
    double RNR_intensity_thr_ = 0.2;

    // double sensor_height_ = 1.723;
    double sensor_height_ = 0.4;
    double th_seeds_ = 0.125;
    double th_dist_ = 0.125;
    double th_seeds_v_ = 0.25;
    double th_dist_v_ = 0.1;
    // max_range_ and min_range_ are already part of RangeImageObstacleDetector's max_distance
    double uprightness_thr_ = 0.707;
    double adaptive_seed_selection_margin_ = -1.2;

    std::vector<int> num_sectors_each_zone_ = {16, 32, 54, 32};
    std::vector<int> num_rings_each_zone_ = {2, 4, 4, 4};

    int max_flatness_storage_ = 1000;
    int max_elevation_storage_ = 1000;

    std::vector<double> elevation_thr_ = {0, 0, 0, 0};
    std::vector<double> flatness_thr_ = {0, 0, 0, 0};

    uint16_t min_ground_ring_index_ = 8; // Rings 0-7 are upwards, 8-15 are downwards. Ground points should be from ring 8 or higher.

    // PatchWork++ internal state variables
    std::vector<std::vector<double>> update_flatness_; // [num_zones]
    std::vector<std::vector<double>> update_elevation_; // [num_zones]

    double d_; // Plane equation parameter
    Eigen::VectorXf normal_; // Plane normal
    Eigen::VectorXf singular_values_; // Singular values from PCA
    Eigen::VectorXf pc_mean_; // Mean of point cloud for plane estimation

    std::vector<double> min_ranges_;
    std::vector<double> sector_sizes_;
    std::vector<double> ring_sizes_;

    std::vector<ZonePatches> czm_patches_;

    // Result point clouds
    std::vector<PointXYZ_Local> ground_pc_local_, non_ground_pc_local_;
    std::vector<PointXYZ_Local> regionwise_ground_local_, regionwise_nonground_local_;
    std::vector<PointXYZ_Local> cloud_ground_local_, cloud_nonground_local_;
    std::vector<PointXYZ_Local> rnr_noise_points_; // Store points identified as RNR noise

    // Helper functions (from original RangeImageObstacleDetector)
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr filterByDistance(pcl::PointCloud<PointXYZIRT>::Ptr cloud);
    void buildRangeImage(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);
    Eigen::Vector3f computeNormal(int row, int col); // Retained for saveNormalsToPCD

    // PatchWork++ inspired helper functions
    void addCloud(std::vector<PointXYZ_Local> &cloud, const std::vector<PointXYZ_Local> &add);
    void flush_patches(std::vector<ZonePatches> &czm);
    void pc2czm(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &src, std::vector<ZonePatches> &czm);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr reflected_noise_removal(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_in);
    
    double calc_point_to_plane_d(PointXYZ_Local p, Eigen::VectorXf normal, double d);
    void calc_mean_stdev(const std::vector<double> &vec, double &mean, double &stdev);

    void update_elevation_thr();
    void update_flatness_thr();

    double xy2theta(const double &x, const double &y);
    double xy2radius(const double &x, const double &y);

    void estimate_plane(const std::vector<PointXYZ_Local> &ground);

    void extract_piecewiseground(const int zone_idx,
                               const std::vector<PointXYZ_Local> &src,
                               std::vector<PointXYZ_Local> &dst,
                               std::vector<PointXYZ_Local> &non_ground_dst);

    void extract_initial_seeds(const int zone_idx,
                             const std::vector<PointXYZ_Local> &p_sorted,
                             std::vector<PointXYZ_Local> &init_seeds);

    void extract_initial_seeds(const int zone_idx,
                             const std::vector<PointXYZ_Local> &p_sorted,
                             std::vector<PointXYZ_Local> &init_seeds,
                             double th_seed);
    
public:
    // Retained for external use
    void saveNormalsToPCD(const std::string& path);

    // New function to save ground point cloud
    void saveGroundToPCD(const std::string& path);

    // New function to save non-ground points before clustering
    void saveNongroundBeforeClusteringToPCD(const std::string& path);

    // New function to save RNR skipped points
    void saveRNRSkippedPointsToPCD(const std::string& path);
};

std::vector<BoundingBox> getObstacleBoundingBoxes(
    pcl::PointCloud<pcl::PointXYZI>::Ptr obstacles);

#endif // OBSTACLE_DETECTOR_H
