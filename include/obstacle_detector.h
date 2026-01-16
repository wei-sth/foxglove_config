#ifndef OBSTACLE_DETECTOR_H
#define OBSTACLE_DETECTOR_H

#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/moment_of_inertia_estimation.h> // Include for pcl::MomentOfInertiaEstimation
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <string>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <iostream>

// for xt16, time is relative time, unit: ns 
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

// robosense airy, time is absolute time
struct RSPointDefault {
    PCL_ADD_POINT4D;
    float intensity;
    uint16_t ring;
    double timestamp;
    uint8_t feature;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(RSPointDefault,
    (float, x, x)(float, y, y)(float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (double, timestamp, timestamp)
    (uint8_t, feature, feature))

struct BoundingBox {
    pcl::PointXYZ min_point;
    pcl::PointXYZ max_point;
    pcl::PointXYZ center;
    float width, height, depth;
};

// bbox in lidar frame
struct RotatedBoundingBox {
    pcl::PointXYZ center;
    float size_x, size_y, size_z;
    Eigen::Quaternionf orientation;
    bool is_ground_aligned;    // true: perpendicular to ground plane; false: perpendicular to lidar XY plane
};

// Helper struct for 2D minimum area rectangle
struct MinAreaRect {
    Eigen::Vector2f center;
    float size_x;
    float size_y;
    float angle; // Angle in radians
};

struct Cell {
    std::vector<int> point_indices;
};

enum class VisResultType { BBOX_LIDAR_XY, BBOX_GROUND};

class WildTerrainSegmenter {
public:
    WildTerrainSegmenter(float max_range);

    // Parameter settings
    int num_rings = 20;       // Distance division
    int num_sectors = 36;     // Angle division (10 degrees per bin)
    float max_range;
    float sensor_height = 0.5; // Sensor installation height relative to ground plane
    float dist_threshold = 0.12; // if distance exceeds dist_threshold: obstacle, if within dist_threshold: ground
    float normal_z_threshold = 0.7f; // about 45 degree, if fitted plane abs(normal_z) < normal_z_threshold: obstacle
    int num_lpr = 10;         // Take the lowest 10 points as seed points each time
    int num_iter = 3;         // Number of fitting iterations

    int debug_r;
    int debug_s;

    void segment(const cv::Mat& range_image, const cv::Mat& x_image, const cv::Mat& y_image, const cv::Mat& z_image, const cv::Mat& valid_mask,
                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr& ground_cloud,
                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr& obstacle_cloud);

    void debugSavePolarGrid(const std::vector<std::vector<Cell>>& polar_grid, const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud_in, const std::string& path);

private:
    // Find the lowest set of points
    std::vector<int> extract_seeds(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, const std::vector<int>& indices);

    // Fit plane using PCA: ax + by + cz + d = 0
    void estimate_plane(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, const std::vector<int>& indices, 
                        Eigen::Vector3f& normal, float& d, float& linearity);
};

class RangeImageObstacleDetector {
public:
    RangeImageObstacleDetector(int num_rings = 16, int num_sectors = 1800, 
                               float max_range = 10.0f, float min_cluster_z_difference = 0.1f, VisResultType vis_type = VisResultType::BBOX_LIDAR_XY);
    
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> detectObstacles(pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw);
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> detectObstacles(pcl::PointCloud<RSPointDefault>::Ptr cloud_raw);
    
private:
    // these three params are not used, I provide transform matrix directly
    // look from +y to origin, get the angle from ground plane to lidar xy plane, get sensor_pitch (ccw is positive)
    float sensor_roll = 0.0;   // Sensor installation roll angle relative to ground plane
    float sensor_pitch = 0.523599;  // Sensor installation pitch angle relative to ground plane, 30 degree
    float sensor_yaw = 0.0;    // Sensor installation yaw angle relative to ground plane
    
    Eigen::Affine3f sensor_transform_ = Eigen::Affine3f::Identity();  // Lidar -> Ground
    Eigen::Affine3f sensor_inv_transform_ = Eigen::Affine3f::Identity();  // Ground -> Lidar
    bool apply_sensor_transform_ = false;

    void updateSensorTransform();

    int num_rings;
    int num_sectors;
    float max_range;
    float min_cluster_z_difference_;
    VisResultType vis_type_;
    
    cv::Mat range_image_;
    cv::Mat x_image_;
    cv::Mat y_image_;
    cv::Mat z_image_;
    cv::Mat valid_mask_;
    std::vector<const pcl::PointXYZINormal*> obstacle_grid_flat_;
    cv::Mat temp_valid_mask_;
    cv::Mat visited_mask_;
    
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr filterByRangeEgo(pcl::PointCloud<PointXYZIRT>::Ptr cloud);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr filterByRangeEgo(pcl::PointCloud<RSPointDefault>::Ptr cloud);
    void buildRangeImage(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr segmentGroundByNormal();
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr segmentGroundByPatchwork();
    Eigen::Vector3f computeNormal(int row, int col);
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clusterEuclidean(pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_with_normal_info);
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clusterConnectivity(pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_with_normal_info);
    bool computeClusterGeom(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster, MinAreaRect& mar, float& min_z_cluster, float& max_z_cluster);
    
    WildTerrainSegmenter wild_terrain_segmenter_;
    std::vector<RotatedBoundingBox> bboxes_lidar_frame_;  // bbox in lidar frame, depending on is_ground_aligned, it can be perpendicular to ground plane or lidar XY plane
public:
    const std::vector<RotatedBoundingBox>& getVisBBoxes() const { return bboxes_lidar_frame_; }
    void saveNormalsToPCD(const std::string& path);

    std::vector<RotatedBoundingBox> getObstacleBBoxesPCA(
        const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters);
    
    std::vector<RotatedBoundingBox> getObstacleBBoxes(
        const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters);

    std::vector<RotatedBoundingBox> getObstacleBBoxesFromGround(
        const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters);

    void saveRotatedBoundingBoxesToObj(
        const std::vector<RotatedBoundingBox>& rotated_bboxes,
        const std::string& file_path);

    // Helper function for rotating calipers
    MinAreaRect findMinAreaRect(const pcl::PointCloud<pcl::PointXYZI>::Ptr& hull_points_2d);
};

#endif // OBSTACLE_DETECTOR_H
