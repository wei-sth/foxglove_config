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

struct BoundingBox {
    pcl::PointXYZ min_point;
    pcl::PointXYZ max_point;
    pcl::PointXYZ center;
    float width, height, depth;
};

struct RotatedBoundingBox {
    pcl::PointXYZ center;
    float width, height, angle; // angle in radians
    pcl::PointXYZ min_z_point; // Store min_z for the cluster
    pcl::PointXYZ max_z_point; // Store max_z for the cluster
};

class RangeImageObstacleDetector {
public:
    RangeImageObstacleDetector(int num_rings = 16, int num_sectors = 1800, 
                               float max_distance = 10.0f, float min_cluster_z_difference = 0.1f);
    
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> detectObstacles(pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw);
    
private:
    int num_rings;
    int num_sectors;
    float max_distance;
    float min_cluster_z_difference_;
    
    cv::Mat range_image_;
    cv::Mat x_image_;
    cv::Mat y_image_;
    cv::Mat z_image_;
    cv::Mat valid_mask_;
    std::vector<const pcl::PointXYZINormal*> obstacle_grid_flat_;
    cv::Mat temp_valid_mask_;
    cv::Mat visited_mask_;
    
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr filterByDistance(pcl::PointCloud<PointXYZIRT>::Ptr cloud);
    void buildRangeImage(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr segmentGroundByNormal();
    Eigen::Vector3f computeNormal(int row, int col);
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clusterEuclidean(pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_with_normal_info);
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clusterConnectivity(pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_with_normal_info);
    
public:
    cv::Mat visualizeRangeImage();
    void visualizeNormals(const std::string& path);
    void saveNormalsToPCD(const std::string& path);
    std::vector<BoundingBox> getObstacleBoundingBoxes(
        pcl::PointCloud<pcl::PointXYZI>::Ptr obstacles);

    std::vector<RotatedBoundingBox> getObstacleBoundingBoxesNew(
        const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters);
    
    void saveRotatedBoundingBoxesToObj(
        const std::vector<RotatedBoundingBox>& rotated_bboxes,
        const std::string& file_path);
};

#endif // OBSTACLE_DETECTOR_H
