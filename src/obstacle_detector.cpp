#include "/home/weizh/foxglove_ws/src/foxglove_config/include/obstacle_detector.h"

#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <algorithm>

RangeImageObstacleDetector::RangeImageObstacleDetector(int num_rings, int num_sectors, 
                               float max_distance, float min_cluster_z_difference)
    : num_rings(num_rings), num_sectors(num_sectors), max_distance(max_distance),
      min_cluster_z_difference_(min_cluster_z_difference) {
    
    range_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, 
                           cv::Scalar(std::numeric_limits<float>::max()));
    x_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    y_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    z_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    
    valid_mask_ = cv::Mat(num_rings, num_sectors, CV_8UC1, cv::Scalar(0));
}

pcl::PointCloud<pcl::PointXYZI>::Ptr RangeImageObstacleDetector::detectObstacles(
    pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw) {
    
    // Step 1: Filter by distance and convert to PointXYZINormal
    auto cloud_filtered_normal = filterByDistance(cloud_raw);
    
    // Step 2: Build Range Image using PointXYZINormal
    buildRangeImage(cloud_filtered_normal);
    
    // Step 3: Segment ground by normal and get obstacles with ring info in normal_x
    auto obstacles_with_normal_info = segmentGroundByNormal();
    
    // Step 4: Perform clustering on the detected obstacles (PointXYZINormal)
    pcl::PointCloud<pcl::PointXYZI>::Ptr clustered_obstacles(new pcl::PointCloud<pcl::PointXYZI>);
    
    if (obstacles_with_normal_info->points.empty()) {
        return clustered_obstacles;
    }

    float current_intensity = 0.0f; // Initialize intensity for clusters

    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZINormal>);
    tree->setInputCloud(obstacles_with_normal_info);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZINormal> ec;
    ec.setClusterTolerance(0.3); // 30cm
    ec.setMinClusterSize(20);
    ec.setMaxClusterSize(10000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(obstacles_with_normal_info);
    ec.extract(cluster_indices);

    for (const auto& indices : cluster_indices) {
        std::set<uint16_t> rings_in_cluster;
        float min_z_cluster = std::numeric_limits<float>::max();
        float max_z_cluster = -std::numeric_limits<float>::max();
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr current_cluster_xyzi(new pcl::PointCloud<pcl::PointXYZI>);

        for (int idx : indices.indices) {
            const auto& pt = obstacles_with_normal_info->points[idx];
            rings_in_cluster.insert(static_cast<uint16_t>(pt.normal_x)); // Retrieve ring from normal_x
            
            min_z_cluster = std::min(min_z_cluster, pt.z);
            max_z_cluster = std::max(max_z_cluster, pt.z);

            pcl::PointXYZI xyzi_pt;
            xyzi_pt.x = pt.x;
            xyzi_pt.y = pt.y;
            xyzi_pt.z = pt.z;
            xyzi_pt.intensity = current_intensity; // Assign unique intensity
            current_cluster_xyzi->points.push_back(xyzi_pt);
        }
        
        // Filter clusters based on the number of rings spanned AND Z-difference
        if (rings_in_cluster.size() >= 2 && (max_z_cluster - min_z_cluster) > min_cluster_z_difference_) {
            *clustered_obstacles += *current_cluster_xyzi;
            current_intensity += 10.0f; // Increment intensity for the next cluster
        }
    }
    
    return clustered_obstacles;
}

pcl::PointCloud<pcl::PointXYZINormal>::Ptr RangeImageObstacleDetector::filterByDistance(pcl::PointCloud<PointXYZIRT>::Ptr cloud) {
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr filtered_normal(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    
    for (const auto& pt_irt : cloud->points) {
        float distance = std::sqrt(pt_irt.x * pt_irt.x + pt_irt.y * pt_irt.y + pt_irt.z * pt_irt.z);
        
        if (distance <= max_distance && distance > 0.5f) {  // Min distance 0.5m
            pcl::PointXYZINormal pt_normal;
            pt_normal.x = pt_irt.x;
            pt_normal.y = pt_irt.y;
            pt_normal.z = pt_irt.z;
            pt_normal.intensity = pt_irt.intensity;
            pt_normal.normal_x = static_cast<float>(pt_irt.ring); // Store ring in normal_x
            pt_normal.normal_y = pt_irt.time; // Store time in normal_y
            // normal_z, curvature can be left as default or set to 0
            
            filtered_normal->points.push_back(pt_normal);
        }
    }
    
    return filtered_normal;
}

void RangeImageObstacleDetector::buildRangeImage(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud) {
    
    range_image_.setTo(std::numeric_limits<float>::max());
    x_image_.setTo(0);
    y_image_.setTo(0);
    z_image_.setTo(0);
    valid_mask_.setTo(0);
    
    for (const auto& pt : cloud->points) {
        uint16_t ring = static_cast<uint16_t>(pt.normal_x); // Retrieve ring from normal_x
        if (ring >= num_rings) continue;
        
        float azimuth = std::atan2(pt.y, pt.x);
        if (azimuth < 0) azimuth += 2 * M_PI;
        
        int col = static_cast<int>(azimuth / (2 * M_PI) * num_sectors);
        col = std::min(col, num_sectors - 1);
        
        int row = ring;
        
        float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        
        if (range < range_image_.at<float>(row, col)) {
            range_image_.at<float>(row, col) = range;
            x_image_.at<float>(row, col) = pt.x;
            y_image_.at<float>(row, col) = pt.y;
            z_image_.at<float>(row, col) = pt.z;
            valid_mask_.at<uint8_t>(row, col) = 1;
        }
    }
}

pcl::PointCloud<pcl::PointXYZINormal>::Ptr RangeImageObstacleDetector::segmentGroundByNormal() {
    
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_with_normal_info(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    
    for (int row = 1; row < num_rings - 1; ++row) {
        for (int col = 1; col < num_sectors - 1; ++col) {
            
            if (!valid_mask_.at<uint8_t>(row, col)) continue;
            
            float x = x_image_.at<float>(row, col);
            float y = y_image_.at<float>(row, col);
            float z = z_image_.at<float>(row, col);
            
            Eigen::Vector3f normal = computeNormal(row, col);
            
            float normal_z_threshold = 0.7f;
            
            if (std::abs(normal.z()) < normal_z_threshold) {
                pcl::PointXYZINormal obstacle_pt;
                obstacle_pt.x = x;
                obstacle_pt.y = y;
                obstacle_pt.z = z;
                obstacle_pt.normal_x = static_cast<float>(row); // Store ring in normal_x
                // Other normal_y, normal_z, intensity, curvature can be set as needed
                obstacles_with_normal_info->points.push_back(obstacle_pt);
            }
        }
    }
    
    return obstacles_with_normal_info;
}

Eigen::Vector3f RangeImageObstacleDetector::computeNormal(int row, int col) {
    
    int col_right = (col + 1) % num_sectors;
    int row_down = row + 1;
    
    if (!valid_mask_.at<uint8_t>(row, col_right) || 
        !valid_mask_.at<uint8_t>(row_down, col)) {
        return Eigen::Vector3f(0, 0, 1);
    }
    
    Eigen::Vector3f p(x_image_.at<float>(row, col),
                     y_image_.at<float>(row, col),
                     z_image_.at<float>(row, col));
    
    Eigen::Vector3f p_right(x_image_.at<float>(row, col_right),
                           y_image_.at<float>(row, col_right),
                           z_image_.at<float>(row, col_right));
    
    Eigen::Vector3f p_down(x_image_.at<float>(row_down, col),
                          y_image_.at<float>(row_down, col),
                          z_image_.at<float>(row_down, col));
    
    Eigen::Vector3f v1 = p_right - p;
    Eigen::Vector3f v2 = p_down - p;
    
    if (v1.norm() > 0.5f || v2.norm() > 0.5f) {
        return Eigen::Vector3f(0, 0, 1);
    }
    
    Eigen::Vector3f normal = v1.cross(v2);
    
    if (normal.norm() > 1e-6) {
        normal.normalize();
    } else {
        return Eigen::Vector3f(0, 0, 1);
    }
    
    if (normal.z() < 0) {
        normal = -normal;
    }
    
    return normal;
}

cv::Mat RangeImageObstacleDetector::visualizeRangeImage() {
    cv::Mat vis_image;
    cv::normalize(range_image_, vis_image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    for (int i = 0; i < num_rings; ++i) {
        for (int j = 0; j < num_sectors; ++j) {
            if (!valid_mask_.at<uint8_t>(i, j)) {
                vis_image.at<uint8_t>(i, j) = 0;
            }
        }
    }
    
    cv::Mat colored;
    cv::applyColorMap(vis_image, colored, cv::COLORMAP_JET);
    return colored;
}

cv::Mat RangeImageObstacleDetector::visualizeNormals() {
    cv::Mat normal_z_image(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    
    for (int row = 1; row < num_rings - 1; ++row) {
        for (int col = 1; col < num_sectors - 1; ++col) {
            if (!valid_mask_.at<uint8_t>(row, col)) continue;
            
            Eigen::Vector3f normal = computeNormal(row, col);
            normal_z_image.at<float>(row, col) = std::abs(normal.z());
        }
    }
    
    cv::Mat vis_image;
    cv::normalize(normal_z_image, vis_image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    cv::Mat colored;
    cv::applyColorMap(vis_image, colored, cv::COLORMAP_JET);
    return colored;
}

std::vector<BoundingBox> getObstacleBoundingBoxes(
    pcl::PointCloud<pcl::PointXYZI>::Ptr obstacles) {
    
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(obstacles);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(0.3);  // 30cm cluster distance
    ec.setMinClusterSize(20);     // Min 20 points
    ec.setMaxClusterSize(10000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(obstacles);
    ec.extract(cluster_indices);
    
    std::vector<BoundingBox> bboxes;
    for (const auto& indices : cluster_indices) {
        pcl::PointXYZ min_pt, max_pt;
        
        min_pt.x = min_pt.y = min_pt.z = std::numeric_limits<float>::max();
        max_pt.x = max_pt.y = max_pt.z = -std::numeric_limits<float>::max();
        
        for (int idx : indices.indices) {
            const auto& pt = obstacles->points[idx];
            min_pt.x = std::min(min_pt.x, pt.x);
            min_pt.y = std::min(min_pt.y, pt.y);
            min_pt.z = std::min(min_pt.z, pt.z);
            max_pt.x = std::max(max_pt.x, pt.x);
            max_pt.y = std::max(max_pt.y, pt.y);
            max_pt.z = std::max(max_pt.z, pt.z);
        }
        
        BoundingBox bbox;
        bbox.min_point = min_pt;
        bbox.max_point = max_pt;
        bbox.center.x = (min_pt.x + max_pt.x) / 2;
        bbox.center.y = (min_pt.y + max_pt.y) / 2;
        bbox.center.z = (min_pt.z + max_pt.z) / 2;
        
        float volume = (max_pt.x - min_pt.x) * 
                      (max_pt.y - min_pt.y) * 
                      (max_pt.z - min_pt.z);
        
        if (volume > 0.001) {  // Minimum volume threshold
            bboxes.push_back(bbox);
        }
    }
    
    return bboxes;
}
