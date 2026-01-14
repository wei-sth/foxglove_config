#include <pcl/io/pcd_io.h>
#include "/home/weizh/foxglove_ws/src/foxglove_config/include/obstacle_detector.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Load a PointXYZI PCD file and convert it to PointXYZIRT.
 *        Ring and time are calculated based on vertical and horizontal angles.
 * 
 * @param pcd_file_path Path to the input PCD file.
 * @return pcl::PointCloud<PointXYZIRT>::Ptr The converted point cloud.
 */
pcl::PointCloud<PointXYZIRT>::Ptr loadPCDAsIRT(const std::string& pcd_file_path) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_i(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_path, *cloud_i) == -1) {
        PCL_ERROR("Couldn't read file %s \n", pcd_file_path.c_str());
        return nullptr;
    }

    pcl::PointCloud<PointXYZIRT>::Ptr cloud_irt(new pcl::PointCloud<PointXYZIRT>);
    cloud_irt->header = cloud_i->header;
    cloud_irt->width = cloud_i->width;
    cloud_irt->height = cloud_i->height;
    cloud_irt->is_dense = cloud_i->is_dense;
    cloud_irt->points.resize(cloud_i->points.size());

    for (size_t i = 0; i < cloud_i->points.size(); ++i) {
        const auto& pt_i = cloud_i->points[i];
        auto& pt_irt = cloud_irt->points[i];

        pt_irt.x = pt_i.x;
        pt_irt.y = pt_i.y;
        pt_irt.z = pt_i.z;
        pt_irt.intensity = pt_i.intensity;

        // Calculate vertical angle to determine ring index
        float v_angle_deg = std::atan2(pt_i.z, std::sqrt(pt_i.x * pt_i.x + pt_i.y * pt_i.y)) * 180.0 / M_PI;
        
        // Map vertical angle to ring index for RoboSense Airy (96 lines)
        // Adjusted range to [1.5, 34.0] degrees (total 32.5 degrees) based on data analysis.
        // This ensures the minimum ring starts at 0 and maximizes vertical resolution.
        int ring = std::floor((v_angle_deg - 1.5) * (96.0 / 32.5));
        if (ring < 0) ring = 0;
        if (ring >= 96) ring = 95;
        pt_irt.ring = static_cast<uint16_t>(ring);

        // Calculate horizontal angle (azimuth) for time
        float h_angle = std::atan2(pt_i.y, pt_i.x);
        // Map azimuth [-PI, PI] to [0, 1] as a proxy for time within one scan
        pt_irt.time = (h_angle + M_PI) / (2.0 * M_PI);
    }
    return cloud_irt;
}

int main(int argc, char * argv[]) {
    // PCD file processing mode
    int num_rings = 96;
    int num_sectors = 2000;
    float max_distance = 55.0f; // Aligned with obstacle_detector_node.cpp
    float min_cluster_z_difference = 0.2f; // Aligned with obstacle_detector_node.cpp

    RangeImageObstacleDetector detector(num_rings, num_sectors, max_distance, min_cluster_z_difference);
    
    pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw(
        new pcl::PointCloud<PointXYZIRT>);
    // std::string pcd_file_path = "/home/weizh/data/bag_11261/bag_11261_0_logs/unitree_slam_lidar_points/1764126218_844803835.pcd";
    // 1764126140_046829002
    // 1764126144_944283160
    // 1764126158_346454512
    std::string pcd_file_path = "/home/weizh/data/bag_11261/bag_11261_0_logs/unitree_slam_lidar_points/1764126158_346454512.pcd";

    if (pcl::io::loadPCDFile<PointXYZIRT>(pcd_file_path, *cloud_raw) == -1) {
        PCL_ERROR("Couldn't read file %s \n", pcd_file_path.c_str());
        return (-1);
    }
    std::cout << "Loaded " << cloud_raw->width * cloud_raw->height
                << " data points from " << pcd_file_path << std::endl;

    // Test the new function: load PointXYZI and convert to PointXYZIRT
    std::string pcd_file_path_rs = "/home/weizh/data/bag_2026_0109/rosbag2_2026_01_08-18_38_26_0_logs/rslidar_points/755_603190160.pcd";
    
    pcl::PointCloud<PointXYZIRT>::Ptr cloud_converted = loadPCDAsIRT(pcd_file_path_rs);
    if (cloud_converted) {
        std::string converted_pcd_path = "/home/weizh/data/converted_irt_0113.pcd";
        pcl::io::savePCDFileBinary(converted_pcd_path, *cloud_converted);
        std::cout << "Converted PointXYZIRT saved to " << converted_pcd_path << std::endl;
    }

    // return 0;
    
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> obstacle_clusters = detector.detectObstacles(cloud_converted);

    // Save all valid points with their computed normals to a PCD file for 3D visualization
    std::string normals_pcd_path = "/home/weizh/data/range_image_normals.pcd";
    detector.saveNormalsToPCD(normals_pcd_path);
    std::cout << "Range image normals saved to " << normals_pcd_path << std::endl;

    // for patchwork++
    // std::string ground_pcd_path = "/home/weizh/data/range_image_ground.pcd";
    // detector.saveGroundToPCD(ground_pcd_path);
    // std::cout << "Range image ground saved to " << ground_pcd_path << std::endl;

    // std::string nonground_pcd_path = "/home/weizh/data/range_image_nonground.pcd";
    // detector.saveNongroundBeforeClusteringToPCD(nonground_pcd_path);
    // std::cout << "Range image nonground saved to " << nonground_pcd_path << std::endl;

    std::vector<RotatedBoundingBox> rotated_bboxes = detector.getObstacleBoundingBoxesNewV2(obstacle_clusters);
    std::cout << "Detected " << rotated_bboxes.size() << " rotated bounding boxes:" << std::endl;
    
    // Save bounding boxes to OBJ file for CloudCompare visualization
    std::string bbox_obj_path = "/home/weizh/data/obstacle_bboxes.obj";
    detector.saveRotatedBoundingBoxesToObj(rotated_bboxes, bbox_obj_path); // Now generates MTL with transparency
    std::cout << "Rotated bounding boxes saved to " << bbox_obj_path << " (with transparency via MTL)" << std::endl;

    // visualization
    pcl::PointCloud<pcl::PointXYZI>::Ptr all_obstacles(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto& cluster : obstacle_clusters) {
        *all_obstacles += *cluster;
    }
    if (all_obstacles->empty()) {
        std::cout << "no obstacle after clustering" << std::endl;
    } else {
        std::string output_pcd_path = "/home/weizh/data/obstacles.pcd";
        all_obstacles->width = all_obstacles->points.size();
        all_obstacles->height = 1;
        all_obstacles->is_dense = true;
        pcl::io::savePCDFileBinary(output_pcd_path, *all_obstacles);
        std::cout << "Detected obstacles saved to " << output_pcd_path << std::endl;
    }

    return 0;
}
