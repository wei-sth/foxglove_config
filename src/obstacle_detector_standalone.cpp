#include <pcl/io/pcd_io.h>
#include "/home/weizh/foxglove_ws/src/foxglove_config/include/obstacle_detector.h"

int main(int argc, char * argv[]) {
    // PCD file processing mode
    int num_rings = 16;
    int num_sectors = 2000;
    float max_distance = 15.0f; // Aligned with obstacle_detector_node.cpp
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
    
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> obstacle_clusters = detector.detectObstacles(cloud_raw);

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
    // for (const auto& rbbox : rotated_bboxes) {
    //     std::cout << "  Center: (" << rbbox.center.x << ", " << rbbox.center.y << ", " << rbbox.center.z << ")"
    //                 << "  Width: " << rbbox.width << ", Height: " << rbbox.height << ", Angle: " << rbbox.angle << std::endl;
    // }

    // Save bounding boxes to OBJ file for CloudCompare visualization
    std::string bbox_obj_path = "/home/weizh/data/obstacle_bboxes.obj";
    detector.saveRotatedBoundingBoxesToObj(rotated_bboxes, bbox_obj_path); // Now generates MTL with transparency
    std::cout << "Rotated bounding boxes saved to " << bbox_obj_path << " (with transparency via MTL)" << std::endl;

    // Optionally save obstacles to PCD for visualization
    // Concatenate all clusters into a single point cloud for saving
    pcl::PointCloud<pcl::PointXYZI>::Ptr all_obstacles(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto& cluster : obstacle_clusters) {
        *all_obstacles += *cluster;
    }

    std::string output_pcd_path = "/home/weizh/data/obstacles.pcd";
    all_obstacles->width = all_obstacles->points.size();
    all_obstacles->height = 1;
    all_obstacles->is_dense = true;
    pcl::io::savePCDFileBinary(output_pcd_path, *all_obstacles);
    std::cout << "Detected obstacles saved to " << output_pcd_path << std::endl;

    return 0;
}
