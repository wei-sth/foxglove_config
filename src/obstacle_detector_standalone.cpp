#include <pcl/io/pcd_io.h>
#include "/home/weizh/foxglove_ws/src/foxglove_config/include/obstacle_detector.h"

int main(int argc, char * argv[]) {
    // PCD file processing mode
    int num_rings = 16;
    int num_sectors = 2000;
    float max_distance = 10.0f;
    float min_cluster_z_difference = 0.2f;

    RangeImageObstacleDetector detector(num_rings, num_sectors, max_distance, min_cluster_z_difference);
    
    pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw(
        new pcl::PointCloud<PointXYZIRT>);
    std::string pcd_file_path = "/home/weizh/data/bag_11261/bag_11261_0_logs/unitree_slam_lidar_points/1764126218_844803835.pcd";

    if (pcl::io::loadPCDFile<PointXYZIRT>(pcd_file_path, *cloud_raw) == -1) {
        PCL_ERROR("Couldn't read file %s \n", pcd_file_path.c_str());
        return (-1);
    }
    std::cout << "Loaded " << cloud_raw->width * cloud_raw->height
                << " data points from " << pcd_file_path << std::endl;
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr obstacles = detector.detectObstacles(cloud_raw);
    std::vector<BoundingBox> bboxes = getObstacleBoundingBoxes(obstacles);
    
    std::cout << "Detected " << bboxes.size() << " bounding boxes:" << std::endl;
    for (const auto& bbox : bboxes) {
        std::cout << "  Center: (" << bbox.center.x << ", " << bbox.center.y << ", " << bbox.center.z << ")"
                    << "  Min: (" << bbox.min_point.x << ", " << bbox.min_point.y << ", " << bbox.min_point.z << ")"
                    << "  Max: (" << bbox.max_point.x << ", " << bbox.max_point.y << ", " << bbox.max_point.z << ")"
                    << std::endl;
    }

    // Optionally save obstacles to PCD for visualization
    std::string output_pcd_path = "/home/weizh/data/obstacles.pcd";
    obstacles->width = obstacles->points.size();
    obstacles->height = 1;
    obstacles->is_dense = true;
    pcl::io::savePCDFileBinary(output_pcd_path, *obstacles);
    std::cout << "Detected obstacles saved to " << output_pcd_path << std::endl;

    return 0;
}