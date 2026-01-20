#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <Eigen/Geometry>
#include "obstacle_detector.h"
#include <cmath>

// Get Lidar extrinsic parameters by fitting a plane to ground points.
// Input PCD file should only contain xyz since cloudcompare changes scalar field type. In cloudcompare, use Edit -> Scalar fields -> Delete all.
void getLidarExtrinsic(const std::string& pcd_file_path) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path, *cloud) == -1) {
        PCL_ERROR("Couldn't read file %s \n", pcd_file_path.c_str());
        return;
    }

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.05);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
        return;
    }

    // visualization, extract inliers and save to a new PCD file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_inliers->width = inliers->indices.size();
    cloud_inliers->height = 1;
    cloud_inliers->is_dense = true;
    cloud_inliers->points.resize(inliers->indices.size());
    for (size_t i = 0; i < inliers->indices.size(); ++i) {
        cloud_inliers->points[i] = cloud->points[inliers->indices[i]];
    }

    std::string inlier_pcd_path = pcd_file_path;
    size_t last_dot = inlier_pcd_path.find_last_of(".");
    if (last_dot != std::string::npos) {
        inlier_pcd_path.insert(last_dot, "_inlier");
    } else {
        inlier_pcd_path += "_inlier.pcd";
    }
    pcl::io::savePCDFileBinary(inlier_pcd_path, *cloud_inliers);
    std::cout << "Inlier points saved to " << inlier_pcd_path << std::endl;

    float a = coefficients->values[0];
    float b = coefficients->values[1];
    float c = coefficients->values[2];
    float d = coefficients->values[3];

    // Normalize the plane equation ax + by + cz + d = 0
    float norm = std::sqrt(a*a + b*b + c*c);
    a /= norm;
    b /= norm;
    c /= norm;
    d /= norm;

    // Ensure the normal points upwards (positive z)
    if (c < 0) {
        a = -a;
        b = -b;
        c = -c;
        d = -d;
    }

    std::cout << "--- Lidar Extrinsic Calibration ---" << std::endl;
    std::cout << "Plane equation: " << a << "x + " << b << "y + " << c << "z + " << d << " = 0" << std::endl;
    std::cout << "Distance from (0,0,0) to ground plane: " << std::abs(d) << " meters" << std::endl;

    // Construct the transformation matrix from Lidar to Ground
    // The ground plane in the new coordinate system will be z = 0
    Eigen::Vector3f z_new(a, b, c);
    Eigen::Vector3f x_temp(1, 0, 0);
    if (std::abs(z_new.dot(x_temp)) > 0.9) {
        x_temp = Eigen::Vector3f(0, 1, 0);
    }
    Eigen::Vector3f y_new = z_new.cross(x_temp).normalized();
    Eigen::Vector3f x_new = y_new.cross(z_new).normalized();

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    // Rotation part: rows are the new basis vectors
    transform.block<1, 3>(0, 0) = x_new.transpose();
    transform.block<1, 3>(1, 0) = y_new.transpose();
    transform.block<1, 3>(2, 0) = z_new.transpose();
    // Translation part: moves the plane to z = 0
    transform(2, 3) = d;

    std::cout << "Transformation Matrix (Lidar -> Ground):" << std::endl;
    std::cout << transform << std::endl;
    std::cout << "-----------------------------------" << std::endl;
}

int main(int argc, char * argv[]) {
    // xt16
    // int num_rings = 16;
    // int num_sectors = 2000;

    // robosense airy 96
    int num_rings = 96;
    int num_sectors = 900;
    float max_distance = 10.0f; // Aligned with obstacle_detector_node.cpp
    float min_cluster_z_difference = 0.2f; // Aligned with obstacle_detector_node.cpp
    VisResultType vis_type = VisResultType::BBOX_GROUND; // VisResultType::BBOX_GROUND | VisResultType::BBOX_LIDAR_XY | VisResultType::BBOX_GROUND_2D

    RangeImageObstacleDetector detector(num_rings, num_sectors, max_distance, min_cluster_z_difference, vis_type);
    pcl::PointCloud<RSPointDefault>::Ptr rs_cloud_raw(new pcl::PointCloud<RSPointDefault>);
    std::string rs_pcd_file_path = "/home/weizh/data/rosbag2_2026_01_20-15_09_01/rosbag2_2026_01_20-15_09_01_0_logs/rslidar_points/1768892941_704764128.pcd";
    if (pcl::io::loadPCDFile<RSPointDefault>(rs_pcd_file_path, *rs_cloud_raw) == -1) {
        PCL_ERROR("Couldn't read file %s \n", rs_pcd_file_path.c_str());
        return (-1);
    }
    std::cout << "Loaded " << rs_cloud_raw->width * rs_cloud_raw->height << " data points from " << rs_pcd_file_path << std::endl;
    
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> obstacle_clusters = detector.detectObstacles(rs_cloud_raw);

    // Save all valid points with their computed normals to a PCD file for 3D visualization
    std::string normals_pcd_path = "/home/weizh/data/range_image_normals.pcd";
    detector.saveNormalsToPCD(normals_pcd_path);
    std::cout << "Range image normals saved to " << normals_pcd_path << std::endl;

    if (vis_type == VisResultType::BBOX_GROUND) {
        std::vector<RotatedBoundingBox> rotated_bboxes = detector.getVisBBoxes();
        std::cout << "Detected " << rotated_bboxes.size() << " rotated bounding boxes:" << std::endl;
        
        // Save bounding boxes to OBJ file for CloudCompare visualization
        std::string bbox_obj_path = "/home/weizh/data/obstacle_bboxes.obj";
        detector.saveRotatedBoundingBoxesToObj(rotated_bboxes, bbox_obj_path); // Now generates MTL with transparency
        std::cout << "Rotated bounding boxes saved to " << bbox_obj_path << " (with transparency via MTL)" << std::endl;
    }
    else if (vis_type == VisResultType::BBOX_GROUND_2D) {
        std::vector<ObstacleBBox2D> rotated_bboxes_2d = detector.getVisBBoxes2D();
        std::cout << "Detected " << rotated_bboxes_2d.size() << " rotated bounding boxes:" << std::endl;

        std::string bbox_json_path = "/home/weizh/data/obstacle_bboxes.json";
        detector.saveObstacleBBoxes2DToJson(rotated_bboxes_2d, bbox_json_path); // Now generates MTL with transparency
        std::cout << "Rotated bounding boxes saved to " << bbox_json_path << std::endl;
    }
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


    // calibration
    getLidarExtrinsic("/home/weizh/data/1768892940_599927902_ground.pcd");

    return 0;
}

// --- Lidar Extrinsic Calibration ---
// Plane equation: -0.468943x + -0.0350392y + 0.882533z + 0.612313 = 0
// Distance from (0,0,0) to ground plane: 0.612313 meters
// Transformation Matrix (Lidar -> Ground):
//   0.883228 -0.0186038   0.468574          0
//         -0   0.999213  0.0396717          0
//  -0.468943 -0.0350392   0.882533   0.612313
//          0          0          0          1
// -----------------------------------