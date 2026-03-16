#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <nlohmann/json.hpp>
#include <Eigen/Geometry>
#include "obstacle_detector.h"
#include <cmath>
#include <pcl/features/normal_3d.h>
#include <numeric>

enum class ClusterType { HUMAN, GRASS, UNKNOWN };

using json = nlohmann::json;

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

void detect_from_pcd() {
    // xt16
    // int num_rings = 16;
    // int num_sectors = 2000;

    // robosense airy 96
    int num_rings = 96;
    int num_sectors = 900;
    float max_distance = 10.0f; // Aligned with obstacle_detector_node.cpp
    float min_cluster_z_difference = 0.1f; // Aligned with obstacle_detector_node.cpp
    VisResultType vis_type = VisResultType::JSON_AND_VOXELE; // VisResultType::BBOX_GROUND | VisResultType::BBOX_LIDAR_XY | VisResultType::BBOX_GROUND_2D

    RangeImageObstacleDetector detector(num_rings, num_sectors, max_distance, min_cluster_z_difference, vis_type);
    pcl::PointCloud<RSPointDefault>::Ptr rs_cloud_raw(new pcl::PointCloud<RSPointDefault>);
    std::string rs_pcd_file_path = "/home/weizh/data/rosbag2_2026_02_03-16_58_33/rosbag2_2026_02_03-16_58_33_0_logs/rslidar_points/1770109113_585270882.pcd";
    if (pcl::io::loadPCDFile<RSPointDefault>(rs_pcd_file_path, *rs_cloud_raw) == -1) {
        PCL_ERROR("Couldn't read file %s \n", rs_pcd_file_path.c_str());
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
}

ClusterType classifyCluster(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster) {
    if (cluster->points.size() < 10) return ClusterType::UNKNOWN;

    // 1. 创建法向量估计对象
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    ne.setInputCloud(cluster);

    // 2. 创建一个空的kdtree，用于近邻搜索
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    ne.setSearchMethod(tree);

    // 3. 输出变量
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

    // 4. 设置搜索半径 (非常重要：太大会平滑掉特征，太小会受噪声影响)
    // 建议设置在 0.1m - 0.2m 左右
    ne.setRadiusSearch(0.15);

    // 计算
    ne.compute(*cloud_normals);

    // 5. 特征统计
    float avg_curvature = 0.0f;
    std::vector<float> nz_values;
    int valid_normals = 0;

    for (const auto& n : cloud_normals->points) {
        if (!std::isfinite(n.normal_x) || !std::isfinite(n.curvature)) continue;
        
        avg_curvature += n.curvature;
        nz_values.push_back(std::abs(n.normal_z)); // 记录 Z 方向分量
        valid_normals++;
    }

    if (valid_normals == 0) return ClusterType::UNKNOWN;
    avg_curvature /= valid_normals;

    // 6. 计算法向量 Z 分量的标准差 (衡量整齐程度)
    float sum = std::accumulate(nz_values.begin(), nz_values.end(), 0.0);
    float mean = sum / nz_values.size();
    float sq_sum = std::inner_product(nz_values.begin(), nz_values.end(), nz_values.begin(), 0.0);
    float std_dev = std::sqrt(sq_sum / nz_values.size() - mean * mean);

    // 7. 启发式判断逻辑 (阈值需要根据实际雷达效果微调)
    // 草：曲率通常较高 (> 0.05)，且法向量 Z 分量分布杂乱 (std_dev 较大)
    // 人：曲率通常较低 (< 0.03)，且由于人体垂直于地面，躯干部分法向量 Z 分量较小且稳定
    
    if (avg_curvature > 0.06) {
        return ClusterType::GRASS;
    } else if (avg_curvature < 0.03 && std_dev < 0.15) {
        return ClusterType::HUMAN;
    }

    return ClusterType::UNKNOWN;
}

void detect_from_pcd_image() {
    std::string image_path = "/home/weizh/data/rosbag2_2026_03_07-09_11_00/rosbag2_2026_03_07-09_11_00_0_logs/image_for_verification.png";
    std::string intrinsics_path = "/home/weizh/data/calibration.json";
    cv::Mat image = cv::imread(image_path);

    // 3. 读取并解析内参 JSON
    std::ifstream f(intrinsics_path);
    // if (!f.is_open()) {
    //     std::cerr << "Could not open json: " << intrinsics_path << std::endl;
    // }
    json config = json::parse(f);

    // 提取内参矩阵 K
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            K.at<double>(i, j) = config["camera_matrix"][i][j];
        }
    }

    // 提取畸变系数 D (fisheye 模型通常需要 4 个系数)
    cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);
    for (int i = 0; i < 4; ++i) {
        D.at<double>(i, 0) = config["dist_coeffs"][i][0];
    }

    // 4. 定义外参矩阵 T_cam_lidar (4x4)
    cv::Mat T_cam_lidar = (cv::Mat_<double>(4, 4) <<
        0.0523359562429440, -0.9986295347545737,  0.0000000000000001,  0.0278654477929195,
        0.3102506583813496,  0.0162595480268680, -0.9505157316277836, -0.0841864926481438,
        0.9492130828523568,  0.0497461497387016,  0.3106764296307317, -0.0161601755839504,
        0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  1.0000000000000000);

    // 提取旋转矩阵 R 和平移向量 t 用于变换（或者直接手动计算）
    cv::Mat R_mat = T_cam_lidar(cv::Rect(0, 0, 3, 3));
    cv::Mat t_vec = T_cam_lidar(cv::Rect(3, 0, 1, 3));

    // 5. 遍历每个点云簇并投影
    // robosense airy 96
    int num_rings = 96;
    int num_sectors = 900;
    float max_distance = 10.0f; // Aligned with obstacle_detector_node.cpp
    float min_cluster_z_difference = 0.1f; // Aligned with obstacle_detector_node.cpp
    VisResultType vis_type = VisResultType::JSON_AND_VOXELL;

    RangeImageObstacleDetector detector(num_rings, num_sectors, max_distance, min_cluster_z_difference, vis_type);
    pcl::PointCloud<RSPointDefault>::Ptr rs_cloud_raw(new pcl::PointCloud<RSPointDefault>);
    std::string rs_pcd_file_path = "/home/weizh/data/rosbag2_2026_03_07-09_11_00/rosbag2_2026_03_07-09_11_00_0_logs/pcd_for_verification.pcd";
    if (pcl::io::loadPCDFile<RSPointDefault>(rs_pcd_file_path, *rs_cloud_raw) == -1) {
        PCL_ERROR("Couldn't read file %s \n", rs_pcd_file_path.c_str());
    }
    std::cout << "Loaded " << rs_cloud_raw->width * rs_cloud_raw->height << " data points from " << rs_pcd_file_path << std::endl;
    
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> obstacle_clusters = detector.detectObstacles(rs_cloud_raw);

    int cluster_id = 0;
    for (const auto& cluster : obstacle_clusters) {
        if (cluster->empty()) continue;

        std::vector<cv::Point3f> pts_3d_cam;
        
        // 步骤 A: 将点从 Lidar 坐标系转换到 Camera 光学坐标系
        for (const auto& pt_lidar : cluster->points) {
            // P_cam = R * P_lidar + t
            cv::Mat pt_lidar_mat = (cv::Mat_<double>(3, 1) << pt_lidar.x, pt_lidar.y, pt_lidar.z);
            cv::Mat pt_cam_mat = R_mat * pt_lidar_mat + t_vec;

            double z_cam = pt_cam_mat.at<double>(2, 0);

            // 过滤：只保留相机前方的点 (Z > 0.5 避免过近的畸变异常)
            if (z_cam > 0.5) {
                pts_3d_cam.push_back(cv::Point3f(
                    (float)pt_cam_mat.at<double>(0, 0),
                    (float)pt_cam_mat.at<double>(1, 0),
                    (float)pt_cam_mat.at<double>(2, 0)
                ));
            }
        }

        if (pts_3d_cam.empty()) continue;

        // 步骤 B: 使用 OpenCV Fisheye 模型投影
        std::vector<cv::Point2f> pts_2d;
        // 因为点已经在相机坐标系下，所以这里的 rvec 和 tvec 传零向量
        cv::Vec3d rvec_zero(0, 0, 0);
        cv::Vec3d tvec_zero(0, 0, 0);
        
        cv::fisheye::projectPoints(pts_3d_cam, pts_2d, rvec_zero, tvec_zero, K, D);

        // 步骤 C: 绘制到图像上
        // 为不同的簇生成不同的颜色

        // 根据类型决定颜色 -----
        ClusterType type = classifyCluster(cluster);
    
    cv::Scalar color;
    if (type == ClusterType::HUMAN) {
        color = cv::Scalar(0, 0, 255);   // 红色代表行人
    } else if (type == ClusterType::GRASS) {
        color = cv::Scalar(0, 255, 0);   // 绿色代表草
    } else {
        color = cv::Scalar(255, 255, 255); // 白色代表未知
    }
    // ------


        // cv::Scalar color( (cluster_id * 50) % 255, (cluster_id * 80) % 255, (cluster_id * 120) % 255 );

        for (const auto& pixel : pts_2d) {
            // 检查像素是否在图像范围内
            if (pixel.x >= 0 && pixel.x < image.cols && pixel.y >= 0 && pixel.y < image.rows) {
                cv::circle(image, pixel, 2, color, -1);
            }
        }
        cluster_id++;
    }

    // 6. 显示或保存结果
    cv::imshow("Projected Lidar Points", image);
    cv::imwrite("output_result.png", image);
    cv::waitKey(0);
}

int main(int argc, char * argv[]) {
    // detect_from_pcd();
    detect_from_pcd_image();

    // calibration
    // getLidarExtrinsic("/home/weizh/data/1768892940_599927902_ground.pcd");

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