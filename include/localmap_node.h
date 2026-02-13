#ifndef LOCALMAP_NODE_H
#define LOCALMAP_NODE_H

#include "types.h"
#include "utility.h"
#include "obstacle_detector.h"
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <thread>
#include <vector>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/pcl/pcl_registration_impl.hpp>


// obstacle voxel, grass is not obstacle
struct Voxel {
    int ix, iy, iz;  // index
    float x, y, z;
    double first_seen_time;
    double last_seen_time;
    int observation_count;
    int cluster_id;
    bool is_dynamic;
};


class LocalMap : public ParamServer {
public:
    LocalMap(const rclcpp::NodeOptions & options);
    ~LocalMap();

private:
    // --- Subscriptions ---
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar;

    // --- Publishers ---
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_registered;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_local_map;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacle_map;

    // --- Data Buffers ---
    struct LidarData {
        double lidar_frame_beg_time;
        double lidar_frame_end_time;
        pcl::PointCloud<PointType>::Ptr cloud;
        pcl::PointCloud<RSPointDefault>::Ptr cloud_raw; // temp solution, Wei
    };
    std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer;
    std::deque<LidarData> lidar_buffer;
    std::mutex mtx_buffer;
    std::condition_variable cv_data;

    // --- Threads ---
    std::thread slam_thread;

    // --- Callbacks ---
    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr msg);
    void lidarHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // --- Core Algorithm ---
    void slamProcessLoop();
    void processIMU(const std::vector<sensor_msgs::msg::Imu::SharedPtr>& imus);
    void pointCloudPreprocessing();
    void performOdometer();
    void performOdometer_v1();
    void updateLocalMap();
    void updatePath(const PointTypePose& pose_in);
    void publishResult();
    void updateObstacleVoxelMap(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& obstacle_clusters, 
        const Eigen::Affine3f& pose, double timestamp);
    void cleanupObstacleVoxelMap(const Eigen::Affine3f& pose, double timestamp);
    pcl::PointCloud<PointType>::Ptr filterByRangeEgo(pcl::PointCloud<RSPointDefault>::Ptr cloud, const double& max_dis);
    void buildRangeImage(pcl::PointCloud<PointType>::Ptr cloud);
    int pitchToRow(float pitch_deg);

    // --- Undistortion / Deskew ---
    void findRotation(double relTime, float *rotXCur, float *rotYCur, float *rotZCur);
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur);
    void deskewPoint(PointType *point, double relTime);
    PointType lidarToImu(const PointType& p);

    // --- Data for processing ---
    sensor_msgs::msg::PointCloud2::SharedPtr current_lidar_msg;
    pcl::PointCloud<PointType>::Ptr laser_cloud_in;
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds;
    
    std::deque<sensor_msgs::msg::Imu> imu_que_opt;
    
    double scan_beg_time;
    double scan_end_time;
    double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
    rclcpp::Time lidarMsgTimestamp;
    double lidarMsgTimeValue;

    float imu_rot_x[2000];
    float imu_rot_y[2000];
    float imu_rot_z[2000];
    double imu_time[2000];
    int imu_ptr_cur;

    bool first_imu_flag = true;
    Eigen::Quaterniond q_main;

    // --- Odometry and Mapping ---
    Eigen::Affine3f current_pose = Eigen::Affine3f::Identity();
    Eigen::Affine3f last_key_pose = Eigen::Affine3f::Identity();
    pcl::PointCloud<PointType>::Ptr local_map;
    pcl::PointCloud<PointType>::Ptr last_laser_cloud_in;
    bool is_first_frame = true;
    nav_msgs::msg::Path globalPath;

    VisResultType vis_type_;
    std::unique_ptr<RangeImageObstacleDetector> detector_;
    std::map<std::tuple<int, int, int>, Voxel> obstacle_voxel_map;
    const double max_range_ = 60;  // max range to keep, used to filter out nan and extremely far points. 60m is from robosense airy manual
    cv::Mat range_image_;
    cv::Mat x_image_;
    cv::Mat y_image_;
    cv::Mat z_image_;
    cv::Mat valid_mask_;

    // from robosense airy manual (unit: degree), index from 1 to 96
    const std::vector<float> ring_pitches = {
        // 1 - 10
        -0.07, 0.88, 1.81, 2.76, 3.69, 4.62, 5.54, 6.48, 7.41, 8.34,
        // 11 - 20
        9.27, 10.21, 11.15, 12.09, 13.03, 13.98, 14.92, 15.87, 16.82, 17.77,
        // 21 - 30
        18.72, 19.67, 20.62, 21.57, 22.51, 23.45, 24.4, 25.33, 26.28, 27.21,
        // 31 - 40
        28.15, 29.08, 30.02, 30.95, 31.88, 32.82, 33.74, 34.68, 35.62, 36.55,
        // 41 - 50
        37.5, 38.43, 39.37, 40.31, 41.25, 42.21, 43.16, 44.09, 45.05, 46.0,
        // 51 - 60
        46.95, 47.9, 48.85, 49.8, 50.73, 51.69, 52.62, 53.56, 54.5, 55.45,
        // 61 - 70
        56.37, 57.3, 58.24, 59.18, 60.12, 61.05, 61.99, 62.93, 63.86, 64.81,
        // 71 - 80
        65.76, 66.69, 67.65, 68.6, 69.56, 70.51, 71.46, 72.42, 73.37, 74.33,
        // 81 - 90
        75.29, 76.24, 77.19, 78.14, 79.07, 80.02, 80.96, 81.9, 82.84, 83.78,
        // 91 - 96
        84.7, 85.64, 86.57, 87.52, 88.46, 89.4
    };
};

#endif // LOCALMAP_NODE_H
