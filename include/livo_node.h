#ifndef LIVO_NODE_H
#define LIVO_NODE_H

#include "utility.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <thread>
#include <vector>

#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/pcl/pcl_registration_impl.hpp>

class LivoNode : public ParamServer {
public:
    LivoNode(const rclcpp::NodeOptions & options);
    ~LivoNode();

private:
    // --- Subscriptions ---
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar;

    // --- Publishers ---
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_registered;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_local_map;

    // --- Data Buffers ---
    struct LidarData {
        double lidar_frame_beg_time;
        double lidar_frame_end_time;
        pcl::PointCloud<PointType>::Ptr cloud;
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
    void updateLocalMap();
    void publishResult();

    // --- Undistortion / Deskew ---
    void findRotation(double relTime, float *rotXCur, float *rotYCur, float *rotZCur);
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur);
    void deskewPoint(PointType *point, double relTime);

    // --- Data for processing ---
    sensor_msgs::msg::PointCloud2::SharedPtr current_lidar_msg;
    pcl::PointCloud<PointType>::Ptr laser_cloud_in;
    
    std::deque<sensor_msgs::msg::Imu> imu_que_opt;
    
    double scan_beg_time;
    double scan_end_time;

    float imu_rot_x[2000];
    float imu_rot_y[2000];
    float imu_rot_z[2000];
    double imu_time[2000];
    int imu_ptr_cur;

    bool first_imu_flag = true;
    Eigen::Quaterniond q_main;

    // --- Odometry and Mapping ---
    Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();
    pcl::PointCloud<PointType>::Ptr local_map;
    pcl::PointCloud<PointType>::Ptr last_laser_cloud_in;
    bool is_first_frame = true;
};

#endif // LIVO_NODE_H
