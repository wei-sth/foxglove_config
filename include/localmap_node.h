#ifndef LOCALMAP_NODE_H
#define LOCALMAP_NODE_H

#include "types.h"
#include "utility.h"
#include "obstacle_detector.h"
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "foxglove_config/msg/segmentation_result.hpp"
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <thread>
#include <vector>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/pcl/pcl_registration_impl.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <mqtt/async_client.h>
#include "obstacle.pb.h"
#include <atomic>
#include <proj.h>


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

struct PublishSnapshot {
    double stamp = 0.0;  // seconds, scan_end_time
    Eigen::Affine3f T_odom_body = Eigen::Affine3f::Identity();
    std::vector<Voxel> voxels_snapshot;  // voxel centers stored in odom frame
    bool has_gps = false;
    double gps_stamp = 0.0;
    double gps_lon = 0.0;
    double gps_lat = 0.0;
    double gps_alt = 0.0;
    geometry_msgs::msg::Quaternion gps_orientation;
    int gps_status = 0;
};

struct KeyFrame {
    double timestamp;
    Eigen::Affine3f pose;
    pcl::PointCloud<PointType>::Ptr cloud;  // cloud in odometry frame
    small_gicp::PointCloud::Ptr cloud_new;
};

class LocalMap : public ParamServer,  public virtual mqtt::callback {
public:
    LocalMap(const rclcpp::NodeOptions & options);
    ~LocalMap();

    void removeDynamicObjTest();

    // debug, save slamProcessLoop total time on exit
    void recordSlamLoopTotalMs_(double total_ms);
    void dumpSlamLoopTotalCsv_(const std::string& out_path);
    void dumpGlobalPathCsvQuat_(const std::string& out_path);
    std::mutex mtx_timing_;
    std::vector<double> slam_total_ms_;
private:
    // --- Subscriptions ---
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_gps;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr sub_gps_orientation;
    rclcpp::Subscription<foxglove_config::msg::SegmentationResult>::SharedPtr sub_segmentation;

    // --- Publishers ---
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_initial_guess;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_registered;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_local_map;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacle_map;

    // --- Timers (independent publishers) ---
    rclcpp::TimerBase::SharedPtr timer_pub_obstacles_;
    void publishObstacleMapTimerCb();
    std::mutex mtx_latest_snapshot_;
    PublishSnapshot latest_snapshot_;
    bool has_latest_snapshot_ = false; // become true when first scan registered

    // --- Data Buffers ---
    struct LidarData {
        rclcpp::Time stamp;  // lidar message header.stamp
        double lidar_frame_beg_time;  // from lidar message header.stamp
        double lidar_frame_end_time;
        pcl::PointCloud<PointType>::Ptr cloud;
        pcl::PointCloud<RSPointDefault>::Ptr cloud_raw; // temp solution, Wei
    };
    LidarData current_lidar_data_;  // slam thread only: this is the frame being processed (NOT aligned)
    std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer;
    // IMU coverage / gap check for initial guess; if violated we reset local map and restart odom from current frame.
    double imu_gap_reset_thresh_s_ = 0.05;     // 50ms gap threshold (tune)
    int imu_min_samples_for_guess_ = 5;        // minimum imu samples in (t_last, t_curr] to trust integration
    std::deque<LidarData> lidar_buffer;
    std::mutex mtx_buffer;
    std::condition_variable cv_data;

    // --- Threads ---
    std::thread slam_thread;

    // --- Callbacks ---
    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr msg);
    void lidarHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void gpsHandler(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
    void gpsOrientationHandler(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg);
    void segmentationHandler(const foxglove_config::msg::SegmentationResult::SharedPtr msg);

    // --- Core Algorithm ---
    void slamProcessLoop();
    void processIMU(const std::vector<sensor_msgs::msg::Imu::SharedPtr>& imus);
    void pointCloudPreprocessing();
    small_gicp::PointCloud::Ptr convertToSmallGICP(pcl::PointCloud<PointType>::Ptr pcl_cloud);
    void performOdometer_v3();
    // Use IMU integrated delta (scan_end_time vs previous scan_end_time) to predict pose for registration initial guess
    Eigen::Isometry3d makeImuInitialGuessIsometry_(const Eigen::Isometry3d& T_last, double t_last, double t_curr) const;
    PointTypePose poseToPose6D(const Eigen::Affine3f& pose) const;
    void updatePath(const PointTypePose& pose_in);
    bool transformToLonLat(double x_odom, double y_odom, double z_odom, double& lon, double& lat, double& alt);
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

    Eigen::Affine3f T_ego_lidar = Eigen::Affine3f::Identity();  // lidar frame -> ego frame
    Eigen::Affine3f T_lidar_ego = Eigen::Affine3f::Identity();
    Eigen::Affine3f T_ego_body = Eigen::Affine3f::Identity();
    Eigen::Affine3f T_lidar_imu = Eigen::Affine3f::Identity();
    Eigen::Affine3f T_imu_lidar = Eigen::Affine3f::Identity();
    Eigen::Matrix3f extRot_f_ = Eigen::Matrix3f::Identity();  // lidar -> imu, double reading from yaml, try float might be OK enough
    Eigen::Vector3f extTrans_f_ = Eigen::Vector3f::Zero();

    // --- Data for processing ---
    sensor_msgs::msg::PointCloud2::SharedPtr current_lidar_msg;
    pcl::PointCloud<PointType>::Ptr laser_cloud_in;
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds;
    
    std::deque<sensor_msgs::msg::Imu> imu_que_opt;
    std::deque<sensor_msgs::msg::NavSatFix> gpsQueue;
    std::deque<geometry_msgs::msg::QuaternionStamped> gpsOrientationQueue;
    std::deque<foxglove_config::msg::SegmentationResult> segmentationQueue;
    std::mutex mtx_segmentation_;
    
    double scan_beg_time;
    double scan_end_time;
    double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_segmentation = -1.0;

    float imu_rot_x[2000];
    float imu_rot_y[2000];
    float imu_rot_z[2000];
    double imu_time[2000];
    int imu_ptr_cur;

    bool first_imu_flag = true;
    Eigen::Quaterniond q_main;

    // --- IMU-based initial guess (slam thread only, no lock needed) ---
    Eigen::Affine3f initial_guess_pose_ = Eigen::Affine3f::Identity();  // odom<-body initial guess used for registration
    double last_imu_guess_time_ = -1.0;     // seconds, last lidar scan_end_time used for prediction
    Eigen::Quaterniond q_imu_guess_ = Eigen::Quaterniond::Identity();  // accumulated orientation (relative)

    // Translation prediction state (for makeImuInitialGuessIsometry_)
    bool has_prev_pose_for_vel_ = false;
    double prev_pose_time_ = -1.0;                 // seconds (scan_end_time of previous-previous frame)
    Eigen::Isometry3d prev_pose_T_ = Eigen::Isometry3d::Identity();  // odom<-body at prev_pose_time_
    Eigen::Vector3d vel_ego_ = Eigen::Vector3d::Zero();              // estimated ego-frame velocity (m/s)
    double max_ego_speed_mps_ = 2.0;               // clamp speed for extrapolation
    double imu_accel_integ_max_dt_ = 0.2;          // seconds, if dt <= this use IMU accel integration, else fallback to constant velocity
    Eigen::Vector3d gravity_dir_body_unit_ = Eigen::Vector3d(0.0, 0.0, -1.0); // unit vector of gravity direction in body(imu) frame
    double gravity_mag_mps2_ = 9.81;
    Eigen::Vector3d gyro_bias_radps_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc_bias_mps2_ = Eigen::Vector3d::Zero();

    // --- Odometry and Mapping ---
    Eigen::Affine3f current_pose = Eigen::Affine3f::Identity();
    Eigen::Affine3f last_key_pose = Eigen::Affine3f::Identity();
    std::deque<KeyFrame> keyframe_queue;
    std::shared_ptr<small_gicp::GaussianVoxelMap> voxel_target;
    int num_threads = 4;  // OpenMP
    pcl::PointCloud<PointType>::Ptr local_map;
    pcl::PointCloud<PointType>::Ptr last_laser_cloud_in;
    bool is_first_frame = true;
    nav_msgs::msg::Path globalPath;
    mutable std::mutex mtx_path_;
    
    // Motion prior (soft constraint) for small_gicp optimization:
    // - prefer rotation around ego_z (in ego frame) only
    // - penalize lateral translation along ego_x (in ego frame)
    bool enable_motion_prior_ = false;
    double motion_prior_rot_lambda_ = 1e6;   // larger => stronger constraint
    double motion_prior_lat_lambda_ = 1e4;   // larger => stronger constraint
    double motion_prior_ego_x_offset_ = 0.0; // (meters) optional: ego origin offset along ego_x to reduce rotation-translation coupling
    double motion_prior_ego_y_offset_ = 0.0; // (meters)

    VisResultType vis_type_;
    std::unique_ptr<RangeImageObstacleDetector> detector_;
    std::map<std::tuple<int, int, int>, Voxel> obstacle_voxel_map;
    const double max_range_ = 60;  // max range to keep, used to filter out nan and extremely far points. 60m is from robosense airy manual
    cv::Mat range_image_;
    cv::Mat x_image_;
    cv::Mat y_image_;
    cv::Mat z_image_;
    cv::Mat valid_mask_;

    // from robosense airy manual (unit: degree), 96 rings in total, ring index from 0 to 95
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

    mqtt::async_client mqtt_client_;
    mqtt::connect_options mqtt_conn_opts_;
    foxglove_config::VoxelCloud pb_cloud_;
    std::string mqtt_payload_buffer_;

    // gps projector
    PJ_CONTEXT *pj_context_ = nullptr;
    PJ *pj_normalized_ = nullptr;
    int epsg_code_ = -1; // -1 means not initialized
    int getBestEPSG(double lon);
    bool initProjection(int epsg_code);
    bool project(double lon, double lat, double &x, double &y);

    // obstacle geo conversion placeholder:
    // odom/local frame -> ENU offset (meters) -> lon/lat via PROJ
    // Replace these placeholders with real extrinsic/geodesic calibration when available.
    Eigen::Vector3d obstacle_geo_origin_llh_ = Eigen::Vector3d::Zero(); // lon, lat, alt
    Eigen::Matrix3d obstacle_geo_rot_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d obstacle_geo_trans_m_ = Eigen::Vector3d::Zero();
};

#endif // LOCALMAP_NODE_H
