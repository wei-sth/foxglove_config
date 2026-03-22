#include "localmap_node.h"
#include <chrono>
#include <fstream>
#include <string>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cmath>

// robosense airy frame_id: rslidar
// bbox is not suitable for indoor, I tried to use nav2_msgs/msg/VoxelGrid, but rviz cannot show
// so use pointcloud, set style as boxes, size = 0.1
// MQTT broker kicks the older client if a new one connects with the same ID. 
// Using a unique ID for each device to avoid connection loops. I use "obstacle_client_pc" (need to use yaml in the future)
// mqtt ip to be used: 127.0.0.1 | 124.221.132.177

// downsample by 0.1m does not improve
// downsample by 0.2m, setNumNeighborsForCovariance(20) from 40 improves
// without setting other params (num of threads ...), voxelgrid_sampling_omp seems slower than pcl::VoxelGrid
// then I change voxelgrid_sampling_omp to voxelgrid_sampling, and set resolution of both source and target to 0.2m (previously source 0.2 and target 0.1), faster
// todo: check https://github.com/koide3/small_gicp/blob/master/src/benchmark/odometry_benchmark_small_gicp_omp.cpp
// must read: small_gicp/pcl/pcl_registration_impl.hpp

// Dynamic objects can leave “ghost trails” in obstacle_voxel_map (clusters appear to grow), but an expanding cluster is not always dynamic—it can also be a long, static fence near the FOV boundary that looks longer as the mower moves.

// todo: 应该针对local map中的所有obstacle划定一个ROI，对这个ROI中的做ray casting，看看是否可行，找到落到这个ROI中的最后一帧lidar的障碍物
// 在查询 range_image[u][v] 时，不要只查一个像素，可以查周围 3x3 的邻域，取最小值或者进行某种插值，防止因为雷达光束打偏了而误删障碍物。

// vis_type_ = VisResultType::JSON_AND_VOXELL for both true and false, obstacle transform workflow: lidar->body(imu)->odom(debug ends here)->body(imu)->lidar->ego
// true:  output frame (obstacle + registered cloud + path): odom
// false: output frame (obstacle): ego
static constexpr bool LOCAL_DEBUG = true;

static Eigen::Matrix4f vectorToMat4f(const std::vector<double>& v) {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    if (v.size() != 16) {
        return m;
    }
    // YAML uses row-major [r11 r12 r13 tx r21 ...]
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            m(r, c) = static_cast<float>(v[r * 4 + c]);
        }
    }
    return m;
}

LocalMap::LocalMap(const rclcpp::NodeOptions & options) : ParamServer("localmap", options), mqtt_client_("tcp://127.0.0.1:1883", "obstacle_client_jetson") {
    // IMU calibration (from your offline stats)
    gyro_bias_radps_ << -0.017762434996958294, 0.010876713446207395, 0.016580594586994103;
    acc_bias_mps2_ << -0.0039463048174464554, -0.062819024226798525, 0.1126052397636137;
    gravity_mag_mps2_ = 9.6776471196106844;
    gravity_dir_body_unit_ << 0.030590827162440038, 0.48695830695576769, -0.87288934498038762;

    // Cache extrinsics in float for fast per-point transforms
    extRot_f_ = extRot.cast<float>();
    extTrans_f_ = extTrans.cast<float>();
    T_ego_lidar = Eigen::Affine3f(vectorToMat4f(T_ground_lidar));
    // front-left-up to right-front-up
    T_ego_lidar.prerotate((Eigen::Matrix3f() << 0, -1, 0, 1, 0, 0, 0, 0, 1).finished());
    T_lidar_ego = T_ego_lidar.inverse();
    T_imu_lidar.linear() = extRot.cast<float>();
    T_imu_lidar.translation() = extTrans.cast<float>();
    T_lidar_imu = T_imu_lidar.inverse();
    T_ego_body = T_ego_lidar * T_lidar_imu;
    
    mqtt_client_.set_callback(*this);
    mqtt_conn_opts_.set_user_name("zbtest");
    mqtt_conn_opts_.set_password("zbtest");
    mqtt_conn_opts_.set_keep_alive_interval(5);
    mqtt_conn_opts_.set_connect_timeout(5);
    mqtt_conn_opts_.set_clean_session(true);
    mqtt_conn_opts_.set_automatic_reconnect(true);
    try {
        mqtt_client_.connect(mqtt_conn_opts_)->wait();
        RCLCPP_INFO(this->get_logger(), "Connected to MQTT Broker");
    } catch (const mqtt::exception& exc) {
        RCLCPP_ERROR(this->get_logger(), "MQTT Connection failed: %s", exc.what());
    }

    // keep at most 2 lidar scans, need realtime obstacle; keep 100 imu data (100Hz, 1 second), in case of lidar data loss
    sub_imu = create_subscription<sensor_msgs::msg::Imu>(imuTopic, rclcpp::QoS(100).best_effort(), std::bind(&LocalMap::imuHandler, this, std::placeholders::_1));
    sub_lidar = create_subscription<sensor_msgs::msg::PointCloud2>(pointCloudTopic, rclcpp::QoS(2).best_effort(), std::bind(&LocalMap::lidarHandler, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Subscribed to IMU topic: %s", imuTopic.c_str());
    RCLCPP_INFO(get_logger(), "Subscribed to Lidar topic: %s", pointCloudTopic.c_str());

    laser_cloud_in.reset(new pcl::PointCloud<PointType>());
    laser_cloud_in_ds.reset(new pcl::PointCloud<PointType>());
    local_map.reset(new pcl::PointCloud<PointType>());
    last_laser_cloud_in.reset(new pcl::PointCloud<PointType>());
    obstacle_voxel_map.clear();

    // publisher
    auto qos_reliable = rclcpp::QoS(10).reliable();
    pub_odom = create_publisher<nav_msgs::msg::Odometry>("/localmap/odometry", qos_reliable);
    pub_path = create_publisher<nav_msgs::msg::Path>("/localmap/path", qos_reliable);
    pub_initial_guess = create_publisher<sensor_msgs::msg::PointCloud2>("/localmap/initial_guess", qos_reliable);
    pub_cloud_registered = create_publisher<sensor_msgs::msg::PointCloud2>("/localmap/cloud_registered", qos_reliable);
    pub_local_map = create_publisher<sensor_msgs::msg::PointCloud2>("/localmap/local_map", qos_reliable);
    pub_obstacle_map = create_publisher<sensor_msgs::msg::PointCloud2>("/localmap/obstacle_voxel_grid", qos_reliable);
    // publish obstacle_voxel_map snapshot each 100ms
    timer_pub_obstacles_ = create_wall_timer(std::chrono::milliseconds(100), std::bind(&LocalMap::publishObstacleMapTimerCb, this));

    imu_ptr_cur = 0;
    scan_beg_time = 0;
    scan_end_time = 0;

    range_image_ = cv::Mat(nRing, hResolution, CV_32FC1, cv::Scalar(std::numeric_limits<float>::max()));
    x_image_ = cv::Mat(nRing, hResolution, CV_32FC1, cv::Scalar(0));
    y_image_ = cv::Mat(nRing, hResolution, CV_32FC1, cv::Scalar(0));
    z_image_ = cv::Mat(nRing, hResolution, CV_32FC1, cv::Scalar(0));
    valid_mask_ = cv::Mat(nRing, hResolution, CV_8UC1, cv::Scalar(0));

    vis_type_ = VisResultType::JSON_AND_VOXELL;
    detector_ = std::make_unique<RangeImageObstacleDetector>(nRing, hResolution, detMaxDistance, detMinClusterHeight, vis_type_);

    slam_thread = std::thread(&LocalMap::slamProcessLoop, this);
}

LocalMap::~LocalMap() {
    if (slam_thread.joinable()) {
        slam_thread.join();
    }
}

void LocalMap::imuHandler(const sensor_msgs::msg::Imu::SharedPtr msg) {
    // use logic in fast-livo2, if msg timestamp < last_timestamp_imu, drop this msg, in this way, imu_buffer is sorted
    double timestamp = rclcpp::Time(msg->header.stamp).seconds();
    // lock block
    {
        std::lock_guard<std::mutex> lock(mtx_buffer);
        if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
        {
            RCLCPP_ERROR(get_logger(), "imu loop back, offset: %lf", last_timestamp_imu - timestamp); // if happens a lot, move out of lock
            return;
        }
        if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
        {
            RCLCPP_WARN(get_logger(), "imu timestamp jumps %0.4lf seconds", timestamp - last_timestamp_imu);
            // do not return, if imu timestamp jumps because of data loss, last_timestamp_imu will never update again and system will not recover
        }

        last_timestamp_imu = timestamp;
        imu_buffer.push_back(msg);
        // imu do not notify, let lidarHandler notify
    }
}

// check imu_prop_callback function in fast-livo2, if speed = 20km/h, 100 ms means 0.55m, for obstacle detection, we need real time pose
// we should based on the pose optimized by last lidar align, calculate relative movement from imu

void LocalMap::lidarHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<RSPointDefault> tmp_rs_cloud;
    pcl::fromROSMsg(*msg, tmp_rs_cloud);

    // robosense point time is relative time(unit: ns), msg header timestamp is scan start time
    LidarData data;
    data.stamp = rclcpp::Time(msg->header.stamp);
    data.lidar_frame_beg_time = data.stamp.seconds();
    data.lidar_frame_end_time = data.lidar_frame_beg_time + 0.1;  // 10Hz
    data.cloud.reset(new pcl::PointCloud<PointType>());
    data.cloud->reserve(tmp_rs_cloud.size());
    
    // remove ego cloud and outlier (noisy points extremely far, 100m)
    for (const auto& p : tmp_rs_cloud.points) {
        // NaNs are implicitly filtered out because any comparison (>, <, ==) with NaN returns false.
        if (p.x*p.x + p.y*p.y + p.z*p.z < 10000) {
            PointType dst;
            dst.x = p.x;
            dst.y = p.y;
            dst.z = p.z;
            dst.intensity = p.intensity;
            dst.normal_x = p.ring;   // ring -- normal_x
            dst.normal_y = p.timestamp * 1e-9;   // relative time -- normal_y, ns -> s
            dst = lidarToImu(dst);
            data.cloud->push_back(dst);
        }
    }

    data.cloud_raw.reset(new pcl::PointCloud<RSPointDefault>(std::move(tmp_rs_cloud))); // tmp_rs_cloud becomes empty

    std::lock_guard<std::mutex> lock(mtx_buffer);
    lidar_buffer.push_back(data);
    cv_data.notify_one(); // notify slam process
}

void LocalMap::slamProcessLoop() {
    while (rclcpp::ok()) {
        LidarData current_lidar_data;
        std::vector<sensor_msgs::msg::Imu::SharedPtr> current_imus;

        std::unique_lock<std::mutex> lock(mtx_buffer);
        cv_data.wait(lock, [this] { return !lidar_buffer.empty() || !rclcpp::ok(); });
        
        if (!rclcpp::ok()) break;

        current_lidar_data = lidar_buffer.front();
        lidar_buffer.pop_front();
        current_lidar_data_ = current_lidar_data;
        scan_beg_time = current_lidar_data.lidar_frame_beg_time;
        scan_end_time = current_lidar_data.lidar_frame_end_time;

        while (!imu_buffer.empty() && rclcpp::Time(imu_buffer.front()->header.stamp).seconds() < scan_end_time) {
            current_imus.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }

        laser_cloud_in = current_lidar_data.cloud;
        lock.unlock();  //unlock what?
        if (current_imus.empty()) {
            RCLCPP_ERROR(get_logger(), "no imu for lidar %lf", scan_beg_time);
        }

        // todo: consider deskew of obstacles
        // todo: consider localization fail
        Eigen::Affine3f localization_pose;
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> obstacle_clusters;

        // #pragma omp parallel sections num_threads(2)
        // {
        //     #pragma omp section
        //     {
        //         performLocalization(cloud_far, localization_pose);
        //     }
            
        //     #pragma omp section
        //     {
        //         obstacle_clusters = detector_->detectObstacles(current_lidar_data.cloud_raw);
        //     }
        // }

        obstacle_clusters = detector_->detectObstacles(current_lidar_data.cloud_raw);

        auto t1 = std::chrono::steady_clock::now();
        processIMU(current_imus);
        auto t2 = std::chrono::steady_clock::now();
        pointCloudPreprocessing();
        auto t3 = std::chrono::steady_clock::now();
        // v0: current scan align to pre scan, use pre pose as initial guess, not good for long duration
        // performOdometer();
        // v1: current scan align to local map, only key frame is added to local map
        // v2: current scan align to local map, only key frame is added to local map, use sliding window to drop old key frames
        performOdometer_v3();
        auto t4 = std::chrono::steady_clock::now();

        // remove dynamic by range image
        auto cloud_filtered_normal = filterByRangeEgo(current_lidar_data.cloud_raw, max_range_);
        buildRangeImage(cloud_filtered_normal);

        updateObstacleVoxelMap(obstacle_clusters, current_pose, scan_end_time);

        // slam thread generates snapshot, better than timer read slam work-in-progress variables.
        PublishSnapshot snap;
        snap.stamp = scan_end_time;
        snap.T_odom_body = current_pose;
        snap.voxels_snapshot.reserve(obstacle_voxel_map.size());
        for (const auto& kv : obstacle_voxel_map) {
            snap.voxels_snapshot.push_back(kv.second);
        }
        {
            std::lock_guard<std::mutex> lk(mtx_latest_snapshot_);
            latest_snapshot_ = std::move(snap);
            has_latest_snapshot_ = true;
        }

        if constexpr (LOCAL_DEBUG) {
            publishResult();
        }
        auto t5 = std::chrono::steady_clock::now();

        double d1 = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double d2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double d3 = std::chrono::duration<double, std::milli>(t4 - t3).count();
        double d4 = std::chrono::duration<double, std::milli>(t5 - t4).count();
        double total = d1 + d2 + d3 + d4;

        recordSlamLoopTotalMs_(total);
        // RCLCPP_INFO(get_logger(), "Time stats: IMU: %.2f%%, Preprocess: %.2f%%, Odom: %.2f%%, Publish: %.2f%%, Total: %.2f ms", d1/total*100.0, d2/total*100.0, d3/total*100.0, d4/total*100.0, total);
    }
}

void LocalMap::updateObstacleVoxelMap(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& obstacle_clusters,
    const Eigen::Affine3f& pose, double timestamp) {
    // detector output clusters are in lidar frame
    if (obstacle_clusters.empty()) {
        return;
    }
    
    float resolution = 0.1f;
    std::map<std::tuple<int, int, int>, Voxel> new_voxels;
    
    #pragma omp parallel for
    for (size_t i = 0; i < obstacle_clusters.size(); ++i) {
        const auto& cluster = obstacle_clusters[i];
        std::map<std::tuple<int, int, int>, Voxel> local_voxels;
        
        const Eigen::Matrix3f R_ob = pose.linear();
        const Eigen::Vector3f t_ob = pose.translation();

        // odom<-body<-lidar
        const Eigen::Matrix3f R_ol = R_ob * extRot_f_;
        const Eigen::Vector3f t_ol = R_ob * extTrans_f_ + t_ob;

        for (const auto& pt : cluster->points) {
            // Fast path: p_odom = R_ol * p_lidar + t_ol
            const Eigen::Vector3f p_lidar(pt.x, pt.y, pt.z);
            const Eigen::Vector3f pt_world = R_ol * p_lidar + t_ol;

            int ix = std::floor(pt_world.x() / resolution);
            int iy = std::floor(pt_world.y() / resolution);
            int iz = std::floor(pt_world.z() / resolution);
            
            auto key = std::make_tuple(ix, iy, iz);
            
            if (local_voxels.find(key) == local_voxels.end()) {
                Voxel voxel;
                voxel.ix = ix;
                voxel.iy = iy;
                voxel.iz = iz;
                voxel.x = ix * resolution + resolution / 2.0f;
                voxel.y = iy * resolution + resolution / 2.0f;
                voxel.z = iz * resolution + resolution / 2.0f;
                voxel.first_seen_time = timestamp;
                voxel.last_seen_time = timestamp;
                voxel.observation_count = 1;
                voxel.cluster_id = i;
                voxel.is_dynamic = false;
                
                local_voxels[key] = voxel;
            }
        }
        
        // 合并到new_voxels（需要加锁）
        #pragma omp critical
        {
            for (const auto& kv : local_voxels) {
                new_voxels[kv.first] = kv.second;
            }
        }
    }
    
    // 2. 更新obstacle_voxel_map
    for (const auto& kv : new_voxels) {
        const auto& key = kv.first;
        const auto& new_voxel = kv.second;
        
        auto it = obstacle_voxel_map.find(key);
        if (it != obstacle_voxel_map.end()) {
            // 已存在，更新时间和观测次数
            it->second.last_seen_time = timestamp;
            it->second.observation_count++;
            
            // 检查是否是静态（观测3次以上）
            if (it->second.observation_count >= 3) {
                it->second.is_dynamic = false;
            }
        } else {
            // 新体素，直接添加
            obstacle_voxel_map[key] = new_voxel;
        }
    }
    
    cleanupObstacleVoxelMap(pose, timestamp);

    RCLCPP_DEBUG(get_logger(), "Obstacle voxel map size: %zu", obstacle_voxel_map.size());
}

void LocalMap::cleanupObstacleVoxelMap(const Eigen::Affine3f& pose, double timestamp) {
    float max_distance = 10.0f;
    
    auto it = obstacle_voxel_map.begin();
    while (it != obstacle_voxel_map.end()) {
        const Voxel& voxel = it->second;
        Eigen::Vector3f voxel_pos(voxel.x, voxel.y, voxel.z);
        float dist = (voxel_pos - pose.translation()).norm();
        
        bool should_remove = false;
        
        if (dist > max_distance || timestamp - voxel.last_seen_time > obstacleLifetime) {
            should_remove = true;
        }
        
        // 条件3：在视野内但未观测到（需要实现视野判断）
        // if (isInFOV(voxel, pose) && timestamp - voxel.last_seen_time > 1.0) {
        //     should_remove = true;
        // }
        
        if (should_remove) {
            it = obstacle_voxel_map.erase(it);
        } else {
            ++it;
        }
    }
}

pcl::PointCloud<PointType>::Ptr LocalMap::filterByRangeEgo(pcl::PointCloud<RSPointDefault>::Ptr cloud, const double& max_dis) {
    // todo: consider only do it once, did once in obstacle detector ..., but with different range and frame
    // Filter by distance, convert to PointType, note result is in sensor frame
    pcl::PointCloud<PointType>::Ptr filtered_normal(new pcl::PointCloud<PointType>);
    filtered_normal->points.reserve(cloud->points.size()); // Reserve space
    
    for (const auto& pt : cloud->points) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        float distance = p.norm();
        
        // NaNs are implicitly filtered out because any comparison (>, <, ==) with NaN returns false.
        if (distance <= max_dis && distance > 0.5f) {  // Min distance 0.5m
            PointType pt_normal;
            pt_normal.x = p.x();
            pt_normal.y = p.y();
            pt_normal.z = p.z();
            pt_normal.intensity = pt.intensity;
            pt_normal.normal_x = static_cast<float>(pt.ring); // Store ring in normal_x
            pt_normal.normal_y = pt.timestamp; // Store time in normal_y
            // normal_z, curvature can be left as default or set to 0
            
            filtered_normal->points.push_back(pt_normal);
        }
    }
    
    return filtered_normal;
}

void LocalMap::buildRangeImage(pcl::PointCloud<PointType>::Ptr cloud) {
    // since we use min range, must filter out ego cloud first

    range_image_.setTo(std::numeric_limits<float>::max());
    x_image_.setTo(0);
    y_image_.setTo(0);
    z_image_.setTo(0);
    valid_mask_.setTo(0);
    
    for (const auto& pt : cloud->points) {
        uint16_t ring = static_cast<uint16_t>(pt.normal_x); // Retrieve ring from normal_x
        if (ring >= nRing) continue;
        
        float azimuth = std::atan2(pt.y, pt.x);
        if (azimuth < 0) azimuth += 2 * M_PI;
        
        int col = static_cast<int>(azimuth / (2 * M_PI) * hResolution);
        col = std::min(col, hResolution - 1);
        
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

int LocalMap::pitchToRow(float pitch_deg) {
    // return [0, nRing - 1]
    // the first >= pitch_deg position
    auto it = std::lower_bound(ring_pitches.begin(), ring_pitches.end(), pitch_deg);

    if (it == ring_pitches.begin()) return 0;
    if (it == ring_pitches.end()) return nRing - 1;

    auto prev = std::prev(it);
    int idx;
    if (pitch_deg - *prev < *it - pitch_deg) {
        idx = std::distance(ring_pitches.begin(), prev);
    } else {
        idx = std::distance(ring_pitches.begin(), it);
    }
    return idx;
}

// void LocalMap::updateVoxelVisibility(const Eigen::Affine3f& current_pose) {
//     // 遍历全局维护的 Voxel Map
//     for (auto it = obstacle_voxel_map.begin(); it != obstacle_voxel_map.end(); ) {
//         Voxel& v = it->second;

//         // 1. 将 Voxel 世界坐标转到当前雷达局部坐标系
//         // Eigen::Vector3f pt_world(v.x, v.y, v.z);
//         // Eigen::Vector3f pt_local = current_pose.inverse() * pt_world;
        
//         // 假设你已经有了局部坐标 pt_local
//         float d_v = std::sqrt(pt_local.x()*pt_local.x() + pt_local.y()*pt_local.y() + pt_local.z()*pt_local.z());
        
//         // 2. 计算投影到 RangeImage 的坐标
//         float azimuth = std::atan2(pt_local.y(), pt_local.x());
//         if (azimuth < 0) azimuth += 2 * M_PI;
//         int col = static_cast<int>(azimuth / (2 * M_PI) * hResolution);
//         col = std::min(col, hResolution - 1);

//         float pitch_deg = std::atan2(pt_local.z(), std::sqrt(pt_local.x()*pt_local.x() + pt_local.y()*pt_local.y())) * 180.0 / M_PI;
//         int row = pitchToRow(pitch_deg);

//         // 3. 可见性校验 (Visibility Test)
//         bool is_pierced = false; // 是否被穿透
//         int window_size = 1;     // 考虑到 0.1m Voxel 和 1度线间距，查 3x3 窗口最鲁棒
        
//         for (int r = row - window_size; r <= row + window_size; ++r) {
//             for (int c = col - window_size; c <= col + window_size; ++c) {
//                 if (r < 0 || r >= nRing || c < 0 || c >= hResolution) continue;
                
//                 if (valid_mask_.at<uint8_t>(r, c) == 0) continue; // 没数据，跳过

//                 float d_obs = range_image_.at<float>(r, c);
                
//                 // 核心判定：如果观测距离明显大于 Voxel 距离，说明 Voxel 位置变空了
//                 if (d_obs > d_v + 0.15) { // 0.15m 为容忍偏差
//                     is_pierced = true;
//                     break;
//                 }
//             }
//             if (is_pierced) break;
//         }

//         // 4. 更新 HMM / Log-Odds 状态
//         if (is_pierced) {
//             v.observation_count--; // 或者你的 log_odds 逻辑
//             if (v.observation_count < -5) { // 举例：连续多次没看到，判定动态并删除
//                 it = obstacle_voxel_map.erase(it);
//                 continue;
//             }
//         } else {
//             // 如果观测距离接近，可以增加其静态权重
//             // if (abs(d_obs - d_v) < 0.1) v.observation_count++;
//         }
        
//         ++it;
//     }
// }

void LocalMap::processIMU(const std::vector<sensor_msgs::msg::Imu::SharedPtr>& imus) {
    for (const auto& imu_msg : imus) {
        imu_que_opt.push_back(*imu_msg);
    }

    if (imu_que_opt.empty()) {
        imu_ptr_cur = 0;
        return;
    }

    // Deskew needs [scan_beg, scan_end] + ONE sample at/before scan_beg for interpolation.
    // IMU initial guess needs (last_imu_guess_time_, scan_end]
    // last_imu_guess_time_ is the scan_end of previous registered scan. Ideally, last_imu_guess_time_ should be very close to the scan_beg of current scan (to be registered), 
    // but due to lidar data loss, there might be large gap.
    double keep_from_t = scan_beg_time;
    if (last_imu_guess_time_ > 0.0) {
        keep_from_t = std::min(last_imu_guess_time_, scan_beg_time);
    }

    // Drop IMUs that are too old. Keep at most ONE sample at/before keep_from_t.
    while (imu_que_opt.size() > 1 && rclcpp::Time(imu_que_opt[1].header.stamp).seconds() <= keep_from_t) {
        imu_que_opt.pop_front();
    }
    if (imu_que_opt.empty()) {
        imu_ptr_cur = 0;
        return;
    }

    // Initialize integration at scan_beg_time with zero rotation.
    imu_ptr_cur = 0;
    imu_rot_x[0] = 0;
    imu_rot_y[0] = 0;
    imu_rot_z[0] = 0;
    imu_time[0] = scan_beg_time;

    for (int i = 0; i < (int)imu_que_opt.size(); ++i) {
        sensor_msgs::msg::Imu& imu_msg = imu_que_opt[i];
        double t = rclcpp::Time(imu_msg.header.stamp).seconds();

        if (t <= scan_beg_time) continue;
        if (t > scan_end_time + 0.01) break;

        double dt = t - imu_time[imu_ptr_cur];
        
        if (imu_ptr_cur + 1 >= 2000) break;
        imu_ptr_cur++;

        imu_rot_x[imu_ptr_cur] = imu_rot_x[imu_ptr_cur - 1] + imu_msg.angular_velocity.x * dt;
        imu_rot_y[imu_ptr_cur] = imu_rot_y[imu_ptr_cur - 1] + imu_msg.angular_velocity.y * dt;
        imu_rot_z[imu_ptr_cur] = imu_rot_z[imu_ptr_cur - 1] + imu_msg.angular_velocity.z * dt;
        imu_time[imu_ptr_cur] = t;
    }
}

void LocalMap::findRotation(double relTime, float *rotXCur, float *rotYCur, float *rotZCur) {
    *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
    int i = 0;
    while (i < imu_ptr_cur && relTime > imu_time[i] - scan_beg_time) {
        i++;
    }

    if (i == 0 || relTime > imu_time[i] - scan_beg_time) {
        *rotXCur = imu_rot_x[i];
        *rotYCur = imu_rot_y[i];
        *rotZCur = imu_rot_z[i];
    } else {
        double ratio = (relTime - (imu_time[i - 1] - scan_beg_time)) / (imu_time[i] - imu_time[i - 1]);
        *rotXCur = imu_rot_x[i - 1] + ratio * (imu_rot_x[i] - imu_rot_x[i - 1]);
        *rotYCur = imu_rot_y[i - 1] + ratio * (imu_rot_y[i] - imu_rot_y[i - 1]);
        *rotZCur = imu_rot_z[i - 1] + ratio * (imu_rot_z[i] - imu_rot_z[i - 1]);
    }
}

void LocalMap::findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur) {
    *posXCur = 0; *posYCur = 0; *posZCur = 0;
    // 简化处理，不考虑位移去畸变，或者假设匀速
}

void LocalMap::deskewPoint(PointType *point, double relTime) {
    if (deskewByImu == false || imu_ptr_cur <= 0) return;

    float rotXCur, rotYCur, rotZCur;
    findRotation(relTime, &rotXCur, &rotYCur, &rotZCur);

    float posXCur, posYCur, posZCur;
    findPosition(relTime, &posXCur, &posYCur, &posZCur);

    Eigen::Affine3f transData = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
    Eigen::Vector3f p(point->x, point->y, point->z);
    Eigen::Vector3f p_deskewed = transData * p;

    point->x = p_deskewed.x();
    point->y = p_deskewed.y();
    point->z = p_deskewed.z();
}

PointType LocalMap::lidarToImu(const PointType& p) {
    PointType dst = p;
    Eigen::Vector3d pt_lidar(p.x, p.y, p.z);
    Eigen::Vector3d pt_imu = extRot * pt_lidar + extTrans;
    dst.x = pt_imu.x();
    dst.y = pt_imu.y();
    dst.z = pt_imu.z();
    return dst;
}

Eigen::Isometry3d LocalMap::makeImuInitialGuessIsometry_(const Eigen::Isometry3d& T_last, double t_last, double t_curr) const {
    // slam thread only
    if (imu_que_opt.empty()) {
        return T_last;
    }

    if (t_last <= 0.0 || t_curr <= 0.0 || t_curr <= t_last) {
        return T_last;
    }

    const double dt_total = t_curr - t_last;

    // integrate gyro (rad/s) from (t_last, t_curr] using imu_que_opt
    // Note: sensor_msgs/Imu angular_velocity is in IMU frame.
    Eigen::Quaterniond q_delta = Eigen::Quaterniond::Identity();

    // integrate accel (m/s^2) for short dt to predict translation in ego frame
    // We integrate in a very conservative way:
    // - only when dt_total is small (<= imu_accel_integ_max_dt_)
    // - remove gravity using a fixed gravity magnitude along IMU +Z (approx)
    // - final result is clamped and fused with constant-velocity fallback
    Eigen::Vector3d delta_p_ego_from_acc = Eigen::Vector3d::Zero();
    bool used_acc_integration = false;

    // find first imu with stamp > t_last
    size_t idx = 0;
    while (idx < imu_que_opt.size() && rclcpp::Time(imu_que_opt[idx].header.stamp).seconds() <= t_last) {
        ++idx;
    }

    double t_prev = t_last;

    // for accel integration (ego frame)
    Eigen::Vector3d v_ego = vel_ego_;  // start from last estimated velocity (ego)
    const Eigen::Matrix3d R_ego_body = T_ego_body.linear().cast<double>();

    for (; idx < imu_que_opt.size(); ++idx) {
        const auto& imu_msg = imu_que_opt[idx];
        const double t = rclcpp::Time(imu_msg.header.stamp).seconds();
        if (t > t_curr) {
            break;
        }

        const double dt = t - t_prev;
        if (dt <= 0.0) {
            t_prev = t;
            continue;
        }

        // --- rotation integration ---
        const Eigen::Vector3d w(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z);
        const Eigen::Vector3d w_unbias = w - gyro_bias_radps_;
        const Eigen::Vector3d dtheta = w_unbias * dt;
        const double angle = dtheta.norm();
        if (angle > 1e-12) {
            const Eigen::Vector3d axis = dtheta / angle;
            const Eigen::AngleAxisd aa(angle, axis);
            q_delta = q_delta * Eigen::Quaterniond(aa);
        }

        // --- translation integration (short dt only) ---
        if (dt_total <= imu_accel_integ_max_dt_) {
            // IMU linear acceleration is in IMU/body frame. Convert to ego frame.
            // IMU linear_acceleration is in unit of g (NOT m/s^2) for this device.
            // Convert to m/s^2 first.
            Eigen::Vector3d a_body(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z);
            a_body *= 9.80665;

            // Gravity/bias removal using IMU calibration (gravity direction in body frame).
            a_body -= acc_bias_mps2_;
            a_body -= gravity_mag_mps2_ * gravity_dir_body_unit_;

            const Eigen::Vector3d a_ego = R_ego_body * a_body;
            v_ego += a_ego * dt;
            delta_p_ego_from_acc += v_ego * dt;
            used_acc_integration = true;
        }

        t_prev = t;
    }

    // if last imu < t_curr, extend with last gyro / accel
    if (t_prev < t_curr && idx > 0) {
        const auto& imu_last = imu_que_opt[std::min(idx, imu_que_opt.size() - 1)];
        const double dt = t_curr - t_prev;
        if (dt > 0.0) {
            // rotation
            const Eigen::Vector3d w(imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z);
            const Eigen::Vector3d w_unbias = w - gyro_bias_radps_;
            const Eigen::Vector3d dtheta = w_unbias * dt;
            const double angle = dtheta.norm();
            if (angle > 1e-12) {
                const Eigen::Vector3d axis = dtheta / angle;
                const Eigen::AngleAxisd aa(angle, axis);
                q_delta = q_delta * Eigen::Quaterniond(aa);
            }

            // translation (short dt only)
            if (dt_total <= imu_accel_integ_max_dt_) {
                Eigen::Vector3d a_body(imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z);
                a_body *= 9.80665;
                a_body -= acc_bias_mps2_;
                a_body -= gravity_mag_mps2_ * gravity_dir_body_unit_;
                const Eigen::Vector3d a_ego = R_ego_body * a_body;
                v_ego += a_ego * dt;
                delta_p_ego_from_acc += v_ego * dt;
                used_acc_integration = true;
            }
        }
    }

    // Apply delta rotation in odom frame: R_pred = R_last * R_delta
    Eigen::Isometry3d T_pred = T_last;
    T_pred.linear() = (T_last.linear() * q_delta.toRotationMatrix());

    // --- translation prediction ---
    // Default: constant-velocity in ego frame (vel_ego_) with speed clamp.
    Eigen::Vector3d vel_ego_clamped = vel_ego_;
    const double speed = vel_ego_clamped.head<2>().norm();
    if (speed > max_ego_speed_mps_) {
        vel_ego_clamped *= (max_ego_speed_mps_ / speed);
    }
    Eigen::Vector3d delta_p_ego_cv = vel_ego_clamped * dt_total;

    // Fuse: use accel integration only for short dt, otherwise CV.
    Eigen::Vector3d delta_p_ego = delta_p_ego_cv;
    if (used_acc_integration) {
        delta_p_ego = delta_p_ego_from_acc;
    }

    // Convert ego translation increment to odom using current ego orientation (from T_last).
    // odom<-ego rotation:
    const Eigen::Matrix3d R_odom_body = T_last.linear();
    const Eigen::Matrix3d R_body_ego = R_ego_body.transpose();
    const Eigen::Matrix3d R_odom_ego = R_odom_body * R_body_ego;
    T_pred.translation() = T_last.translation() + R_odom_ego * delta_p_ego;

    return T_pred;
}

void LocalMap::pointCloudPreprocessing() {
    if (deskewByImu) {
        for (int i = 0; i < (int)laser_cloud_in->size(); ++i) {
            PointType &p = laser_cloud_in->points[i];
            double relTime = p.normal_y; 
            deskewPoint(&p, relTime);
        }
    }

    laser_cloud_in_ds = small_gicp::voxelgrid_sampling(*laser_cloud_in, 0.2);
}

void LocalMap::performOdometer() {
    if (is_first_frame) {
        // 第一帧：将其变换到世界坐标系并存入 last_laser_cloud_in 作为下一帧的 target
        last_laser_cloud_in->clear();
        pcl::transformPointCloud(*laser_cloud_in_ds, *last_laser_cloud_in, current_pose);
        is_first_frame = false;
        return;
    }

    if (laser_cloud_in_ds->empty() || last_laser_cloud_in->empty()) return;

    // 策略：将当前帧（Source）与上一帧配准后的点云（Target）进行配准
    // 局部实例化 VGICP 对象，确保每次配准都是干净的状态，避免 out_of_range 错误
    small_gicp::RegistrationPCL<PointType, PointType> vgicp;
    vgicp.setRegistrationType("VGICP");
    vgicp.setVoxelResolution(1.0);
    vgicp.setNumThreads(4);
    vgicp.setMaxCorrespondenceDistance(1.0);
    vgicp.setNumNeighborsForCovariance(40);
    vgicp.setMaximumIterations(100);

    vgicp.setInputSource(laser_cloud_in_ds);
    vgicp.setInputTarget(last_laser_cloud_in);

    // 执行配准
    pcl::PointCloud<PointType>::Ptr aligned_cloud(new pcl::PointCloud<PointType>());
    
    try {
        // 使用当前位姿作为初始猜测进行配准
        vgicp.align(*aligned_cloud, current_pose.matrix());

        if (vgicp.hasConverged()) {
            // 更新当前位姿 (直接获得在世界坐标系下的位姿)
            current_pose = vgicp.getFinalTransformation();
        } else {
            RCLCPP_WARN(get_logger(), "VGICP did not converge!");
        }
    } catch (const std::out_of_range& e) {
        RCLCPP_ERROR(get_logger(), "VGICP align caught out_of_range: %s. Source size: %zu, Target size: %zu", 
                     e.what(), laser_cloud_in_ds->size(), last_laser_cloud_in->size());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "VGICP align caught exception: %s", e.what());
    }

    // 将当前帧配准后的点云存入 last_laser_cloud_in，作为下一帧的 target
    last_laser_cloud_in->clear();
    pcl::transformPointCloud(*laser_cloud_in_ds, *last_laser_cloud_in, current_pose);
}

void LocalMap::performOdometer_v1() {
    // keep all key frames in local map
    static int cnt = 1;
    if (is_first_frame) {
        pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laser_cloud_in_ds, *cloud_world, current_pose);
        *local_map += *cloud_world;
        last_key_pose = current_pose;
        is_first_frame = false;
        return;
    }

    if (laser_cloud_in_ds->empty() || local_map->empty()) return;

    small_gicp::RegistrationPCL<PointType, PointType> vgicp;
    vgicp.setRegistrationType("VGICP");
    vgicp.setVoxelResolution(1.0);
    vgicp.setNumThreads(4);
    vgicp.setMaxCorrespondenceDistance(1.0);
    vgicp.setNumNeighborsForCovariance(20);
    vgicp.setMaximumIterations(100);

    vgicp.setInputSource(laser_cloud_in_ds);
    vgicp.setInputTarget(local_map);

    pcl::PointCloud<PointType>::Ptr aligned_cloud(new pcl::PointCloud<PointType>());
    
    try {
        // 使用当前位姿作为初始猜测进行配准
        vgicp.align(*aligned_cloud, current_pose.matrix());

        if (vgicp.hasConverged()) {
            current_pose = vgicp.getFinalTransformation();
        } else {
            RCLCPP_WARN(get_logger(), "VGICP did not converge!");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "VGICP align caught exception: %s", e.what());
    }

    // keyframe condition: translation exceeds 0.3 m or rotation exceeds 10 degrees
    float delta_dist = (current_pose.translation() - last_key_pose.translation()).norm();
    Eigen::Quaternionf q_curr(current_pose.linear());
    Eigen::Quaternionf q_last(last_key_pose.linear());
    float delta_angle = q_last.angularDistance(q_curr) * 180.0 / M_PI;

    if (delta_dist > 0.3 || delta_angle > 10.0) {
        pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laser_cloud_in_ds, *cloud_world, current_pose);
        *local_map += *cloud_world;

        // source and target should have the same resolution for vgicp
        local_map = small_gicp::voxelgrid_sampling(*local_map, 0.2);

        if constexpr (LOCAL_DEBUG) {
            // std::string output_pcd_path = "/home/weizh/data/local_map_" + std::to_string(cnt) + ".pcd";
            // pcl::io::savePCDFileBinary(output_pcd_path, *local_map);
            // ++cnt;
        }

        last_key_pose = current_pose;
    }

    updatePath(poseToPose6D(current_pose));
}

void LocalMap::performOdometer_v2() {
    // drop key frames in local map by sliding window
    static int cnt = 1;
    if (is_first_frame) {
        pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laser_cloud_in_ds, *cloud_world, current_pose);
        *local_map += *cloud_world;
        last_key_pose = current_pose;
        is_first_frame = false;
        return;
    }

    if (laser_cloud_in_ds->empty() || local_map->empty()) return;

    small_gicp::RegistrationPCL<PointType, PointType> vgicp;
    vgicp.setRegistrationType("VGICP");
    vgicp.setVoxelResolution(1.0);
    vgicp.setNumThreads(4);
    vgicp.setMaxCorrespondenceDistance(1.0);
    vgicp.setNumNeighborsForCovariance(20);
    vgicp.setMaximumIterations(100);

    vgicp.setInputSource(laser_cloud_in_ds);
    vgicp.setInputTarget(local_map);

    // to be tested, consider the case when mower moves very fast and local map increases drastically within short time
    // pcl::CropBox<PointType> crop;
    // float range = 50.0f; 
    // crop.setMin(Eigen::Vector4f(current_pose.translation().x() - range, current_pose.translation().y() - range, current_pose.translation().z() - range, 1.0));
    // crop.setMax(Eigen::Vector4f(current_pose.translation().x() + range, current_pose.translation().y() + range, current_pose.translation().z() + range, 1.0));
    // crop.setInputCloud(local_map);
    // pcl::PointCloud<PointType>::Ptr cropped_map(new pcl::PointCloud<PointType>());
    // crop.filter(*cropped_map);
    // vgicp.setInputTarget(cropped_map);

    pcl::PointCloud<PointType>::Ptr aligned_cloud(new pcl::PointCloud<PointType>());
    
    try {
        // 使用当前位姿作为初始猜测进行配准
        vgicp.align(*aligned_cloud, current_pose.matrix());

        if (vgicp.hasConverged()) {
            current_pose = vgicp.getFinalTransformation();
        } else {
            RCLCPP_WARN(get_logger(), "VGICP did not converge!");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "VGICP align caught exception: %s", e.what());
    }

    // keyframe condition: translation exceeds 0.3 m or rotation exceeds 10 degrees
    float delta_dist = (current_pose.translation() - last_key_pose.translation()).norm();
    Eigen::Quaternionf q_curr(current_pose.linear());
    Eigen::Quaternionf q_last(last_key_pose.linear());
    float delta_angle = q_last.angularDistance(q_curr) * 180.0 / M_PI;

    if (delta_dist > 0.3 || delta_angle > 10.0) {
        pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laser_cloud_in_ds, *cloud_world, current_pose);

        KeyFrame kf;
        kf.timestamp = current_lidar_data_.lidar_frame_beg_time;
        kf.pose = current_pose;
        kf.cloud = cloud_world;
        keyframe_queue.push_back(kf);

        // drop frames 15 seconds ago
        const double current_time = current_lidar_data_.lidar_frame_beg_time;
        while (!keyframe_queue.empty() && (current_time - keyframe_queue.front().timestamp > 15.0)) {
            keyframe_queue.pop_front();
        }

        // reconstruct local_map
        local_map->clear();
        for (const auto& frame : keyframe_queue) {
            *local_map += *(frame.cloud);
        }

        // source and target should have the same resolution for vgicp
        local_map = small_gicp::voxelgrid_sampling(*local_map, 0.2);

        if constexpr (LOCAL_DEBUG) {
            // std::string output_pcd_path = "/home/weizh/data/local_map_" + std::to_string(cnt) + ".pcd";
            // pcl::io::savePCDFileBinary(output_pcd_path, *local_map);
            // ++cnt;
        }

        last_key_pose = current_pose;
    }

    updatePath(poseToPose6D(current_pose));
}

small_gicp::PointCloud::Ptr LocalMap::convertToSmallGICP(pcl::PointCloud<PointType>::Ptr pcl_cloud) {
    auto out = std::make_shared<small_gicp::PointCloud>();
    out->resize(pcl_cloud->size());
    for (size_t i = 0; i < pcl_cloud->size(); ++i) {
        const auto& pt = pcl_cloud->points[i];
        out->point(i) << pt.x, pt.y, pt.z, 1.0;
    }
    return out;
}

PointTypePose LocalMap::poseToPose6D(const Eigen::Affine3f& pose) const {
    PointTypePose p;
    const Eigen::Matrix3f R = pose.rotation();
    p.x = pose.translation().x();
    p.y = pose.translation().y();
    p.z = pose.translation().z();
    p.intensity = 0;
    p.roll = atan2(R(2, 1), R(2, 2));
    p.pitch = atan2(-R(2, 0), sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
    p.yaw = atan2(R(1, 0), R(0, 0));
    p.time = scan_beg_time;
    return p;
}

void LocalMap::performOdometer_v3() {
    if (laser_cloud_in_ds->empty()) return;

    // --- last pose (float->double) ---
    Eigen::Isometry3d T_last = Eigen::Isometry3d::Identity();
    T_last.linear()      = current_pose.linear().cast<double>();
    T_last.translation() = current_pose.translation().cast<double>();

    // --- IMU coverage / gap check for initial guess ---
    // We require enough IMU samples in (last_imu_guess_time_, scan_end_time] and no large gaps.
    // Otherwise, the IMU-based initial guess is unreliable (especially during sharp turns),
    // so we reset local map and restart odom from current frame.
    bool imu_ok_for_guess = true;
    double max_gap = 0.0;
    int imu_cnt = 0;

    if (last_imu_guess_time_ > 0.0) {
        double t_prev = -1.0;
        for (const auto& imu_msg : imu_que_opt) {
            const double t = rclcpp::Time(imu_msg.header.stamp).seconds();
            if (t <= last_imu_guess_time_) continue;
            if (t > scan_end_time) break;

            imu_cnt++;
            if (t_prev > 0.0) {
                const double gap = t - t_prev;
                if (gap > max_gap) max_gap = gap;
            }
            t_prev = t;
        }

        if (imu_cnt < imu_min_samples_for_guess_ || max_gap > imu_gap_reset_thresh_s_) {
            imu_ok_for_guess = false;
        }
    }

    if (!imu_ok_for_guess) {
        RCLCPP_ERROR(get_logger(), "IMU coverage insufficient for lidar %lf, reset local map", scan_beg_time);

        // reset local map (restart odom from current frame)
        keyframe_queue.clear();
        voxel_target.reset();
        local_map->clear();
        last_key_pose = current_pose;

        // reset IMU-guess state
        last_imu_guess_time_ = -1.0;
        has_prev_pose_for_vel_ = false;
        prev_pose_time_ = -1.0;
        prev_pose_T_ = Eigen::Isometry3d::Identity();
        vel_ego_.setZero();
    }

    // --- IMU initial guess (rotation + translation prediction) ---
    Eigen::Isometry3d T_curr = T_last;
    if (voxel_target != nullptr && last_imu_guess_time_ > 0.0) {
        // Only use IMU prediction when map is already initialized and we have a valid previous timestamp.
        T_curr = makeImuInitialGuessIsometry_(T_last, last_imu_guess_time_, scan_end_time);
    }
    last_imu_guess_time_ = scan_end_time;

    // Cache the initial guess
    initial_guess_pose_ = Eigen::Affine3f::Identity();
    initial_guess_pose_.linear() = T_curr.linear().cast<float>();
    initial_guess_pose_.translation() = T_curr.translation().cast<float>();

    // --- source 转换 + 协方差估计 ---
    // laser_cloud_in_ds 已经是 0.2m 下采样，直接用
    auto source_cloud = convertToSmallGICP(laser_cloud_in_ds);
    small_gicp::estimate_covariances_omp(*source_cloud, 20, num_threads);

    // --- 第一帧：初始化 voxel_target ---
    if (voxel_target == nullptr) {
        pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laser_cloud_in_ds, *cloud_world, current_pose);

        KeyFrame kf;
        kf.timestamp = scan_beg_time;
        kf.cloud_new  = convertToSmallGICP(cloud_world);
        // 世界系下估计协方差，insert 后体素协方差才有效
        small_gicp::estimate_covariances_omp(*kf.cloud_new, 20, num_threads);
        keyframe_queue.push_back(kf);
        voxel_target = std::make_shared<small_gicp::GaussianVoxelMap>(0.3);
        voxel_target->insert(*kf.cloud_new);
        // RCLCPP_INFO(get_logger(), "voxel_target init: voxels=%zu", voxel_target->size());
        last_key_pose = current_pose;
        updatePath(poseToPose6D(current_pose));
        return;
    }

    // --- 配准：Scan-to-Model ---
    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> registration;
    registration.reduction.num_threads  = num_threads;
    registration.optimizer.max_iterations = 50;
    // 草地场景收敛阈值可以稍松，避免迭代过多
    registration.criteria.translation_eps = 1e-3;
    registration.criteria.rotation_eps    = 1e-3;
    // registration.rejector.max_dist_sq = 1.0; // not improved

    auto result = registration.align(
        *voxel_target,   // target: GaussianVoxelMap，traits::cov 返回体素协方差
        *source_cloud,   // source: 带法向协方差的点云
        *voxel_target,   // 近邻搜索结构复用 target
        T_curr           // 初始猜测
    );

    if (result.converged) {
        current_pose = result.T_target_source.cast<float>();

        // Update ego velocity estimate for translation prediction (constant-velocity fallback).
        // Use last successful pose pair to estimate v_ego.
        const double t_pose = scan_end_time;
        const Eigen::Isometry3d T_pose = result.T_target_source;  // odom<-body (registered)

        if (!has_prev_pose_for_vel_) {
            prev_pose_T_ = T_pose;
            prev_pose_time_ = t_pose;
            has_prev_pose_for_vel_ = true;
        } else {
            const double dt_vel = t_pose - prev_pose_time_;
            if (dt_vel > 1e-3) {
                // delta position in odom
                const Eigen::Vector3d dp_odom = T_pose.translation() - prev_pose_T_.translation();

                // Convert to ego frame using current pose orientation.
                const Eigen::Matrix3d R_odom_body = T_pose.linear();
                const Eigen::Matrix3d R_ego_body = T_ego_body.linear().cast<double>();
                const Eigen::Matrix3d R_odom_ego = R_odom_body * R_ego_body.transpose();
                const Eigen::Vector3d dp_ego = R_odom_ego.transpose() * dp_odom;

                vel_ego_ = dp_ego / dt_vel;

                // clamp speed
                const double speed_xy = vel_ego_.head<2>().norm();
                if (speed_xy > max_ego_speed_mps_) {
                    vel_ego_ *= (max_ego_speed_mps_ / speed_xy);
                }
            }

            prev_pose_T_ = T_pose;
            prev_pose_time_ = t_pose;
        }
    } else {
        RCLCPP_WARN(get_logger(), "VGICP did not converge!");
    }

    // --- 关键帧判断 ---
    float delta_dist  = (current_pose.translation() - last_key_pose.translation()).norm();
    float delta_angle = Eigen::Quaternionf(last_key_pose.linear())
                            .angularDistance(Eigen::Quaternionf(current_pose.linear()))
                        * 180.0f / M_PI;

    if (delta_dist > 0.3f || delta_angle > 10.0f) {
        // 1. 当前帧转到世界系
        pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laser_cloud_in_ds, *cloud_world, current_pose);

        // 2. 入队
        KeyFrame kf;
        kf.timestamp = scan_beg_time;
        kf.cloud_new = convertToSmallGICP(cloud_world);
        // 关键：世界系下估计协方差，体素聚合后协方差才反映真实局部几何
        small_gicp::estimate_covariances_omp(*kf.cloud_new, 20, num_threads);
        keyframe_queue.push_back(kf);

        // remove key frames older than 10 sec
        while (!keyframe_queue.empty() && (scan_beg_time - keyframe_queue.front().timestamp > 10.0)) {
            keyframe_queue.pop_front();
        }

        // 4. 重建 voxel_target
        // GaussianVoxelMap 不支持删除，只能整体重建，但速度很快
        voxel_target = std::make_shared<small_gicp::GaussianVoxelMap>(0.3);
        for (const auto& frame : keyframe_queue) {
            voxel_target->insert(*frame.cloud_new);
            // frame.cloud_new 已经在入队时估好协方差，直接 insert 即可
        }
        // RCLCPP_INFO(get_logger(), "voxel_target rebuild: keyframes=%zu voxels=%zu", keyframe_queue.size(), voxel_target->size());

        last_key_pose = current_pose;
    }

    updatePath(poseToPose6D(current_pose));
}

void LocalMap::performOdometer_v4() {
    if (laser_cloud_in_ds->empty()) return;

    // --- last pose (float->double) ---
    Eigen::Isometry3d T_last = Eigen::Isometry3d::Identity();
    T_last.linear()      = current_pose.linear().cast<double>();
    T_last.translation() = current_pose.translation().cast<double>();

    // --- IMU initial guess (rotation prediction) ---
    // Use scan_end_time as pose timestamp proxy; keep translation unchanged.
    Eigen::Isometry3d T_curr = T_last;
    if (last_imu_guess_time_ > 0.0) {
        T_curr = makeImuInitialGuessIsometry_(T_last, last_imu_guess_time_, scan_end_time);
    }
    last_imu_guess_time_ = scan_end_time;

    // Cache the initial guess
    initial_guess_pose_ = Eigen::Affine3f::Identity();
    initial_guess_pose_.linear() = T_curr.linear().cast<float>();
    initial_guess_pose_.translation() = T_curr.translation().cast<float>();

    // --- source 转换 + 协方差估计 ---
    // laser_cloud_in_ds 已经是 0.2m 下采样，直接用
    auto source_cloud = convertToSmallGICP(laser_cloud_in_ds);
    small_gicp::estimate_covariances_omp(*source_cloud, 20, num_threads);

    // --- 第一帧：初始化 voxel_target ---
    if (voxel_target == nullptr) {
        pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laser_cloud_in_ds, *cloud_world, current_pose);

        KeyFrame kf;
        kf.timestamp = scan_beg_time;
        kf.cloud_new  = convertToSmallGICP(cloud_world);
        // 世界系下估计协方差，insert 后体素协方差才有效
        small_gicp::estimate_covariances_omp(*kf.cloud_new, 20, num_threads);
        keyframe_queue.push_back(kf);
        voxel_target = std::make_shared<small_gicp::GaussianVoxelMap>(0.3);
        voxel_target->insert(*kf.cloud_new);
        last_key_pose = current_pose;
        updatePath(poseToPose6D(current_pose));
        return;
    }

    // --- 配准：Scan-to-Model ---
    // Optimizer update rule (see small_gicp/registration/optimizer.hpp):
    //   T <- T * exp([w, v])   (right-multiplicative increment)
    // so the twist [w, v] is expressed in the SOURCE(local) frame.
    MotionPriorGeneralFactor gp;
    gp.lambda_rot = enable_motion_prior_ ? motion_prior_rot_lambda_ : 0.0;
    gp.lambda_lat = enable_motion_prior_ ? motion_prior_lat_lambda_ : 0.0;
    gp.R_ego_body = T_ego_body.linear().cast<double>();
    gp.ego_x_offset = motion_prior_ego_x_offset_;
    gp.ego_y_offset = motion_prior_ego_y_offset_;

    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP, MotionPriorGeneralFactor> registration;
    registration.reduction.num_threads  = num_threads;
    registration.optimizer.max_iterations = 50;
    // 草地场景收敛阈值可以稍松，避免迭代过多
    registration.criteria.translation_eps = 1e-3;
    registration.criteria.rotation_eps    = 1e-3;
    registration.general_factor = gp;
    // registration.rejector.max_dist_sq = 1.0; // not improved

    auto result = registration.align(
        *voxel_target,   // target: GaussianVoxelMap，traits::cov 返回体素协方差
        *source_cloud,   // source: 带法向协方差的点云
        *voxel_target,   // 近邻搜索结构复用 target
        T_curr           // 初始猜测
    );

    if (result.converged) {
        current_pose = result.T_target_source.cast<float>();
    } else {
        RCLCPP_WARN(get_logger(), "VGICP did not converge!");
    }

    // --- 关键帧判断 ---
    float delta_dist  = (current_pose.translation() - last_key_pose.translation()).norm();
    float delta_angle = Eigen::Quaternionf(last_key_pose.linear())
                            .angularDistance(Eigen::Quaternionf(current_pose.linear()))
                        * 180.0f / M_PI;

    if (delta_dist > 0.3f || delta_angle > 10.0f) {
        // 1. 当前帧转到世界系
        pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laser_cloud_in_ds, *cloud_world, current_pose);

        // 2. 入队
        KeyFrame kf;
        kf.timestamp = scan_beg_time;
        kf.cloud_new = convertToSmallGICP(cloud_world);
        // 关键：世界系下估计协方差，体素聚合后协方差才反映真实局部几何
        small_gicp::estimate_covariances_omp(*kf.cloud_new, 20, num_threads);
        keyframe_queue.push_back(kf);

        // remove key frames older than 10 sec
        while (!keyframe_queue.empty() && (scan_beg_time - keyframe_queue.front().timestamp > 10.0)) {
            keyframe_queue.pop_front();
        }

        // 4. 重建 voxel_target
        // GaussianVoxelMap 不支持删除，只能整体重建，但速度很快
        voxel_target = std::make_shared<small_gicp::GaussianVoxelMap>(0.3);
        for (const auto& frame : keyframe_queue) {
            voxel_target->insert(*frame.cloud_new);
            // frame.cloud_new 已经在入队时估好协方差，直接 insert 即可
        }

        last_key_pose = current_pose;
    }

    updatePath(poseToPose6D(current_pose));
}

void LocalMap::updatePath(const PointTypePose& pose_in) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = current_lidar_data_.stamp;
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf2::Quaternion q;
    q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    std::lock_guard<std::mutex> lk(mtx_path_);
    globalPath.poses.push_back(pose_stamped);
}

void LocalMap::publishObstacleMapTimerCb() {
    // LOCAL_DEBUG=true: publish odom-frame voxel map, false: ego frame
    if (!has_latest_snapshot_) return;

    PublishSnapshot snap;
    {
        std::lock_guard<std::mutex> lk(mtx_latest_snapshot_);
        if (!has_latest_snapshot_) {
            // in case timer starts before slam produces the first snapshot.
            return;
        }
        snap = latest_snapshot_;
    }

    const rclcpp::Time pb_stamp(static_cast<int64_t>(snap.stamp * 1e9));
    const auto& voxels_snapshot = snap.voxels_snapshot;
    pcl::PointCloud<pcl::PointXYZI>::Ptr voxel_cloud_out(new pcl::PointCloud<pcl::PointXYZI>());
    voxel_cloud_out->reserve(voxels_snapshot.size());

    std::string out_frame_id = odometryFrame;

    if constexpr (!LOCAL_DEBUG) {
        //  ego <- lidar <- body <- odom
        const Eigen::Affine3f T_ego_odom = T_ego_body * snap.T_odom_body.inverse();
        out_frame_id = egoFrame;
        for (const auto& voxel : voxels_snapshot) {
            Eigen::Vector3f p_odom(voxel.x, voxel.y, voxel.z);
            Eigen::Vector3f p_ego = T_ego_odom * p_odom;
            pcl::PointXYZI pt;
            pt.x = p_ego.x();
            pt.y = p_ego.y();
            pt.z = p_ego.z();
            pt.intensity = std::min(255.0f, voxel.observation_count * 50.0f);
            voxel_cloud_out->points.push_back(pt);
        }
    } else {
        for (const auto& voxel : voxels_snapshot) {
            pcl::PointXYZI pt;
            pt.x = voxel.x;
            pt.y = voxel.y;
            pt.z = voxel.z;
            pt.intensity = std::min(255.0f, voxel.observation_count * 50.0f);
            voxel_cloud_out->points.push_back(pt);
        }
    }
    voxel_cloud_out->width = voxel_cloud_out->size();
    voxel_cloud_out->height = 1;
    voxel_cloud_out->is_dense = true;

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*voxel_cloud_out, cloud_msg);
    cloud_msg.header.stamp = pb_stamp;
    cloud_msg.header.frame_id = out_frame_id;
    pub_obstacle_map->publish(cloud_msg);

    // generate protobuf
    pb_cloud_.clear_voxels();
    pb_cloud_.set_timestamp(static_cast<uint64_t>(pb_stamp.nanoseconds()));
    pb_cloud_.set_resolution(0.1f);
    pb_cloud_.mutable_voxels()->Reserve(static_cast<int>(voxel_cloud_out->size()));
    for (const auto& pt : voxel_cloud_out->points) {
        auto* v = pb_cloud_.add_voxels();
        v->set_x(pt.x);
        v->set_y(pt.y);
        v->set_z(pt.z);
    }

    // publish protobuf bytes via MQTT
    try {
        pb_cloud_.SerializeToString(&mqtt_payload_buffer_);
        if (!mqtt_payload_buffer_.empty()) {
            RCLCPP_INFO(this->get_logger(), "Attempting MQTT Publish...");
            mqtt_client_.publish("lidar/data", mqtt_payload_buffer_, 0, false);  // qos=0: do not wait for confirm, 1: at least once
            RCLCPP_INFO(this->get_logger(), "MQTT Publish returned.");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to send MQTT message: %s", e.what());
    }

    // debug, save protobuf to file, note if bag plays at rate < 1, it overwrites pb files
    // const std::string out_path = std::string("/home/weizh/data/voxel_cloud_") + std::to_string(pb_cloud_.timestamp()) + ".pb";
    // std::ofstream ofs(out_path, std::ios::out | std::ios::binary | std::ios::trunc);
    // if (ofs.is_open()) {
    //     pb_cloud_.SerializeToOstream(&ofs);
    //     ofs.close();
    // }
}

void LocalMap::publishResult() {
    // publish odom, path, registered cloud; obstacle_voxel_map is published by timer (see publishObstacleMapTimerCb)
    // only called in debug, since release requires specific data format (data consumer not ros based)
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = current_lidar_data_.stamp;
    odom.header.frame_id = odometryFrame;
    odom.child_frame_id = bodyFrame;

    Eigen::Matrix3f rot = current_pose.linear();
    Eigen::Quaternionf q(rot);
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.position.x = current_pose.translation().x();
    odom.pose.pose.position.y = current_pose.translation().y();
    odom.pose.pose.position.z = current_pose.translation().z();
    pub_odom->publish(odom);

    // publish path
    if (pub_path->get_subscription_count() != 0) {
        globalPath.header.stamp = current_lidar_data_.stamp;
        globalPath.header.frame_id = odometryFrame;
        pub_path->publish(globalPath);
    }

    pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud_in, *cloud_world, current_pose);
    publishCloud(pub_cloud_registered, cloud_world, current_lidar_data_.stamp, odometryFrame);
    pcl::PointCloud<PointType>::Ptr cloud_guess(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud_in, *cloud_guess, initial_guess_pose_);
    publishCloud(pub_initial_guess, cloud_guess, current_lidar_data_.stamp, odometryFrame);

    if (pub_local_map->get_subscription_count() != 0) {
        publishCloud(pub_local_map, local_map, current_lidar_data_.stamp, odometryFrame);
    }
}

void LocalMap::removeDynamicObjTest() {
    // offline test
    // current problem:
    // 1. a tall dynamic object cannot be fully identified as dynamic, the lower part is traversed by ray, but the higher part has no points behind it
    // 2. voxel x y z might exceed the boundary of original object, thus there might be points behind it even though it is not dynamic.
    // assume dynamicObj is from pre lidar frames, rs_cur_cloud is current lidar frame, both in global frame
    pcl::PointCloud<RSPointDefault>::Ptr rs_cur_cloud(new pcl::PointCloud<RSPointDefault>);
    std::string rs_cur_cloud_fp = "/home/weizh/data/1769046781_303218842.pcd";
    pcl::io::loadPCDFile<RSPointDefault>(rs_cur_cloud_fp, *rs_cur_cloud);
    std::cout << "Loaded " << rs_cur_cloud->width * rs_cur_cloud->height << " data points from " << rs_cur_cloud_fp << std::endl;

    // load fake dynamic pcd, convert dynamicObj -> cluster cloud (XYZI), build a new obstacle_voxel_map
    // _fake_dynamic.pcd is obtained manually from cloudcompare by segment obstacle from 1769046781_303218842.pcd -> move -> delete scalar fields -> save, 
    // in _fake_dynamic.pcd, VIEWPOINT is not 0 0 0 1 0 0 0, x y z are still the values in 1769046781_303218842.pcd, cloud compare shows its x y z and position after VIEWPOINT calculation
    // so we need to convert dynamicObj to world frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr dynamicObj(new pcl::PointCloud<pcl::PointXYZ>);
    std::string dynamicObjFp = "/home/weizh/data/1769046781_303218842_fake_dynamic.pcd";
    pcl::io::loadPCDFile<pcl::PointXYZ>(dynamicObjFp, *dynamicObj);
    std::cout << "Loaded " << dynamicObj->width * dynamicObj->height << " data points from " << dynamicObjFp << std::endl;
    // convert
    // sensor_origin_ is Eigen::Vector4f (x, y, z, 1)
    // sensor_orientation_ is Eigen::Quaternionf (w, x, y, z)
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translate(dynamicObj->sensor_origin_.head<3>());
    transform.rotate(dynamicObj->sensor_orientation_);
    pcl::transformPointCloud(*dynamicObj, *dynamicObj, transform); // overwrite
    dynamicObj->sensor_origin_ = Eigen::Vector4f::Zero();
    dynamicObj->sensor_orientation_ = Eigen::Quaternionf::Identity();
    
    // test pitchToRow, this function needs to work properly. Result shows ring is OK, but hard to tell if good enough.
    pcl::PointCloud<RSPointDefault>::Ptr dynamicObjWithRow(new pcl::PointCloud<RSPointDefault>());
    dynamicObjWithRow->reserve(dynamicObj->size());
    for (const auto& p : dynamicObj->points) {
        RSPointDefault q;
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        float pitch_deg = std::atan2(p.z, std::sqrt(p.x*p.x + p.y*p.y)) * 180.0 / M_PI;
        q.ring = pitchToRow(pitch_deg);
        dynamicObjWithRow->push_back(q);
    }
    pcl::io::savePCDFileBinary("/home/weizh/data/1769046781_303218842_fake_dynamic_row.pcd", *dynamicObjWithRow);

    pcl::PointCloud<pcl::PointXYZI>::Ptr dyn_cluster(new pcl::PointCloud<pcl::PointXYZI>());
    dyn_cluster->reserve(dynamicObj->size());
    for (const auto& p : dynamicObj->points) {
        pcl::PointXYZI q;
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        q.intensity = 1.0f;
        dyn_cluster->push_back(q);
    }
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> obstacle_clusters;
    obstacle_clusters.push_back(dyn_cluster);
    obstacle_voxel_map.clear();
    const Eigen::Affine3f pose = Eigen::Affine3f::Identity();
    const double timestamp = 0.0;
    updateObstacleVoxelMap(obstacle_clusters, pose, timestamp);

    // build range image for both
    auto cloud_filtered_normal = filterByRangeEgo(rs_cur_cloud, max_range_);
    buildRangeImage(cloud_filtered_normal);

    // use ray casting to check if voxel is dynamic, since range image is based on current lidar frame, not global frame, need to convert voxel map to current lidar frame
    pcl::PointCloud<pcl::PointXYZI>::Ptr raycast_res(new pcl::PointCloud<pcl::PointXYZI>());
    raycast_res->reserve(obstacle_voxel_map.size());
    for (auto it = obstacle_voxel_map.begin(); it != obstacle_voxel_map.end(); ) {
        Voxel& v = it->second;

        // convert voxel from world to local frame, pt_world = pt_local in this test case
        Eigen::Vector3f pt_world(v.x, v.y, v.z);
        Eigen::Vector3f pt_local = pose.inverse() * pt_world;
        float d_v = std::sqrt(pt_local.x()*pt_local.x() + pt_local.y()*pt_local.y() + pt_local.z()*pt_local.z());
        
        // calculate coords in range image
        float azimuth = std::atan2(pt_local.y(), pt_local.x());
        if (azimuth < 0) azimuth += 2 * M_PI;
        int col = static_cast<int>(azimuth / (2 * M_PI) * hResolution);
        col = std::min(col, hResolution - 1);
        float pitch_deg = std::atan2(pt_local.z(), std::sqrt(pt_local.x()*pt_local.x() + pt_local.y()*pt_local.y())) * 180.0 / M_PI;
        int row = pitchToRow(pitch_deg);

        // visibility test
        bool is_traversed = false;  // is traversed by ray
        int window_size = 1;     // 考虑到 0.1m Voxel 和 1度线间距，查 3x3 窗口最鲁棒
        
        for (int r = row - window_size; r <= row + window_size; ++r) {
            for (int c = col - window_size; c <= col + window_size; ++c) {
                if (r < 0 || r >= nRing || c < 0 || c >= hResolution) continue;
                
                if (valid_mask_.at<uint8_t>(r, c) == 0) continue; // 没数据，跳过

                float d_obs = range_image_.at<float>(r, c);
                
                // 核心判定：如果观测距离明显大于 Voxel 距离，说明 Voxel 位置变空了
                if (d_obs > d_v + 0.15) { // 0.15m 为容忍偏差
                    is_traversed = true;
                    break;
                }
            }
            if (is_traversed) break;
        }

        pcl::PointXYZI q;
        q.x = pt_local.x();
        q.y = pt_local.y();
        q.z = pt_local.z();
        q.intensity = is_traversed ? 1.0f : 2.0f;  // is_traversed (dynamic) use intensity 1.0
        raycast_res->push_back(q);
        
        ++it;
    }

    pcl::io::savePCDFileBinary("/home/weizh/data/1769046781_303218842_static_vs_dynamic.pcd", *raycast_res);
}

void LocalMap::recordSlamLoopTotalMs_(double total_ms) {
    std::lock_guard<std::mutex> lk(mtx_timing_);
    slam_total_ms_.push_back(total_ms);
}

void LocalMap::dumpSlamLoopTotalCsv_(const std::string& out_path) {
    std::vector<double> snapshot;
    {
        std::lock_guard<std::mutex> lk(mtx_timing_);
        snapshot = slam_total_ms_;
    }

    std::ofstream ofs(out_path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        RCLCPP_ERROR(get_logger(), "Failed to open csv output: %s", out_path.c_str());
        return;
    }

    // single column csv
    ofs << "total_ms\n";
    for (double v : snapshot) {
        ofs << std::fixed << std::setprecision(6) << v << "\n";
    }
    ofs.close();

    RCLCPP_INFO(get_logger(), "Saved slam total(ms) csv: %s (rows=%zu)", out_path.c_str(), snapshot.size());
}

void LocalMap::dumpGlobalPathCsvQuat_(const std::string& out_path) {
    nav_msgs::msg::Path snapshot;
    {
        std::lock_guard<std::mutex> lk(mtx_path_);
        snapshot = globalPath;
    }

    std::ofstream ofs(out_path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        RCLCPP_ERROR(get_logger(), "Failed to open globalPath(quat) csv output: %s", out_path.c_str());
        return;
    }

    ofs << "timestamp,x,y,z,qx,qy,qz,qw\n";
    ofs << std::fixed << std::setprecision(9);

    for (const auto& ps : snapshot.poses) {
        const double t = rclcpp::Time(ps.header.stamp).seconds();
        ofs << t << ","
            << ps.pose.position.x << ","
            << ps.pose.position.y << ","
            << ps.pose.position.z << ","
            << ps.pose.orientation.x << ","
            << ps.pose.orientation.y << ","
            << ps.pose.orientation.z << ","
            << ps.pose.orientation.w << "\n";
    }
    ofs.close();

    RCLCPP_INFO(get_logger(), "Saved globalPath(quat) csv: %s (rows=%zu)", out_path.c_str(), snapshot.poses.size());
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
  
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);

    auto node = std::make_shared<LocalMap>(options);
    // node->removeDynamicObjTest(); // offline test

    rclcpp::spin(node);

    // Save once right before exit (Ctrl+C).
    {
        std::time_t now = std::time(nullptr);
        std::tm tm_local{};
        localtime_r(&now, &tm_local);
        std::ostringstream ts;
        ts << std::put_time(&tm_local, "%Y%m%d_%H%M%S");
        const std::string suffix = ts.str();

        node->dumpSlamLoopTotalCsv_(std::string("/home/weizh/data/slam_total_") + suffix + ".csv");
        node->dumpGlobalPathCsvQuat_(std::string("/home/weizh/data/global_path_") + suffix + ".csv");
    }

    rclcpp::shutdown();
    return 0;
}
