#include "localmap_node.h"
#include <chrono>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
// downsample by 0.1m does not improve
// downsample by 0.2m, setNumNeighborsForCovariance(20) from 40 improves
// without setting other params (num of threads ...), voxelgrid_sampling_omp seems slower than pcl::VoxelGrid
// then I change voxelgrid_sampling_omp to voxelgrid_sampling, and set resolution of both source and target to 0.2m (previously source 0.2 and target 0.1), faster
// todo: check https://github.com/koide3/small_gicp/blob/master/src/benchmark/odometry_benchmark_small_gicp_omp.cpp

// todo: 应该针对local map中的所有obstacle划定一个ROI，对这个ROI中的做ray casting，看看是否可行，找到落到这个ROI中的最后一帧lidar的障碍物
// 在查询 range_image[u][v] 时，不要只查一个像素，可以查周围 3x3 的邻域，取最小值或者进行某种插值，防止因为雷达光束打偏了而误删障碍物。
// 是否能否跟踪cluster，然后当cluster变得越来越大时，是不是能够判断到底是不是看到的越来越多，还是拖影呢？

static constexpr bool LOCAL_DEBUG = true;

LocalMap::LocalMap(const rclcpp::NodeOptions & options) : ParamServer("localmap", options) {
    sub_imu = create_subscription<sensor_msgs::msg::Imu>(imuTopic,
        rclcpp::QoS(rclcpp::KeepLast(10000)).reliable(),
        std::bind(&LocalMap::imuHandler, this, std::placeholders::_1));
    sub_lidar = create_subscription<sensor_msgs::msg::PointCloud2>(pointCloudTopic,
        rclcpp::QoS(rclcpp::KeepLast(10000)).reliable(),
        std::bind(&LocalMap::lidarHandler, this, std::placeholders::_1));

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
    pub_cloud_registered = create_publisher<sensor_msgs::msg::PointCloud2>("/localmap/cloud_registered", qos_reliable);
    pub_local_map = create_publisher<sensor_msgs::msg::PointCloud2>("/localmap/local_map", qos_reliable);
    pub_obstacle_map = create_publisher<sensor_msgs::msg::PointCloud2>("/localmap/obstacle_voxel_grid", qos_reliable);

    imu_ptr_cur = 0;
    scan_beg_time = 0;
    scan_end_time = 0;

    vis_type_ = VisResultType::JSON_AND_VOXELE;
    if constexpr (LOCAL_DEBUG) { vis_type_ = VisResultType::JSON_AND_VOXELL; }
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
            RCLCPP_ERROR(get_logger(), "imu loop back, offset: %lf \n", last_timestamp_imu - timestamp); // if happens a lot, move out of lock
            return;
        }
        if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
        {
            RCLCPP_WARN(get_logger(), "imu time stamp jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
            return;
        }

        last_timestamp_imu = timestamp;
        imu_buffer.push_back(msg);
        // imu do not notify, let lidarHandler notify
    }
}

// check imu_prop_callback function in fast-livo2, if speed = 20km/h, 100 ms means 0.55m, for obstacle detection, we need real time pose
// we should based on the pose optimized by last lidar align, calculate relative movement from imu

void LocalMap::lidarHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    lidarMsgTimestamp = msg->header.stamp;
    lidarMsgTimeValue = rclcpp::Time(msg->header.stamp).seconds();
    pcl::PointCloud<RSPointDefault> tmp_rs_cloud;
    pcl::fromROSMsg(*msg, tmp_rs_cloud);

    // robosense point time is relative time(unit: ns), msg header timestamp is scan start time
    LidarData data;
    data.lidar_frame_beg_time = lidarMsgTimeValue;
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
        scan_beg_time = current_lidar_data.lidar_frame_beg_time;
        scan_end_time = current_lidar_data.lidar_frame_end_time;

        while (!imu_buffer.empty() && rclcpp::Time(imu_buffer.front()->header.stamp).seconds() < scan_end_time) {
            current_imus.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }

        laser_cloud_in = current_lidar_data.cloud;
        lock.unlock();  //unlock what?

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
        // updateLocalMap();
        // v1: current scan align to local map, only key frame is added to local map
        performOdometer_v1();
        auto t4 = std::chrono::steady_clock::now();

        // remove dynamic by range image
        auto cloud_filtered_normal = filterByRangeEgo(current_lidar_data.cloud_raw, max_range_);
        buildRangeImage(cloud_filtered_normal);

        updateObstacleVoxelMap(obstacle_clusters, current_pose, scan_end_time);
      
        // 3. 发布结果 (也可以异步发布)
        publishResult();
        auto t5 = std::chrono::steady_clock::now();

        double d1 = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double d2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double d3 = std::chrono::duration<double, std::milli>(t4 - t3).count();
        double d4 = std::chrono::duration<double, std::milli>(t5 - t4).count();
        double total = d1 + d2 + d3 + d4;

        if (total > 0) {
            RCLCPP_INFO(get_logger(), "Time stats: IMU: %.2f%%, Preprocess: %.2f%%, Odom: %.2f%%, Publish: %.2f%%, Total: %.2f ms",
                        d1/total*100.0, d2/total*100.0, d3/total*100.0, d4/total*100.0, total);
        }
    }
}

void LocalMap::updateObstacleVoxelMap(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& obstacle_clusters,
    const Eigen::Affine3f& pose, double timestamp) {
    if (obstacle_clusters.empty()) {
        return;
    }
    
    float resolution = 0.1f;
    
    // 1. 体素化并转换到世界坐标系
    std::map<std::tuple<int, int, int>, Voxel> new_voxels;
    
    #pragma omp parallel for
    for (size_t i = 0; i < obstacle_clusters.size(); ++i) {
        const auto& cluster = obstacle_clusters[i];
        std::map<std::tuple<int, int, int>, Voxel> local_voxels;
        
        for (const auto& pt : cluster->points) {
            // 转换到世界坐标系
            Eigen::Vector3f pt_local(pt.x, pt.y, pt.z);
            Eigen::Vector3f pt_world = pose * pt_local;
            
            // 计算体素坐标
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
    double timeout = 5.0;
    
    auto it = obstacle_voxel_map.begin();
    while (it != obstacle_voxel_map.end()) {
        const Voxel& voxel = it->second;
        
        // 计算距离当前位置的距离
        Eigen::Vector3f voxel_pos(voxel.x, voxel.y, voxel.z);
        float dist = (voxel_pos - pose.translation()).norm();
        
        bool should_remove = false;
        
        // 条件1：超过最大距离
        if (dist > max_distance) {
            should_remove = true;
        }
        
        // 条件2：超时
        if (timestamp - voxel.last_seen_time > timeout) {
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

    // 清理过旧的 IMU，保留一个在 scan_beg_time 之前的以用于插值
    while (imu_que_opt.size() > 1 && rclcpp::Time(imu_que_opt[1].header.stamp).seconds() < scan_beg_time) {
        imu_que_opt.pop_front();
    }

    // 初始化积分起点为 scan_beg_time，旋转为 0
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

    // 关键帧判断：位移超过0.3m或者旋转超过10度
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

    PointTypePose thisPose6D;
    Eigen::Matrix3f rot = current_pose.linear();
    Eigen::Quaternionf q(rot);
    thisPose6D.x = current_pose.translation().x();
    thisPose6D.y = current_pose.translation().y();
    thisPose6D.z = current_pose.translation().z();
    thisPose6D.intensity = 0;  // index
    Eigen::Matrix3f R = current_pose.rotation();
    float roll  = atan2(R(2,1), R(2,2));
    float pitch = atan2(-R(2,0), sqrt(R(2,1)*R(2,1) + R(2,2)*R(2,2)));
    float yaw   = atan2(R(1,0), R(0,0));
    thisPose6D.roll = roll;
    thisPose6D.pitch = pitch;
    thisPose6D.yaw = yaw;
    thisPose6D.time = lidarMsgTimeValue;
    // add pose to path for visualization
    updatePath(thisPose6D);
}

void LocalMap::updateLocalMap() {
    if (laser_cloud_in_ds->empty()) return;

    // 将当前帧转换到世界坐标系
    pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud_in_ds, *cloud_world, current_pose);

    // 更新局部地图 (简单叠加，实际应用中可能需要降采样或滑动窗口)
    *local_map += *cloud_world;

    // 简单的局部地图规模控制：如果点数过多，进行体素滤波
    if (local_map->size() > 100000) {
        local_map = small_gicp::voxelgrid_sampling(*local_map, 0.5);
    }
}

void LocalMap::updatePath(const PointTypePose& pose_in) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    rclcpp::Time t(static_cast<uint32_t>(pose_in.time * 1e9));
    pose_stamped.header.stamp = t;
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

    globalPath.poses.push_back(pose_stamped);
}

void LocalMap::publishResult() {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = lidarMsgTimestamp;
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
        globalPath.header.stamp = lidarMsgTimestamp;
        globalPath.header.frame_id = odometryFrame;
        pub_path->publish(globalPath);
    }

    pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud_in, *cloud_world, current_pose);
    publishCloud(pub_cloud_registered, cloud_world, lidarMsgTimestamp, odometryFrame);

    if (pub_local_map->get_subscription_count() != 0) {
        publishCloud(pub_local_map, local_map, lidarMsgTimestamp, odometryFrame);
    }

    // publish obstacle_voxel_map

    // if (pub_obstacle_map->get_subscription_count() == 0) {
    //     return;
    // }
    
    if (!obstacle_voxel_map.empty()) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        voxel_cloud->reserve(obstacle_voxel_map.size());
        
        for (const auto& kv : obstacle_voxel_map) {
            const Voxel& voxel = kv.second;

            pcl::PointXYZI pt;
            pt.x = voxel.x;
            pt.y = voxel.y;
            pt.z = voxel.z;
            pt.intensity = std::min(255.0f, voxel.observation_count * 50.0f);
            
            voxel_cloud->points.push_back(pt);
        }
        
        // voxel_cloud not empty
        voxel_cloud->width = voxel_cloud->size();
        voxel_cloud->height = 1;
        voxel_cloud->is_dense = true;
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*voxel_cloud, cloud_msg);
        cloud_msg.header.stamp = lidarMsgTimestamp;
        cloud_msg.header.frame_id = odometryFrame;
        pub_obstacle_map->publish(cloud_msg);
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
  
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);

    auto node = std::make_shared<LocalMap>(options);
    rclcpp::spin(node);
  
    rclcpp::shutdown();
    return 0;
}
