#include "livo_node.h"

LivoNode::LivoNode(const rclcpp::NodeOptions & options) : ParamServer("livo", options) {
    // 1. 初始化订阅者 (使用 SensorDataQoS 减少延迟)
    auto qos = rclcpp::SensorDataQoS();
    
    // 使用 ParamServer 中的 topic 参数
    sub_imu = create_subscription<sensor_msgs::msg::Imu>(
        imuTopic, qos, std::bind(&LivoNode::imuHandler, this, std::placeholders::_1));
    sub_lidar = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointCloudTopic, qos, std::bind(&LivoNode::lidarHandler, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Subscribed to IMU topic: %s", imuTopic.c_str());
    RCLCPP_INFO(get_logger(), "Subscribed to Lidar topic: %s", pointCloudTopic.c_str());

    laser_cloud_in.reset(new pcl::PointCloud<PointType>());
    local_map.reset(new pcl::PointCloud<PointType>());
    last_laser_cloud_in.reset(new pcl::PointCloud<PointType>());

    // 初始化发布者
    // IMU: 高频、小包、绝不能丢
    // auto qos_imu = rclcpp::QoS(rclcpp::KeepLast(200)).best_effort().durability_volatile();
    // // Lidar: 低频、大包、实时优先
    // auto qos_lidar = rclcpp::QoS(rclcpp::KeepLast(2)).best_effort().durability_volatile();
    auto qos_reliable = rclcpp::QoS(10).reliable();
    pub_odom = create_publisher<nav_msgs::msg::Odometry>("/livo/odometry", qos_reliable);
    pub_cloud_registered = create_publisher<sensor_msgs::msg::PointCloud2>("/livo/cloud_registered", qos_reliable);
    pub_local_map = create_publisher<sensor_msgs::msg::PointCloud2>("/livo/local_map", qos_reliable);

    // 初始化成员变量
    imu_ptr_cur = 0;
    scan_beg_time = 0;
    scan_end_time = 0;

    // 2. 启动独立的算法工作线程
    slam_thread = std::thread(&LivoNode::slamProcessLoop, this);
}

LivoNode::~LivoNode() {
    if (slam_thread.joinable()) {
        slam_thread.join();
    }
}

// --- 回调函数：只负责存数据，不计算 ---
void LivoNode::imuHandler(const sensor_msgs::msg::Imu::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer);
    imu_buffer.push_back(msg);
    // IMU 通常不触发唤醒，由点云触发
}

void LivoNode::lidarHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 1. 在回调中直接转换格式 (并行化优化)
    pcl::PointCloud<HesaiPointXYZIRT> tmp_hesai_cloud;
    pcl::fromROSMsg(*msg, tmp_hesai_cloud);

    // xt16 point time is relative time(unit: ns), msg header timestamp is message sending time (almost the time of last point)
    LidarData data;
    data.lidar_frame_end_time = rclcpp::Time(msg->header.stamp).seconds();
    data.lidar_frame_beg_time = data.lidar_frame_end_time - 0.1;  // 10Hz, each duration is 0.1s, so start time should be minus 0.1
    data.cloud.reset(new pcl::PointCloud<PointType>());
    data.cloud->reserve(tmp_hesai_cloud.size());
    
    for (const auto& p : tmp_hesai_cloud.points) {
        PointType dst;
        dst.x = p.x;
        dst.y = p.y;
        dst.z = p.z;
        dst.intensity = p.intensity;
        dst.normal_x = p.ring;   // 将 ring 存入 normal_x
        dst.normal_y = p.time * 1e-9;   // 将相对时间 time 存入 normal_y, ns -> s

        // 转换到 IMU 坐标系
        dst = lidarToImu(dst);

        data.cloud->push_back(dst);
    }

    // 2. 存入缓冲区
    std::lock_guard<std::mutex> lock(mtx_buffer);
    lidar_buffer.push_back(data);
    cv_data.notify_one(); // 唤醒算法线程
}

// --- 核心算法线程：最可控执行流 ---
void LivoNode::slamProcessLoop() {
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

        // 2. 顺序执行算法 (逻辑清晰，易于 Debug)
        laser_cloud_in = current_lidar_data.cloud;

        processIMU(current_imus);       // IMU 预积分
        pointCloudPreprocessing();      // 点云处理
        performOdometer();              // 配准（前端里程计）
        updateLocalMap();               // 更新局部图
      
        // 3. 发布结果 (也可以异步发布)
        publishResult();
    }
}

void LivoNode::processIMU(const std::vector<sensor_msgs::msg::Imu::SharedPtr>& imus) {
    for (const auto& imu_msg : imus) {
        // 直接使用原始 IMU 数据，因为所有计算都基于 IMU 坐标系
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

void LivoNode::findRotation(double relTime, float *rotXCur, float *rotYCur, float *rotZCur) {
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

void LivoNode::findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur) {
    *posXCur = 0; *posYCur = 0; *posZCur = 0;
    // 简化处理，不考虑位移去畸变，或者假设匀速
}

void LivoNode::deskewPoint(PointType *point, double relTime) {
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

PointType LivoNode::lidarToImu(const PointType& p) {
    PointType dst = p;
    Eigen::Vector3d pt_lidar(p.x, p.y, p.z);
    Eigen::Vector3d pt_imu = extRot * pt_lidar + extTrans;
    dst.x = pt_imu.x();
    dst.y = pt_imu.y();
    dst.z = pt_imu.z();
    return dst;
}

void LivoNode::pointCloudPreprocessing() {
    // 此时 laser_cloud_in 已经在 slamProcessLoop 中被赋值为预转换好的点云

    // 去畸变
    if (deskewByImu) {
        for (int i = 0; i < (int)laser_cloud_in->size(); ++i) {
            PointType &p = laser_cloud_in->points[i];
            // 假设 PointType 的 normal_y 存储了相对时间 (relTime)
            double relTime = p.normal_y; 
            deskewPoint(&p, relTime);
        }
    }

    // 先不要降采样
    // pcl::VoxelGrid<PointType> dsFilter;
    // dsFilter.setInputCloud(laser_cloud_in);
    // dsFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);
    // dsFilter.filter(*laser_cloud_in);
}

void LivoNode::performOdometer() {
    if (is_first_frame) {
        // 第一帧：将其变换到世界坐标系并存入 last_laser_cloud_in 作为下一帧的 target
        last_laser_cloud_in->clear();
        pcl::transformPointCloud(*laser_cloud_in, *last_laser_cloud_in, current_pose);
        is_first_frame = false;
        return;
    }

    if (laser_cloud_in->empty() || last_laser_cloud_in->empty()) return;

    // 策略：将当前帧（Source）与上一帧配准后的点云（Target）进行配准
    // 局部实例化 VGICP 对象，确保每次配准都是干净的状态，避免 out_of_range 错误
    small_gicp::RegistrationPCL<PointType, PointType> vgicp;
    vgicp.setRegistrationType("VGICP");
    vgicp.setVoxelResolution(1.0);
    vgicp.setNumThreads(4);
    vgicp.setMaxCorrespondenceDistance(1.0);
    vgicp.setNumNeighborsForCovariance(40);
    vgicp.setMaximumIterations(100);

    vgicp.setInputSource(laser_cloud_in);
    vgicp.setInputTarget(last_laser_cloud_in);

    // 执行配准
    pcl::PointCloud<PointType>::Ptr aligned_cloud(new pcl::PointCloud<PointType>());
    
    try {
        // 使用当前位姿作为初始猜测进行配准
        vgicp.align(*aligned_cloud, current_pose);

        if (vgicp.hasConverged()) {
            // 更新当前位姿 (直接获得在世界坐标系下的位姿)
            current_pose = vgicp.getFinalTransformation();
        } else {
            RCLCPP_WARN(get_logger(), "VGICP did not converge!");
        }
    } catch (const std::out_of_range& e) {
        RCLCPP_ERROR(get_logger(), "VGICP align caught out_of_range: %s. Source size: %zu, Target size: %zu", 
                     e.what(), laser_cloud_in->size(), last_laser_cloud_in->size());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "VGICP align caught exception: %s", e.what());
    }

    // 将当前帧配准后的点云存入 last_laser_cloud_in，作为下一帧的 target
    last_laser_cloud_in->clear();
    pcl::transformPointCloud(*laser_cloud_in, *last_laser_cloud_in, current_pose);
}

void LivoNode::updateLocalMap() {
    if (laser_cloud_in->empty()) return;

    // 将当前帧转换到世界坐标系
    pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud_in, *cloud_world, current_pose);

    // 更新局部地图 (简单叠加，实际应用中可能需要降采样或滑动窗口)
    *local_map += *cloud_world;

    // 简单的局部地图规模控制：如果点数过多，进行体素滤波
    if (local_map->size() > 100000) {
        pcl::VoxelGrid<PointType> dsFilter;
        dsFilter.setInputCloud(local_map);
        dsFilter.setLeafSize(0.5, 0.5, 0.5);
        pcl::PointCloud<PointType>::Ptr filtered_map(new pcl::PointCloud<PointType>());
        dsFilter.filter(*filtered_map);
        local_map = filtered_map;
    }
}

void LivoNode::publishResult() {
    // 1. 发布里程计
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = rclcpp::Time(static_cast<uint64_t>(scan_end_time * 1e9));
    odom.header.frame_id = odometryFrame;
    odom.child_frame_id = bodyFrame;

    Eigen::Matrix3f rot = current_pose.block<3, 3>(0, 0);
    Eigen::Quaternionf q(rot);
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.position.x = current_pose(0, 3);
    odom.pose.pose.position.y = current_pose(1, 3);
    odom.pose.pose.position.z = current_pose(2, 3);
    pub_odom->publish(odom);

    // 2. 发布当前帧点云 (已转换到世界坐标系)
    pcl::PointCloud<PointType>::Ptr cloud_world(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud_in, *cloud_world, current_pose);
    publishCloud(pub_cloud_registered, cloud_world, odom.header.stamp, odometryFrame);

    // 3. 发布局部地图
    if (pub_local_map->get_subscription_count() != 0) {
        publishCloud(pub_local_map, local_map, odom.header.stamp, odometryFrame);
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
  
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true); // 开启进程内零拷贝

    auto node = std::make_shared<LivoNode>(options);
  
    // 使用单线程执行器处理 ROS 消息收发即可
    rclcpp::spin(node);
  
    rclcpp::shutdown();
    return 0;
}
