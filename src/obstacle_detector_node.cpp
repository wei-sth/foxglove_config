
class ObstacleDetectorNode : public rclcpp::Node {
public:
    ObstacleDetectorNode() : Node("obstacle_detector_node") {
        // Declare parameters
        this->declare_parameter<int>("num_rings", 16);
        this->declare_parameter<int>("num_sectors", 2000);
        this->declare_parameter<float>("max_distance", 10.0f);
        this->declare_parameter<float>("min_cluster_z_difference", 0.2f);
        this->declare_parameter<std::string>("input_topic", "/lidar_points");
        this->declare_parameter<std::string>("output_topic", "/obstacle_bboxes");

        // Get parameters
        int num_rings = this->get_parameter("num_rings").as_int();
        int num_sectors = this->get_parameter("num_sectors").as_int();
        float max_distance = this->get_parameter("max_distance").as_float();
        float min_cluster_z_difference = this->get_parameter("min_cluster_z_difference").as_float();
        std::string input_topic = this->get_parameter("input_topic").as_string();
        std::string output_topic = this->get_parameter("output_topic").as_string();

        detector_ = std::make_unique<RangeImageObstacleDetector>(
            num_rings, num_sectors, max_distance, min_cluster_z_difference);

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic, 10, std::bind(&ObstacleDetectorNode::pointCloudCallback, this, std::placeholders::_1));
        
        publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>(output_topic, 10);

        RCLCPP_INFO(this->get_logger(), "ObstacleDetectorNode initialized.");
        RCLCPP_INFO(this->get_logger(), "Subscribing to topic: %s", input_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to topic: %s", output_topic.c_str());
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message.");

        pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw(new pcl::PointCloud<PointXYZIRT>);
        pcl::fromROSMsg(*msg, *cloud_raw);

        pcl::PointCloud<pcl::PointXYZI>::Ptr obstacles = detector_->detectObstacles(cloud_raw);
        std::vector<BoundingBox> bboxes = getObstacleBoundingBoxes(obstacles);

        geometry_msgs::msg::PoseArray bbox_array_msg;
        bbox_array_msg.header = msg->header; // Synchronize timestamp

        for (const auto& bbox : bboxes) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = bbox.center.x;
            pose.position.y = bbox.center.y;
            pose.position.z = bbox.center.z;
            // Orientation can be set if needed, for now, default to no rotation
            pose.orientation.w = 1.0; 
            bbox_array_msg.poses.push_back(pose);
        }
        publisher_->publish(bbox_array_msg);
        RCLCPP_INFO(this->get_logger(), "Published %zu bounding boxes.", bboxes.size());
    }

    std::unique_ptr<RangeImageObstacleDetector> detector_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr publisher_;
};


int main(int argc, char * argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--pcd") {
        // PCD file processing mode
        int num_rings = 16;
        int num_sectors = 2000;
        float max_distance = 10.0f;
        float min_cluster_z_difference = 0.2f;

        RangeImageObstacleDetector detector(num_rings, num_sectors, max_distance, min_cluster_z_difference);
        
        pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw(
            new pcl::PointCloud<PointXYZIRT>);
        std::string pcd_file_path = "/home/weizh/data/bag_11261/bag_11261_0_logs/unitree_slam_lidar_points/1764126218_844803835.pcd";
        if (argc > 2) {
            pcd_file_path = argv[2];
        }

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
        
    } else {
        // ROS node mode
        rclcpp::init(argc, argv);
        rclcpp::spin(std::make_shared<ObstacleDetectorNode>());
        rclcpp::shutdown();
    }
    
    return 0;
}
