#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "/home/weizh/foxglove_ws/src/foxglove_config/include/obstacle_detector.h"
#include <cmath> // For std::sin and std::cos

class ObstacleDetectorNode : public rclcpp::Node {
public:
    ObstacleDetectorNode() : Node("obstacle_detector_node") {
        // Declare parameters
        this->declare_parameter<int>("num_rings", 16);
        this->declare_parameter<int>("num_sectors", 2000);
        this->declare_parameter<float>("max_distance", 10.0f);
        this->declare_parameter<float>("min_cluster_z_difference", 0.2f);
        this->declare_parameter<std::string>("input_topic", "/unitree/slam_lidar/points");
        this->declare_parameter<std::string>("output_topic", "/obstacle_bbox");

        // Get parameters
        int num_rings = this->get_parameter("num_rings").as_int();
        int num_sectors = this->get_parameter("num_sectors").as_int();
        float max_distance = this->get_parameter("max_distance").get_value<float>();
        float min_cluster_z_difference = this->get_parameter("min_cluster_z_difference").get_value<float>();
        std::string input_topic = this->get_parameter("input_topic").as_string();
        std::string output_topic = this->get_parameter("output_topic").as_string();

        detector_ = std::make_unique<RangeImageObstacleDetector>(
            num_rings, num_sectors, max_distance, min_cluster_z_difference);

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic, 10, std::bind(&ObstacleDetectorNode::pointCloudCallback, this, std::placeholders::_1));
        
        publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(output_topic, 10);

        RCLCPP_INFO(this->get_logger(), "ObstacleDetectorNode initialized.");
        RCLCPP_INFO(this->get_logger(), "Subscribing to topic: %s", input_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to topic: %s", output_topic.c_str());
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message.");

        pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw(new pcl::PointCloud<PointXYZIRT>);
        pcl::fromROSMsg(*msg, *cloud_raw);

        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> obstacle_clusters = detector_->detectObstacles(cloud_raw);
        std::vector<RotatedBoundingBox> rotated_bboxes = detector_->getObstacleBoundingBoxesNew(obstacle_clusters);

        visualization_msgs::msg::MarkerArray marker_array_msg;

        // Clear previous markers
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header.frame_id = msg->header.frame_id;
        delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array_msg.markers.push_back(delete_marker);

        for (size_t i = 0; i < rotated_bboxes.size(); ++i) {
            const auto& rbbox = rotated_bboxes[i];
            visualization_msgs::msg::Marker marker;
            marker.header = msg->header;
            marker.ns = "obstacle_bboxes";
            marker.id = i;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position.x = rbbox.center.x;
            marker.pose.position.y = rbbox.center.y;
            marker.pose.position.z = (rbbox.min_z_point.z + rbbox.max_z_point.z) / 2.0f; // Center Z

            // Manually set orientation from yaw angle
            // Assuming rotation only around Z-axis (yaw)
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = std::sin(rbbox.angle / 2.0);
            marker.pose.orientation.w = std::cos(rbbox.angle / 2.0);

            marker.scale.x = rbbox.width;
            marker.scale.y = rbbox.height;
            marker.scale.z = rbbox.max_z_point.z - rbbox.min_z_point.z;

            // Simple color cycling
            marker.color.a = 0.5;
            marker.color.r = (i * 0.3 + 0.1);
            marker.color.g = (i * 0.7 + 0.2);
            marker.color.b = (i * 0.9 + 0.3);
            
            // Normalize colors to be between 0 and 1
            marker.color.r = fmod(marker.color.r, 1.0);
            marker.color.g = fmod(marker.color.g, 1.0);
            marker.color.b = fmod(marker.color.b, 1.0);

            marker_array_msg.markers.push_back(marker);

            if (rbbox.width > 2.0 || rbbox.height > 2.0) {
                RCLCPP_WARN(this->get_logger(), "Large rotated bounding box detected! Timestamp: %d_%u, Width: %.2f, Height: %.2f",
                            msg->header.stamp.sec, msg->header.stamp.nanosec, rbbox.width, rbbox.height);
            }
        }
        publisher_->publish(marker_array_msg);
        // RCLCPP_INFO(this->get_logger(), "Published %zu rotated bounding box markers.", rotated_bboxes.size());
    }

    std::unique_ptr<RangeImageObstacleDetector> detector_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
};


int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObstacleDetectorNode>());
    rclcpp::shutdown();
    
    return 0;
}
