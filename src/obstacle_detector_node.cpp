#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "obstacle_detector.h"
#include <cmath> // For std::sin and std::cos
#include <mqtt/async_client.h>

// robosense airy frame_id: rslidar
// bbox is not suitable for indoor, I tried to use nav2_msgs/msg/VoxelGrid, but rviz cannot show
// so use pointcloud, set style as boxes, size = 0.1
// MQTT broker kicks the older client if a new one connects with the same ID. 
// Using a unique ID for each device to avoid connection loops. I use "obstacle_client_pc" (need to use yaml in the future)

class ObstacleDetectorNode : public rclcpp::Node, public virtual mqtt::callback {
public:
    ObstacleDetectorNode() 
    : Node("obstacle_detector_node"),
      mqtt_client_("tcp://124.221.132.177:1883", "obstacle_client_jetson") {
        mqtt_client_.set_callback(*this);
        mqtt_conn_opts_.set_user_name("zbtest");
        mqtt_conn_opts_.set_password("zbtest");
        mqtt_conn_opts_.set_keep_alive_interval(5);
        mqtt_conn_opts_.set_connect_timeout(5);
        mqtt_conn_opts_.set_clean_session(true);
        mqtt_conn_opts_.set_automatic_reconnect(true);

        try {
            mqtt_client_.connect(mqtt_conn_opts_)->wait();
            RCLCPP_INFO(this->get_logger(), "Connected to MQTT Broker: 124.221.132.177");
        } catch (const mqtt::exception& exc) {
            RCLCPP_ERROR(this->get_logger(), "MQTT Connection failed: %s", exc.what());
        }

        // Declare parameters
        this->declare_parameter<int>("num_rings", 96);
        this->declare_parameter<int>("num_sectors", 900);
        this->declare_parameter<int>("vis_type", static_cast<int>(VisResultType::BBOX_2D_AND_VOXEL));
        this->declare_parameter<float>("max_distance", 10.0f);
        this->declare_parameter<float>("min_cluster_z_difference", 0.2f);
        this->declare_parameter<std::string>("input_topic", "/rslidar_points"); // /rslidar_points | /unitree/slam_lidar/points
        this->declare_parameter<std::string>("output_topic", "/obstacle_bbox");
        this->declare_parameter<std::string>("voxel_grid_topic", "/obstacle_voxel_grid");

        // Get parameters
        int num_rings = this->get_parameter("num_rings").as_int();
        int num_sectors = this->get_parameter("num_sectors").as_int();
        vis_type_ = static_cast<VisResultType>(this->get_parameter("vis_type").as_int());
        float max_distance = this->get_parameter("max_distance").get_value<float>();
        float min_cluster_z_difference = this->get_parameter("min_cluster_z_difference").get_value<float>();
        std::string input_topic = this->get_parameter("input_topic").as_string();
        std::string output_topic = this->get_parameter("output_topic").as_string();
        std::string voxel_grid_topic = this->get_parameter("voxel_grid_topic").as_string();

        detector_ = std::make_unique<RangeImageObstacleDetector>(num_rings, num_sectors, max_distance, min_cluster_z_difference, vis_type_);

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(input_topic, rclcpp::QoS(10).best_effort(), std::bind(&ObstacleDetectorNode::pointCloudCallback, this, std::placeholders::_1));
        
        publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(output_topic, 10);
        voxel_grid_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(voxel_grid_topic, 10);

        RCLCPP_INFO(this->get_logger(), "ObstacleDetectorNode initialized.");
        RCLCPP_INFO(this->get_logger(), "Subscribing to topic: %s", input_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to topic: %s", output_topic.c_str());
    }

    void connection_lost(const std::string& cause) override {
        RCLCPP_ERROR(this->get_logger(), "MQTT connection lost: %s", cause.c_str());
    }
    void connected(const std::string& cause) override {
        RCLCPP_INFO(this->get_logger(), "MQTT connected: %s", cause.c_str());
    }
    // not used but required by interface
    void message_arrived(mqtt::const_message_ptr /*msg*/) override {}
    void delivery_complete(mqtt::delivery_token_ptr /*tok*/) override {}

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        timestamp = rclcpp::Time(msg->header.stamp).seconds();
        RCLCPP_INFO(this->get_logger(), "Received frame %.9f", timestamp);

        pcl::PointCloud<RSPointDefault>::Ptr cloud_raw(new pcl::PointCloud<RSPointDefault>);
        pcl::fromROSMsg(*msg, *cloud_raw);

        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> obstacle_clusters = detector_->detectObstacles(cloud_raw);
        
        if (vis_type_ == VisResultType::BBOX_LIDAR_XY || vis_type_ == VisResultType::BBOX_GROUND) {
            std::vector<RotatedBoundingBox> rotated_bboxes = detector_->getVisBBoxes();

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
                marker.pose.position.z = rbbox.center.z;

                marker.pose.orientation.x = rbbox.orientation.x();
                marker.pose.orientation.y = rbbox.orientation.y();
                marker.pose.orientation.z = rbbox.orientation.z();
                marker.pose.orientation.w = rbbox.orientation.w();

                marker.scale.x = rbbox.size_x;
                marker.scale.y = rbbox.size_y;
                marker.scale.z = rbbox.size_z;

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

                // if (rbbox.size_x > 2.0 || rbbox.size_y > 2.0 || rbbox.size_z > 2.0) {
                //     RCLCPP_WARN(this->get_logger(), "Large rotated bounding box detected! Timestamp: %d_%u, size_x: %.2f, size_y: %.2f, size_z: %.2f", 
                //                 msg->header.stamp.sec, msg->header.stamp.nanosec, rbbox.size_x, rbbox.size_y, rbbox.size_z);
                // }
            }
            publisher_->publish(marker_array_msg);
            // RCLCPP_INFO(this->get_logger(), "Published %zu rotated bounding box markers.", rotated_bboxes.size());
        }

        // Send 2D BBoxes via MQTT
        if (vis_type_ == VisResultType::BBOX_GROUND_2D || vis_type_ == VisResultType::BBOX_2D_AND_VOXEL) {
            auto mqtt_lidar_data = detector_->getMqttLidarData(obstacle_clusters);
            if (!mqtt_lidar_data.empty()) {
                try {
                    nlohmann::json j;
                    j["lidar_data"] = mqtt_lidar_data;
                    std::string payload = j.dump();
                    RCLCPP_INFO(this->get_logger(), "Attempting MQTT Publish...");
                    mqtt_client_.publish("lidar/data", payload, 0, false);  // qos=0: do not wait for confirm, 1: at least once
                    RCLCPP_INFO(this->get_logger(), "MQTT Publish returned.");
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to send MQTT message: %s", e.what());
                }
            }
        }

        // Publish VoxelGrid as PointCloud2 for RViz2 visualization
        if (vis_type_ == VisResultType::BBOX_2D_AND_VOXEL && !obstacle_clusters.empty()) {
            float resolution = 0.1f;
            pcl::PointCloud<pcl::PointXYZI>::Ptr voxel_pc(new pcl::PointCloud<pcl::PointXYZI>);
            
            // Use a set to keep track of occupied voxels and avoid duplicates
            std::set<std::tuple<int, int, int>> occupied_voxels;

            for (const auto& cluster : obstacle_clusters) {
                for (const auto& pt : cluster->points) {
                    int ix = std::floor(pt.x / resolution);
                    int iy = std::floor(pt.y / resolution);
                    int iz = std::floor(pt.z / resolution);

                    if (occupied_voxels.find({ix, iy, iz}) == occupied_voxels.end()) {
                        occupied_voxels.insert({ix, iy, iz});
                        pcl::PointXYZI voxel_pt;
                        voxel_pt.x = ix * resolution + resolution / 2.0f;
                        voxel_pt.y = iy * resolution + resolution / 2.0f;
                        voxel_pt.z = iz * resolution + resolution / 2.0f;
                        voxel_pt.intensity = pt.intensity;
                        voxel_pc->points.push_back(voxel_pt);
                    }
                }
            }

            if (!voxel_pc->empty()) {
                auto voxel_pc_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
                pcl::toROSMsg(*voxel_pc, *voxel_pc_msg);
                voxel_pc_msg->header = msg->header;
                voxel_grid_pub_->publish(std::move(voxel_pc_msg));
            }
        }

        RCLCPP_INFO(this->get_logger(), "End process frame %.9f", timestamp);
    }

    mqtt::async_client mqtt_client_;
    mqtt::connect_options mqtt_conn_opts_;
    VisResultType vis_type_;
    std::unique_ptr<RangeImageObstacleDetector> detector_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr voxel_grid_pub_;
    double timestamp;  // for debug
};


int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObstacleDetectorNode>());
    rclcpp::shutdown();
    
    return 0;
}
