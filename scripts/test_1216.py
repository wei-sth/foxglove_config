#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField # Added for PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2 # Added for PointCloud2 utility
import numpy as np
import math
import pcd # User's custom PCD module
import pandas as pd # Required for DataFrame

class ObstacleBBoxPublisher(Node):
    def __init__(self):
        super().__init__('obstacle_bbox_publisher')
        self.marker_publisher = self.create_publisher(MarkerArray, '/obstacle_bounding_boxes', 10)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, '/raw_point_cloud', 10) # New publisher for raw point cloud
        self.timer = self.create_timer(1.0, self.timer_callback) # Combined timer callback

        # Path for the raw input PCD file
        self.raw_pcd_file_path = "/home/weizh/data/bag_11261/bag_11261_0_logs/unitree_slam_lidar_points/1764126218_844803835.pcd"
        self.get_logger().info(f'Loading raw PCD file: {self.raw_pcd_file_path}...')
        self.raw_pcd_data_df = self.load_pcd_file(self.raw_pcd_file_path, 
                                                  columns=['x','y','z','intensity','ring','time']) # Load with new columns

        if self.raw_pcd_data_df is None:
            self.get_logger().error("Failed to load raw PCD file. Exiting.")
            rclpy.shutdown()
            return

        self.get_logger().info(f'Loaded {len(self.raw_pcd_data_df)} points from {self.raw_pcd_file_path}')

        # Path for the obstacles PCD file (output from C++ code)
        self.obstacles_pcd_file_path = "/home/weizh/data/obstacles.pcd"
        self.get_logger().info(f'Loading obstacles PCD file: {self.obstacles_pcd_file_path}...')
        self.obstacles_pcd_data_df = self.load_pcd_file(self.obstacles_pcd_file_path,
                                                        columns=['x','y','z','intensity']) # Assuming XYZI for obstacles.pcd

        if self.obstacles_pcd_data_df is None:
            self.get_logger().error("Failed to load obstacles.pcd. Bounding boxes will not be published. Exiting.")
            rclpy.shutdown()
            return

        self.get_logger().info(f'Loaded {len(self.obstacles_pcd_data_df)} points from {self.obstacles_pcd_file_path}')
        self.clusters_bboxes = self.process_pcd_for_bboxes(self.obstacles_pcd_data_df)
        self.get_logger().info(f'Detected {len(self.clusters_bboxes)} clusters.')

    def load_pcd_file(self, path, columns): # Added columns parameter
        try:
            # Use the user's custom pcd module to read the file
            points_raw = pcd.read_pcd_file(path)
            
            # Convert to pandas DataFrame with specified columns
            points_df = pd.DataFrame(points_raw, columns=columns)
            
            if points_df.empty:
                self.get_logger().error(f"PCD file {path} is empty after processing.")
                return None
            
            self.get_logger().info(f"Custom pcd module loaded and data extracted into DataFrame for {path}.")
            return points_df
        except Exception as e:
            self.get_logger().error(f"Error loading PCD file {path} using custom pcd module: {e}")
            return None

    def process_pcd_for_bboxes(self, pcd_data_df):
        # Extract points (x, y, z) and intensities from the DataFrame
        points = pcd_data_df[['x', 'y', 'z']].values
        intensities = pcd_data_df['intensity'].values
        
        if len(intensities) == 0:
            self.get_logger().error("Intensity information is empty in PCD. Bounding boxes will not be generated correctly.")
            return []
        
        # Group points by intensity
        unique_intensities = np.unique(intensities)
        
        clusters_data = {}
        for intensity_val in unique_intensities:
            cluster_indices = np.where(intensities == intensity_val)[0]
            if len(cluster_indices) > 0:
                clusters_data[intensity_val] = points[cluster_indices]

        bboxes = []
        for intensity_val, cluster_points in clusters_data.items():
            if len(cluster_points) > 0:
                min_bound = np.min(cluster_points, axis=0)
                max_bound = np.max(cluster_points, axis=0)

                bbox = {
                    'min_point': min_bound,
                    'max_point': max_bound,
                    'center': (min_bound + max_bound) / 2,
                    'dimensions': max_bound - min_bound,
                    'intensity': intensity_val # Store intensity for potential use in color
                }
                bboxes.append(bbox)
        return bboxes

    def publish_point_cloud(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link" # IMPORTANT: Set this to your LiDAR's frame_id

        # Prepare fields for PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring', offset=16, datatype=PointField.UINT16, count=1),
            PointField(name='time', offset=18, datatype=PointField.FLOAT32, count=1),
        ]

        # Convert DataFrame to a list of tuples for pc2.create_cloud
        # Ensure the order matches the fields
        points_data = self.raw_pcd_data_df[['x', 'y', 'z', 'intensity', 'ring', 'time']].values.tolist()
        
        cloud_msg = pc2.create_cloud(header, fields, points_data)
        self.pointcloud_publisher.publish(cloud_msg)
        self.get_logger().info(f'Published {len(points_data)} raw point cloud points.')

    def publish_bboxes(self):
        marker_array = MarkerArray()
        
        # Clear previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "base_link" # Or your desired frame_id
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i, bbox_data in enumerate(self.clusters_bboxes):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "base_link" # IMPORTANT: Set this to your LiDAR's frame_id
            marker.ns = "obstacle_bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position
            marker.pose.position.x = float(bbox_data['center'][0])
            marker.pose.position.y = float(bbox_data['center'][1])
            marker.pose.position.z = float(bbox_data['center'][2])
            
            # Orientation (identity for axis-aligned bounding box)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # Scale
            marker.scale.x = float(bbox_data['dimensions'][0])
            marker.scale.y = float(bbox_data['dimensions'][1])
            marker.scale.z = float(bbox_data['dimensions'][2])

            # Color (random color for each bbox, or based on intensity)
            # Color (simple color cycling based on marker ID)
            marker.color.a = 0.5 # Alpha (transparency)
            marker.color.r = (i * 0.3 + 0.1) % 1.0
            marker.color.g = (i * 0.7 + 0.2) % 1.0
            marker.color.b = (i * 0.9 + 0.3) % 1.0
            
            marker_array.markers.append(marker)
        
        self.marker_publisher.publish(marker_array)
        self.get_logger().info(f'Published {len(self.clusters_bboxes)} bounding box markers.')

    def timer_callback(self):
        self.publish_point_cloud()
        self.publish_bboxes()

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleBBoxPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
