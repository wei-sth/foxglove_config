#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import std_msgs.msg


class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        
        self.fix_pub = self.create_publisher(NavSatFix, '/fix', 10)
        
        # 1Hz
        self.timer = self.create_timer(1.0, self.publish_fix)
        
        self.get_logger().info('NavSatFix publisher start')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, # TRANSIENT_LOCAL
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.global_map_pub = self.create_publisher(PointCloud2, '/global_map', qos_profile)
        self.publish_global_map()
    
    def publish_fix(self):
        fix_msg = NavSatFix()
        fix_msg.header = Header()
        fix_msg.header.stamp = self.get_clock().now().to_msg()
        fix_msg.header.frame_id = 'gps_link'
        
        fix_msg.latitude = 30.53700758
        fix_msg.longitude = 114.3550648
        fix_msg.altitude = 0.0
        
        # need to revise
        fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        
        self.fix_pub.publish(fix_msg)

    def publish_global_map(self):
        pcd_path = "/home/weizh/Downloads/LOAM/GlobalMap.pcd"
        pcd_path = "/home/weizh/data/grass_global_map.pcd"
        self.get_logger().info(f'loading pcd file: {pcd_path} ...')

        pcd = o3d.io.read_point_cloud(pcd_path)
        
        if pcd.is_empty():
            self.get_logger().error("fail loading pcd file")
            return
        
        self.get_logger().info(f'original num points: {len(pcd.points)}')
        pcd = pcd.voxel_down_sample(voxel_size=0.3) 
        self.get_logger().info(f'downsampled num points: {len(pcd.points)}')

        points = np.asarray(pcd.points)

        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "odom"
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.global_map_pub.publish(pc2_msg)
        self.get_logger().info('static global map published(latched)')

def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
