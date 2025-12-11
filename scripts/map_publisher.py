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
        
        # 创建发布者
        self.fix_pub = self.create_publisher(NavSatFix, '/fix', 10)
        
        # 定时发布（1Hz）
        self.timer = self.create_timer(1.0, self.publish_fix)
        
        self.get_logger().info('NavSatFix 发布器已启动')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, # 重点：持久化
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.global_map_pub = self.create_publisher(PointCloud2, '/global_map', qos_profile)
        self.publish_global_map()
    
    def publish_fix(self):
        """发布 NavSatFix 消息"""
        fix_msg = NavSatFix()
        fix_msg.header = Header()
        fix_msg.header.stamp = self.get_clock().now().to_msg()
        fix_msg.header.frame_id = 'gps_link'  # 或者其他合适的frame_id
        
        fix_msg.latitude = 30.53700758
        fix_msg.longitude = 114.3550648
        fix_msg.altitude = 0.0  # 示例高度
        
        # 设置协方差类型为未知，或者根据实际情况设置
        fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        
        self.fix_pub.publish(fix_msg)

    def publish_global_map(self):
        pcd_path = "/home/weizh/Downloads/LOAM/GlobalMap.pcd"
        self.get_logger().info(f'正在加载点云: {pcd_path} ...')

        # 2. 使用 Open3D 读取
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        if pcd.is_empty():
            self.get_logger().error("点云文件读取失败或为空！")
            return

        # 3. (可选) 下采样：防止地图太大导致 Foxglove 崩溃
        # voxel_size 单位是米。0.2 表示每 20cm 只保留一个点。
        # 如果地图非常大，建议开启；如果想看原图，注释掉这行。
        self.get_logger().info(f'原始点数: {len(pcd.points)}')
        pcd = pcd.voxel_down_sample(voxel_size=0.2) 
        self.get_logger().info(f'下采样后点数: {len(pcd.points)}')

        # 转换点云为 numpy 数组
        points = np.asarray(pcd.points)

        # 4. 构建 ROS 2 消息
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "odom"

        # 使用 sensor_msgs_py 快速打包
        pc2_msg = pc2.create_cloud_xyz32(header, points)

        # 5. 发布一次
        self.global_map_pub.publish(pc2_msg)
        self.get_logger().info('地图已发布 (Latched). 请打开 Foxglove 查看 /global_map 话题')

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
