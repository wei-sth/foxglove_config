#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Header

class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        
        # 创建发布者
        self.fix_pub = self.create_publisher(NavSatFix, '/fix', 10)
        
        # 定时发布（1Hz）
        self.timer = self.create_timer(1.0, self.publish_fix)
        
        self.get_logger().info('NavSatFix 发布器已启动')
    
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
