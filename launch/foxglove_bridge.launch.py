from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_to_base',
            arguments=['0', '0', '0', '0', '0', '0', 'camera_color_optical_frame', 'rslidar']
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'rslidar']
        ),

        Node(
            package='foxglove_config',
            executable='map_publisher.py',
            name='map_publisher',
            output='screen'
        ),

        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{
                'port': 8765,
                'address': '0.0.0.0',
                'send_buffer_limit': 10000000,
                'use_compression': True,
            }],
            output='screen'
        )
    ])