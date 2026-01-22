import os
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    share_dir = get_package_share_directory('foxglove_config')
    rviz_config_file = os.path.join(share_dir, 'rviz', 'det.rviz')  # mapping.rviz

    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_to_base',
            arguments=['0', '0', '0', '0', '0', '0', 'camera_color_optical_frame', 'rslidar']
        ),

        # publish static global map and NavSatFix
        # Node(
        #     package='foxglove_config',
        #     executable='map_publisher.py',
        #     name='map_publisher',
        #     output='screen'
        # ),

        Node(
            package='foxglove_config',
            executable='obstacle_detector_node',
            name='obstacle_detector_node',
            output='screen'
        ),

        Node(
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        )

        # Node(
        #     package='foxglove_bridge',
        #     executable='foxglove_bridge',
        #     name='foxglove_bridge',
        #     parameters=[{
        #         'port': 8765,
        #         'address': '0.0.0.0',
        #         'send_buffer_limit': 10000000,
        #         'use_compression': True,
        #     }],
        #     output='screen'
        # )
    ])