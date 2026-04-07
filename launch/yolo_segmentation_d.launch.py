import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = get_package_share_directory('foxglove_config')
    rviz_config_file = os.path.join(share_dir, 'rviz', 'yolo_segmentation_d.rviz')

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(share_dir, 'config', 'mower.yaml'),
        description='Path to the ROS2 parameters file to use.'
    )

    use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz2.'
    )

    parameter_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        params_declare,
        use_rviz,
        Node(
            package='foxglove_config',
            executable='yolo_segmentation_node.py',
            name='yolo_segmentation_node',
            parameters=[parameter_file],
            output='screen'
        ),
        Node(
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),
    ])
