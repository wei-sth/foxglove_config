import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = get_package_share_directory('foxglove_config')
    parameter_file = LaunchConfiguration('params_file')
    rviz_config_file = os.path.join(share_dir, 'rviz', 'mapping.rviz')  # mapping.rviz

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            share_dir, 'config', 'go2_grass.yaml'),
        description='FPath to the ROS2 parameters file to use.')

    return LaunchDescription([
        params_declare,
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments='0.0 0.0 0.0 0.0 0.0 0.0 map odom'.split(' '),
            parameters=[parameter_file],
            output='screen'
        ),
        Node(
            package='foxglove_config',
            executable='livo',
            name='livo',
            parameters=[parameter_file],
            output='screen'
        ),
        # record slam result
        # ExecuteProcess(
        #     cmd=['ros2', 'bag', 'record', '-o', '/home/weizh/data/liorf_output', 
        #          '/liorf/mapping/cloud_registered_raw'],
        #     output='screen'
        # ),
        Node(
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        )
    ])