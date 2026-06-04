import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('kiss_imu_ros'), 'config', 'lio.yaml')
    return LaunchDescription([
        Node(
            package='kiss_imu_ros',
            executable='lio_node',
            name='lio_node',
            output='screen',
            parameters=[cfg],
        ),
    ])
