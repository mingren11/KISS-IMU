import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'kiss_imu_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ming Ren',
    maintainer_email='shuaiqiduoyi@gmail.com',
    description='Real-time ROS2 LiDAR-inertial odometry front-end built on KISS-IMU.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lio_node = kiss_imu_ros.lio_node:main',
        ],
    },
)
