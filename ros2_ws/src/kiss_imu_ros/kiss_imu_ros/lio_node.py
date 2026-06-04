"""ROS2 node: subscribe IMU + PointCloud2, publish Odometry + TF."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from .ring_buffer import ImuBuffer
from .pointcloud2 import xyz_from_pointcloud2
from .online_estimator import OnlineEstimator, EstimatorConfig


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def params_to_cfg(p: dict) -> EstimatorConfig:
    R = list(p['R_I_L'])
    R3 = [R[0:3], R[3:6], R[6:9]]
    return EstimatorConfig(
        lo_model=p['lo_model'], device=p['device'], use_submap=bool(p['use_submap']),
        lm_weight=tuple(p['lm_weight']), gravity=tuple(p['gravity']),
        R_I_L=R3, T_I_L=list(p['T_I_L']),
    )


class LioNode(Node):
    def __init__(self):
        super().__init__('lio_node')
        decl = dict(
            imu_topic='/imu', points_topic='/points', odom_topic='odometry',
            odom_frame='odom', base_frame='base_link', lo_model='kiss_icp',
            use_submap=False, device='cuda:0', lm_weight=[1.0, 0.1, 1.0, 0.1, 0.1],
            gravity=[0.0, 0.0, 9.81], R_I_L=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
            T_I_L=[0.0, 0.0, 0.0], voxel_size=0.5, publish_path=True)
        for k, v in decl.items():
            self.declare_parameter(k, v)
        gp = lambda k: self.get_parameter(k).value
        self.odom_frame = gp('odom_frame'); self.base_frame = gp('base_frame')
        self.publish_path = gp('publish_path')

        self.buf = ImuBuffer()
        self.estimator = OnlineEstimator(params_to_cfg({k: gp(k) for k in
            ['lo_model', 'device', 'use_submap', 'lm_weight', 'gravity', 'R_I_L', 'T_I_L']}))
        self._last_scan_t = None

        imu_cb = MutuallyExclusiveCallbackGroup()
        pts_cb = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Imu, gp('imu_topic'), self.on_imu,
                                 qos_profile_sensor_data, callback_group=imu_cb)
        self.create_subscription(PointCloud2, gp('points_topic'), self.on_points,
                                 qos_profile_sensor_data, callback_group=pts_cb)
        self.odom_pub = self.create_publisher(Odometry, gp('odom_topic'), 10)
        self.path_pub = self.create_publisher(Path, 'path', 10) if self.publish_path else None
        self.tf_bc = TransformBroadcaster(self)
        self._path = Path(); self._path.header.frame_id = self.odom_frame
        self.get_logger().info('lio_node ready')

    def on_imu(self, msg: Imu):
        t = _stamp_to_sec(msg.header.stamp)
        self.buf.append(t,
                        (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z),
                        (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))

    def on_points(self, msg: PointCloud2):
        t = _stamp_to_sec(msg.header.stamp)
        t0 = self._last_scan_t if self._last_scan_t is not None else (t - 0.1)
        ts, acc, gyro = self.buf.pop_window(t0, t)
        self._last_scan_t = t
        scan = xyz_from_pointcloud2(msg)
        result = self.estimator.step(scan, acc, gyro, ts)
        self._publish(result, msg.header.stamp)

    def _publish(self, result, stamp):
        p, q = result.pose[:3], result.pose[3:]
        odom = Odometry()
        odom.header.stamp = stamp; odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, p)
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, \
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = map(float, q)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, result.vel)
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp; tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, p)
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
        self.tf_bc.sendTransform(tf)

        if self.path_pub is not None:
            ps = PoseStamped(); ps.header = odom.header; ps.pose = odom.pose.pose
            self._path.poses.append(ps); self._path.header.stamp = stamp
            self.path_pub.publish(self._path)


def main(args=None):
    rclpy.init(args=args)
    node = LioNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
