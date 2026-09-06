"""Publishes a rough simulated ultrasonic reading as sensor_msgs/Range.

Gazebo's native sensor-rendering pipeline (needed for even a bare lidar-type
sensor) proved unreliable on this VM's virtual GPU -- see CLAUDE.md for the
debugging trail. Instead of a real simulated sensor, this node computes the
straight-line distance from the gripper's ultrasonic_link frame (via TF, from
robot_state_publisher) to the nearest tracked object (via Gazebo pose data,
bridged in as tf2_msgs/TFMessage) and publishes that as if a real ultrasonic
sensor had measured it. Matches the project's "rough simulation is fine" goal.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Range
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

TRACKED_OBJECT_NAMES = ('can', 'bottle')
RANGE_MIN = 0.02
RANGE_MAX = 0.4
FIELD_OF_VIEW = 0.26  # radians, roughly a narrow-cone ultrasonic sensor


class UltrasonicRangeNode(Node):

    def __init__(self):
        super().__init__('ultrasonic_range_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.object_positions = {}
        self.create_subscription(
            TFMessage, 'world_pose', self.pose_callback, 10)

        self.publisher = self.create_publisher(Range, 'ultrasonic/range', 10)
        self.create_timer(0.1, self.timer_callback)

    def pose_callback(self, msg):
        for t in msg.transforms:
            frame = t.child_frame_id
            for name in TRACKED_OBJECT_NAMES:
                if name in frame:
                    tl = t.transform.translation
                    self.object_positions[name] = (tl.x, tl.y, tl.z)

    def timer_callback(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'world', 'ultrasonic_link', Time())
        except Exception:
            return

        sx = tf.transform.translation.x
        sy = tf.transform.translation.y
        sz = tf.transform.translation.z

        nearest = None
        for (ox, oy, oz) in self.object_positions.values():
            distance = math.sqrt((ox - sx) ** 2 + (oy - sy) ** 2 + (oz - sz) ** 2)
            if nearest is None or distance < nearest:
                nearest = distance

        if nearest is None:
            return

        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ultrasonic_link'
        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = FIELD_OF_VIEW
        msg.min_range = RANGE_MIN
        msg.max_range = RANGE_MAX
        msg.range = min(max(nearest, RANGE_MIN), RANGE_MAX)
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = UltrasonicRangeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
