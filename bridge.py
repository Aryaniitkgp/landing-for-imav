#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from gz.transport13 import Node as GzNode
from gz.msgs10.image_pb2 import Image as GzImage, PixelFormatType
import cv2
import numpy as np
from cv_bridge import CvBridge

# Robust format mapping
PIXEL_FORMAT = {
    PixelFormatType.L_INT8: 'mono8',
    PixelFormatType.L_INT16: 'mono16',
    PixelFormatType.RGB_INT8: 'rgb8',
    PixelFormatType.RGBA_INT8: 'rgba8',
    PixelFormatType.BGRA_INT8: 'bgra8',
    PixelFormatType.BGR_INT8: 'bgr8',
}

class UniversalCameraBridge(Node):
    def __init__(self):
        super().__init__('universal_camera_bridge')
        self.bridge = CvBridge()
        self.gz_node = GzNode()
        
        self.pubs = {}
        self._frame_counts = {}
        self._first_logged = set()

        # Explicit best-effort QoS with depth=1 to prevent lag/stale frames
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge_fixed_cameras()

    # Fixed Gazebo -> ROS2 camera mapping (no auto-discovery).
    CAMERA_MAP = {
        '/drone/down_camera':  '/drone/camera/down_camera',
        '/drone/front_camera': '/drone/camera/front_camera',
    }

    def bridge_fixed_cameras(self):
        self.get_logger().info('Bridging fixed camera topics (auto-discovery disabled)...')

        for gz_topic, ros_topic in self.CAMERA_MAP.items():
            # Setup ROS 2 Publisher
            self.pubs[gz_topic] = self.create_publisher(Image, ros_topic, self.qos_profile)
            self._frame_counts[gz_topic] = 0

            # Setup Gazebo Subscriber Subscription
            self.gz_node.subscribe(GzImage, gz_topic, lambda msg, t=gz_topic: self._callback(msg, t))
            self.get_logger().info(f'🚀 Bridged: [GZ] {gz_topic} ➔ [ROS2] {ros_topic}')

    def _callback(self, gz_img, gz_topic):
        try:
            encoding = PIXEL_FORMAT.get(gz_img.pixel_format_type, 'rgb8')
            channels = 1 if encoding.startswith('mono') else 3
            
            # Unpack raw bytes safely into NumPy matrix
            img_np = np.frombuffer(gz_img.data, dtype=np.uint8).reshape((gz_img.height, gz_img.width, channels))
            
            # Downsample to 640x480 to keep DDS network traffic minimal and fast
            img_resized = cv2.resize(img_np, (640, 480))
            
            # Transpile directly to ROS 2 Image Message
            msg = self.bridge.cv2_to_imgmsg(img_resized, encoding=encoding)
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_optical_frame"
            
            self.pubs[gz_topic].publish(msg)

            self._frame_counts[gz_topic] += 1
            if gz_topic not in self._first_logged:
                self._first_logged.add(gz_topic)
                self.get_logger().info(f'✅ Streaming started for {gz_topic} (Resized {gz_img.width}x{gz_img.height} ➔ 640x480)')
            elif self._frame_counts[gz_topic] % 30 == 0:
                short_name = gz_topic.split('/')[-2] if len(gz_topic.split('/')) > 1 else 'camera'
                self.get_logger().info(f'Frames processed [{short_name}]: {self._frame_counts[gz_topic]}')
                
        except Exception as e:
            self.get_logger().error(f'Bridge processing error on {gz_topic}: {str(e)}')

def main():
    rclpy.init()
    node = UniversalCameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()