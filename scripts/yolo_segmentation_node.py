#!/usr/bin/env python3
"""
YOLO segmentation publisher for foxglove_config.

Expected ROS parameters:
- colorImageTopic: compressed image topic to subscribe to, e.g. /camera/color/image_raw/compressed
- model_path: filesystem path to Ultralytics segmentation weights, e.g. /home/weizh/models/best.pt
- output_topic: segmentation result topic, defaults to /localmap/segmentation
- conf_threshold: optional confidence threshold for detections, defaults to 0.25
- iou_threshold: optional NMS IoU threshold, defaults to 0.7
- device: optional Ultralytics device selector, defaults to auto
- queue_size: optional subscriber queue depth, defaults to 10

This node publishes foxglove_config/msg/SegmentationResult and does not publish images.
"""

from __future__ import annotations

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from foxglove_config.msg import SegmentationInstance, SegmentationResult

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific
    YOLO = None
    _ULTRALYTICS_IMPORT_ERROR = exc
else:
    _ULTRALYTICS_IMPORT_ERROR = None


class YoloSegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__('yolo_segmentation_node')

        self.declare_parameter('colorImageTopic', '/camera/color/image_raw/compressed')
        self.declare_parameter('model_path', '/mnt/d/Dev/X-AnyLabeling/runs/segment/train4/weights/best.pt')   # note: must check model path train2/train3...
        self.declare_parameter('output_topic', '/localmap/segmentation')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.7)
        self.declare_parameter('device', '')
        self.declare_parameter('queue_size', 10)

        self._bridge = CvBridge()
        self._model = self._load_model()

        color_image_topic = self.get_parameter('colorImageTopic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        queue_size = int(self.get_parameter('queue_size').value)

        self._publisher = self.create_publisher(SegmentationResult, output_topic, 10)
        self._subscription = self.create_subscription(
            CompressedImage,
            color_image_topic,
            self._image_callback,
            queue_size,
        )

        self.get_logger().info(
            f'Subscribed to {color_image_topic} and publishing segmentation results on {output_topic}'
        )

    def _load_model(self):
        if YOLO is None:
            raise RuntimeError(f'ultralytics is not available: {_ULTRALYTICS_IMPORT_ERROR}')

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not model_path:
            raise RuntimeError('Parameter "model_path" must be set to a valid Ultralytics weights file')

        device = self.get_parameter('device').get_parameter_value().string_value
        if device:
            return YOLO(model_path).to(device)
        return YOLO(model_path)

    def _image_callback(self, msg: CompressedImage) -> None:
        try:
            cv_image = self._bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as exc:
            self.get_logger().error(f'Failed to decode compressed image: {exc}')
            return

        conf_threshold = float(self.get_parameter('conf_threshold').value)
        iou_threshold = float(self.get_parameter('iou_threshold').value)

        try:
            results = self._model.predict(
                source=cv_image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f'YOLO inference failed: {exc}')
            return

        segmentation_msg = SegmentationResult()
        segmentation_msg.header = msg.header
        segmentation_msg.image_width = int(cv_image.shape[1])
        segmentation_msg.image_height = int(cv_image.shape[0])
        segmentation_msg.instances = []

        if not results:
            self._publisher.publish(segmentation_msg)
            return

        result = results[0]
        boxes = getattr(result, 'boxes', None)
        masks = getattr(result, 'masks', None)

        if boxes is None or masks is None or boxes.cls is None or boxes.conf is None or masks.data is None:
            self._publisher.publish(segmentation_msg)
            return

        mask_data = masks.data.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(np.uint32)
        confidences = boxes.conf.detach().cpu().numpy().astype(np.float32)

        for idx in range(len(classes)):
            mask = mask_data[idx]
            mask_binary = (mask > 0.5).astype(np.uint8)
            mask_height, mask_width = mask_binary.shape[:2]
            encoded_mask = np.packbits(mask_binary.reshape(-1), bitorder='little').astype(np.uint8).tolist()

            instance_msg = SegmentationInstance()
            instance_msg.class_id = int(classes[idx])
            instance_msg.confidence = float(confidences[idx])
            instance_msg.mask_width = int(mask_width)
            instance_msg.mask_height = int(mask_height)
            instance_msg.mask_encoding = 0
            instance_msg.mask_data = encoded_mask
            segmentation_msg.instances.append(instance_msg)

        self._publisher.publish(segmentation_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = YoloSegmentationNode()
        rclpy.spin(node)
    except Exception as exc:
        if node is not None:
            node.get_logger().error(str(exc))
        else:
            print(str(exc))
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()