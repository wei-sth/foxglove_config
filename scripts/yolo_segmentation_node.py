#!/usr/bin/env python3
"""
YOLO segmentation publisher for foxglove_config.

Expected ROS parameters:
- colorImageTopic: compressed image topic to subscribe to, e.g. /camera/color/image_raw/compressed
- model_path: filesystem path to Ultralytics segmentation weights, e.g. /home/weizh/models/best.pt
- output_topic: segmentation result topic, defaults to /localmap/segmentation
- debug_image_topic: optional debug image topic, defaults to /localmap/segmentation/image
- conf_threshold: optional confidence threshold for detections, defaults to 0.25
- iou_threshold: optional NMS IoU threshold, defaults to 0.7
- device: optional Ultralytics device selector, defaults to auto
- queue_size: optional subscriber queue depth, defaults to 10

This node publishes foxglove_config/msg/SegmentationResult and a debug raw image.
"""

from __future__ import annotations

import numpy as np
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

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
        self.declare_parameter('debug_image_topic', '/localmap/segmentation/image')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.7)
        self.declare_parameter('device', '')
        self.declare_parameter('queue_size', 10)

        self._bridge = CvBridge()
        self._model = self._load_model()

        color_image_topic = self.get_parameter('colorImageTopic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value
        queue_size = int(self.get_parameter('queue_size').value)

        self._publisher = self.create_publisher(SegmentationResult, output_topic, 10)
        self._debug_publisher = self.create_publisher(Image, debug_image_topic, 10)
        self._subscription = self.create_subscription(
            CompressedImage,
            color_image_topic,
            self._image_callback,
            queue_size,
        )

        self.get_logger().info(
            f'Subscribed to {color_image_topic} and publishing segmentation results on {output_topic}'
        )
        self.get_logger().info(
            f'Publishing debug visualization on {debug_image_topic}'
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

    def _decode_compressed_image(self, msg: CompressedImage):
        return self._bridge.compressed_imgmsg_to_cv2(msg)

    def _mask_color(self, class_id: int) -> tuple[int, int, int]:
        palette = [
            (255, 56, 56),
            (255, 157, 151),
            (255, 112, 31),
            (255, 178, 29),
            (207, 210, 49),
            (72, 249, 10),
            (146, 204, 23),
            (61, 219, 134),
            (26, 147, 52),
            (0, 212, 187),
            (44, 153, 168),
            (0, 194, 255),
            (52, 69, 147),
            (100, 115, 255),
            (0, 24, 236),
            (132, 56, 255),
            (82, 0, 133),
            (203, 56, 255),
            (255, 149, 200),
            (255, 55, 199),
        ]
        return palette[int(class_id) % len(palette)]

    def _class_name(self, class_id: int) -> str:
        names = getattr(self._model, 'names', None)
        if isinstance(names, dict):
            return str(names.get(int(class_id), f'class_{class_id}'))
        if isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
            return str(names[int(class_id)])
        return f'class_{class_id}'

    def _make_debug_image(self, image: np.ndarray, result) -> np.ndarray:
        overlay = image.copy()
        boxes = getattr(result, 'boxes', None)
        masks = getattr(result, 'masks', None)

        if boxes is None or masks is None or boxes.cls is None or boxes.conf is None or masks.data is None:
            return overlay

        mask_data = masks.data.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
        confidences = boxes.conf.detach().cpu().numpy().astype(np.float32)
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.int32)

        for idx in range(len(classes)):
            class_id = int(classes[idx])
            color = self._mask_color(class_id)
            class_name = self._class_name(class_id)
            conf = float(confidences[idx])
            x1, y1, x2, y2 = xyxy[idx].tolist()

            mask = mask_data[idx]
            mask_binary = mask > 0.5
            if mask_binary.shape[:2] != overlay.shape[:2]:
                mask_binary = cv2.resize(
                    mask_binary.astype(np.uint8),
                    (overlay.shape[1], overlay.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            colored_mask = np.zeros_like(overlay, dtype=np.uint8)
            colored_mask[mask_binary] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.35, 0.0)

            contours, _ = cv2.findContours(
                mask_binary.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f'{class_name} {conf:.2f}'
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_y = max(y1, text_h + baseline + 4)
            cv2.rectangle(
                overlay,
                (x1, text_y - text_h - baseline - 4),
                (x1 + text_w + 6, text_y + 2),
                color,
                -1,
            )
            cv2.putText(
                overlay,
                label,
                (x1 + 3, text_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return overlay

    def _publish_debug_image(self, msg: CompressedImage, debug_image: np.ndarray) -> None:
        try:
            debug_msg = self._bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert debug image to ROS Image: {exc}')
            return

        debug_msg.header = msg.header
        self._debug_publisher.publish(debug_msg)

    def _image_callback(self, msg: CompressedImage) -> None:
        try:
            cv_image = self._decode_compressed_image(msg)
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

        debug_image = cv_image.copy()

        if not results:
            self._publisher.publish(segmentation_msg)
            self._publish_debug_image(msg, debug_image)
            return

        result = results[0]
        boxes = getattr(result, 'boxes', None)
        masks = getattr(result, 'masks', None)

        if boxes is None or masks is None or boxes.cls is None or boxes.conf is None or masks.data is None:
            self._publisher.publish(segmentation_msg)
            self._publish_debug_image(msg, debug_image)
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
        debug_image = self._make_debug_image(cv_image, result)
        self._publish_debug_image(msg, debug_image)


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
