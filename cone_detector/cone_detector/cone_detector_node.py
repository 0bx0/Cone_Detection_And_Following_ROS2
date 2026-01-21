#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cone_msgs.msg import ConeDetection, ConeDetectionArray
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO
import numpy as np
from ament_index_python.packages import get_package_share_directory
class ConeDetectorNode(Node):
    def __init__(self):
        super().__init__('cone_detector_node')
        
        # --- Parameters ---
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('model_path', '') 
        self.declare_parameter('conf_thresh', 0.4)
        self.declare_parameter('debug', True)

        self.image_topic = self.get_parameter('image_topic').value
        model_path_param = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('conf_thresh').value
        self.debug = self.get_parameter('debug').value

        try:
            pkg_share = get_package_share_directory('cone_detector')
            self.model_path = os.path.join(pkg_share, 'models', 'best.pt')
        except Exception as e:
            self.get_logger().error(f"❌ Could not find cone_detector package: {e}")
            raise e

        self.get_logger().info(f"Loading Model: {self.model_path}")
                
        # Explicit file check to prevent silent failures
        if not os.path.exists(self.model_path):
             self.get_logger().error(f"❌ MODEL FILE NOT FOUND: {self.model_path}")
             # Check what IS there to help
             parent_dir = os.path.dirname(self.model_path)
             if os.path.exists(parent_dir):
                 files = os.listdir(parent_dir)
                 self.get_logger().error(f"   📂 Found in {parent_dir}: {files}")
             else:
                 self.get_logger().error(f"   📂 Directory does not exist: {parent_dir}")
             raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            self.model = YOLO(self.model_path)
            # Get class names from model
            self.class_names = self.model.names
            self.get_logger().info(f"Model classes: {self.class_names}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO: {e}")
            raise e

        # --- Setup ---
        self.bridge = CvBridge()
        
        # Color ranges for HSV detection (OpenCV format)
        self.color_ranges = {
            'red': {
                # Red wraps around HSV hue range → two segments
                'lower': [np.array([0, 50, 50]), np.array([170, 50, 50])],
                'upper': [np.array([10, 255, 255]), np.array([180, 255, 255])]
            },
            'blue': {
                'lower': np.array([100, 50, 50]),
                'upper': np.array([130, 255, 255])
            },
            'yellow': {
                'lower': np.array([20, 100, 100]),
                'upper': np.array([30, 255, 255])
            },
            'orange': {
                'lower': np.array([10, 100, 100]),
                'upper': np.array([25, 255, 255])
            },
            'green': {
                'lower': np.array([35, 50, 50]),
                'upper': np.array([85, 255, 255])
            },
            'cyan': {
                'lower': np.array([85, 50, 50]),
                'upper': np.array([100, 255, 255])
            },
        }

        
        # Publishers
        self.pub_annotated = self.create_publisher(Image, 'cone_detector/annotated_image', 10)
        self.pub_detections = self.create_publisher(ConeDetectionArray, '/cone_detector/detections', 10)
        
        # Subscribers
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        
        self.get_logger().info("Cone Detector (Simplified) Ready.")
        self.get_logger().info(f"Subscribed to: {self.image_topic}")

        # Metrics for debugging
        self.last_log_time = self.get_clock().now()
        self.frame_count = 0
        self.total_detections = 0

    def detect_color_in_roi(self, cv_image, x1, y1, x2, y2):
        """
        Detect the dominant color within a bounding box using HSV color ranges
        """
        roi = cv_image[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        max_color = "unknown"
        max_pixels = 0
        
        # Check each color range
        for color_name, ranges in self.color_ranges.items():
            if isinstance(ranges['lower'], list):
                # Red has two ranges (wraps around 180)
                mask1 = cv2.inRange(hsv_roi, ranges['lower'][0], ranges['lower'][1])
                mask2 = cv2.inRange(hsv_roi, ranges['upper'][0], ranges['upper'][1])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_roi, ranges['lower'], ranges['upper'])
            
            pixels = cv2.countNonZero(mask)
            if pixels > max_pixels:
                max_pixels = pixels
                max_color = color_name
        
        return max_color

    def image_cb(self, msg):
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge conversion failed: {e}")
            return
        
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)

        # DEBUG: Log raw image stats
        if self.frame_count % 10 == 0:
             self.get_logger().info(f"📷 RX IMAGE | {msg.width}x{msg.height} | Frame: {self.frame_count}")

        # Run Inference
        start_t = self.get_clock().now()
        results = self.model.predict(cv_image, conf=self.conf_thresh, verbose=False)[0]
        infer_dur = (self.get_clock().now() - start_t).nanoseconds / 1e6 # ms
        
        # Prepare Output Messages
        det_array = ConeDetectionArray()
        det_array.header = msg.header
        
        annotated_img = cv_image.copy()

        FOCAL_LENGTH = 700.0
        REAL_HEIGHT = 0.3

        detections_in_frame = 0
        
        if results.boxes:
            detections_in_frame = len(results.boxes)
            self.total_detections += detections_in_frame
            
            # DEBUG: Log EVERY detection frame heavily
            log_msg = f"🔍 DETECTED {detections_in_frame} CONES (Infer: {infer_dur:.1f}ms):"
            
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                if h <= 0: continue

                distance = (FOCAL_LENGTH * REAL_HEIGHT) / h
                
                # Detect color within bounding box using OpenCV
                color_name = self.detect_color_in_roi(cv_image, x1, y1, x2, y2)
                
                # Append to log
                log_msg += f"\n    [{i}] {color_name.upper()} | Dist: {distance:5.2f}m | Conf: {conf:.2f} | Center: {cx:4.0f}x"

                # Populate Message
                det = ConeDetection()
                det.header = msg.header
                det.cx, det.cy = float(cx), float(cy)
                det.width, det.height = float(w), float(h)
                det.distance, det.confidence = float(distance), float(conf)
                det.color = color_name  # Add color detected by OpenCV
                det_array.detections.append(det)

                # Annotation
                label = f"{color_name} {distance:.2f}m ({conf:.2f})"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.get_logger().info(log_msg)

        # Publish Messages
        self.pub_detections.publish(det_array)
        
        if infer_dur > 80.0:
             self.get_logger().warn(f"⚠️ SLOW FRAME: {infer_dur:.1f}ms")

        try:
            cv2.putText(annotated_img, f"Cones: {detections_in_frame} | Infer: {infer_dur:.0f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
            annotated_msg.header = msg.header
            self.pub_annotated.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
