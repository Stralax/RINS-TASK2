#!/usr/bin/env python3
# filepath: /home/beta/RINS-TASK2/helper_scripts/bird_detector.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time

class BirdDetector(Node):
    def __init__(self):
        super().__init__('bird_detector_node')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', 'yolov8n.pt'),
                ('confidence_threshold', 0.4),
                ('device', ''),  # Empty string for auto (CPU/GPU)
                ('input_image_topic', '/camera/image_raw'),
                ('publish_visualization', True),
                ('bird_class_id', 16),  # COCO class ID for bird
                ('custom_model', False)  # Set to True if using a custom bird model
            ]
        )
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        self.publish_viz = self.get_parameter('publish_visualization').get_parameter_value().bool_value
        self.bird_class_id = self.get_parameter('bird_class_id').get_parameter_value().integer_value
        self.custom_model = self.get_parameter('custom_model').get_parameter_value().bool_value
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.get_logger().info(f"Loading YOLO model from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info("YOLO model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise
        
        # Create QoS profile for reliable image publishing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscribers and publishers
        self.image_subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            qos_profile)
        
        self.bird_image_publisher = self.create_publisher(
            Image,
            '/bird_detector/bird_image',
            qos_profile)
        
        if self.publish_viz:
            self.detection_viz_publisher = self.create_publisher(
                Image,
                '/bird_detector/detection_visualization',
                qos_profile)
        
        # Statistics
        self.detections_count = 0
        self.last_detection_time = time.time() - 10  # Initialize to avoid publishing immediately
        self.min_detection_interval = 2.0  # Minimum seconds between detections
        
        self.get_logger().info(f"Bird detector initialized. Listening on {self.input_topic}")
        self.get_logger().info(f"Bird detections will be published to /bird_detector/bird_image")
    
    def image_callback(self, msg):
        """Process incoming images to detect birds"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Create a copy for visualization
            viz_image = cv_image.copy()
            
            # Run YOLO detection
            if self.custom_model:
                # For a custom bird-specific model
                results = self.model.predict(
                    cv_image, 
                    conf=self.confidence_threshold, 
                    device=self.device,
                    verbose=False
                )
            else:
                # For standard model, filter only bird class
                results = self.model.predict(
                    cv_image, 
                    conf=self.confidence_threshold, 
                    classes=[self.bird_class_id],
                    device=self.device,
                    verbose=False
                )
            
            # Process detections
            bird_detected = False
            current_time = time.time()
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy() if not self.custom_model else None
                
                # Draw all detections for visualization
                for i, box in enumerate(boxes):
                    # Skip non-bird detections if using standard model
                    if not self.custom_model and classes[i] != self.bird_class_id:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box)
                    conf = confs[i]
                    
                    # Draw rectangle and confidence on visualization image
                    color = (0, 255, 0)  # Green for birds
                    cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        viz_image, 
                        f"Bird: {conf:.2f}", 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2
                    )
                    
                    # Only publish if enough time has passed since last detection
                    if current_time - self.last_detection_time >= self.min_detection_interval:
                        # Crop the bird image
                        bird_img = cv_image[y1:y2, x1:x2].copy()
                        
                        # Add a small margin if possible
                        img_h, img_w = cv_image.shape[:2]
                        margin = int(min((x2-x1), (y2-y1)) * 0.1)  # 10% margin
                        
                        # Calculate expanded boundaries with margin
                        x1_margin = max(0, x1 - margin)
                        y1_margin = max(0, y1 - margin)
                        x2_margin = min(img_w, x2 + margin)
                        y2_margin = min(img_h, y2 + margin)
                        
                        # Crop with margin
                        bird_img_with_margin = cv_image[y1_margin:y2_margin, x1_margin:x2_margin].copy()
                        
                        try:
                            # Convert the cropped image back to ROS Image message
                            bird_msg = self.bridge.cv2_to_imgmsg(bird_img_with_margin, "bgr8")
                            bird_msg.header = msg.header  # Keep original header
                            
                            # Publish the cropped bird image
                            self.bird_image_publisher.publish(bird_msg)
                            
                            self.detections_count += 1
                            self.last_detection_time = current_time
                            bird_detected = True
                            
                            self.get_logger().info(f"Bird detected! Total detections: {self.detections_count}")
                            
                            # Break after publishing first detection to avoid flooding
                            break
                        except CvBridgeError as e:
                            self.get_logger().error(f"Error converting bird crop to ROS message: {e}")
            
            # Publish visualization image if enabled
            if self.publish_viz:
                try:
                    viz_msg = self.bridge.cv2_to_imgmsg(viz_image, "bgr8")
                    viz_msg.header = msg.header
                    self.detection_viz_publisher.publish(viz_msg)
                except CvBridgeError as e:
                    self.get_logger().error(f"Error converting visualization image: {e}")
            
            if not bird_detected and results and len(results[0].boxes) > 0:
                self.get_logger().debug(f"Birds detected ({len(results[0].boxes)}), but skipped publishing due to time interval")
        
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    bird_detector = BirdDetector()
    
    try:
        rclpy.spin(bird_detector)
    except KeyboardInterrupt:
        pass
    finally:
        bird_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()