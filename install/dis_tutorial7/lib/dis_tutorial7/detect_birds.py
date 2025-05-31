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

from sklearn.decomposition import PCA
import tf2_geometry_msgs as tfg
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import tf_transformations

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.duration import Duration
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Vector3Stamped, PoseArray, Pose

from dataclasses import dataclass

@dataclass
class BirdTracker:
    position: np.ndarray  # 3D position in map frame
    normal: np.ndarray    # Normal vector
    last_seen: float      # Timestamp
    marker_id: int        # Assigned marker ID

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
                ('input_image_topic', '/top_camera/rgb/preview/image_raw'),
                ('publish_visualization', True),
                ('bird_class_id', 14),  # COCO class ID for bird
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
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_id = 0
        self.marker_lifetime = 0.0  # Lifetime for markers in seconds

        self.tracked_birds = []  # List of BirdTracker objects
        # self.bird_max_age = 5.0  # How long to keep a bird in tracking (seconds)
        self.bird_distance_threshold = 0.5  # Distance threshold to consider it's the same bird (meters)

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

        # Add PointCloud subscription
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/top_camera/rgb/preview/depth/points',  # Adjust topic as needed
            self.pointcloud_callback,
            qos_profile)
        
        # Add marker publisher
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/bird_markers',
            qos_profile)
        
        # Store the latest point cloud data
        self.latest_pointcloud = None
        
        # Statistics
        self.detections_count = 0
        self.last_detection_time = time.time() - 10  # Initialize to avoid publishing immediately
        self.min_detection_interval = 2.0  # Minimum seconds between detections
        
        self.get_logger().info(f"Bird detector initialized. Listening on {self.input_topic}")
        self.get_logger().info(f"Bird detections will be published to /bird_detector/bird_image")
    
    #  Add point cloud callback
    def pointcloud_callback(self, data):
        """Store the latest point cloud data for processing"""
        try:
            self.latest_pointcloud = {
                'data': data,
                'stamp': data.header.stamp
            }
            self.get_logger().debug("Received new point cloud data")
        except Exception as e:
            self.get_logger().error(f"Error in pointcloud callback: {e}")

    def transform_point_to_map(self, point_3d):
        """Transform a point from camera frame to map frame"""
        try:
            # Create PointStamped object
            point_stamped = PointStamped()
            point_stamped.header.frame_id = "map"  # Adjust frame ID as needed
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = float(point_3d[0])
            point_stamped.point.y = float(point_3d[1])
            point_stamped.point.z = float(point_3d[2])
            
            # Get latest transform
            transform = self.tf_buffer.lookup_transform(
                "map", 
                "base_link",  # Adjust frame ID as needed
                rclpy.time.Time(),  # Get latest transform
                rclpy.duration.Duration(seconds=1.0)
            )
            
            # Transform the point
            transformed_point = tfg.do_transform_point(point_stamped, transform)
            
            return np.array([
                transformed_point.point.x, 
                transformed_point.point.y, 
                transformed_point.point.z
            ])
            
        except TransformException as e:
            self.get_logger().warn(f"Could not transform point: {e}")
            return None

    def transform_vector_to_map(self, vector):
        """Transform a direction vector from camera frame to map frame"""
        try:
            # Create Vector3Stamped for the normal vector
            vector_stamped = Vector3Stamped()
            vector_stamped.header.frame_id = "top_camera"  # Adjust frame ID as needed
            vector_stamped.header.stamp = self.get_clock().now().to_msg()
            vector_stamped.vector.x = float(vector[0])
            vector_stamped.vector.y = float(vector[1])
            vector_stamped.vector.z = float(vector[2])
            
            # Get latest transform
            transform = self.tf_buffer.lookup_transform(
                "map", 
                "base_link",  # Adjust frame ID as needed
                rclpy.time.Time(),  # Get latest transform
                rclpy.duration.Duration(seconds=1.0)
            )
            
            # Transform the vector
            transformed_vector = tfg.do_transform_vector3(vector_stamped, transform)
            
            # Return as numpy array
            result = np.array([
                transformed_vector.vector.x, 
                transformed_vector.vector.y, 
                transformed_vector.vector.z
            ])
            
            # Normalize the vector
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
            
            return result
            
        except TransformException as e:
            self.get_logger().warn(f"Could not transform vector: {e}")
            return vector  # Fall back to original vector if transformation fails

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
                    imgsz=640, # Resize to standard input size
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
                    # Use color based on confidence (red if < 0.9, green if >= 0.9)
                    color = (0, 255, 0) if conf >= 0.91 else (0, 165, 255)
                    
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
                    # AND confidence is at least 0.90
                    if current_time - self.last_detection_time >= self.min_detection_interval and conf >= 0.90:
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
                        
                        # Calculate bird position and normal
                        position_3d, normal = self.get_bird_position_and_normal(x1, y1, x2, y2)
                        
                        # Display the cropped bird image
                        cv2.imshow("Detected Bird", bird_img_with_margin)
                        cv2.waitKey(1)  # Update display
                        
                        try:
                            # Convert the cropped image back to ROS Image message
                            bird_msg = self.bridge.cv2_to_imgmsg(bird_img_with_margin, "bgr8")
                            bird_msg.header = msg.header  # Keep original header
                            
                            # Publish the cropped bird image
                            self.bird_image_publisher.publish(bird_msg)
                            
                            self.detections_count += 1
                            self.last_detection_time = current_time
                            bird_detected = True
                            
                            # Publish bird markers if position and normal were calculated successfully
                            if position_3d is not None and normal is not None:
                                self.publish_bird_marker(position_3d, normal, conf)
                            
                            self.get_logger().info(f"High-confidence bird detected ({conf:.2f})! Total detections: {self.detections_count}")
                            
                            # Break after publishing first detection to avoid flooding
                            break
                        except CvBridgeError as e:
                            self.get_logger().error(f"Error converting bird crop to ROS message: {e}")
                    elif conf >= 0.91:
                        # If confidence is high but we're still in cooldown period
                        self.get_logger().debug(f"High-confidence bird detected ({conf:.2f}), but still in cooldown period")
            
            # Display the visualization image
            cv2.imshow("Bird Detection", viz_image)
            cv2.waitKey(1)  # Update display
            
            # Publish visualization image if enabled
            if self.publish_viz:
                try:
                    viz_msg = self.bridge.cv2_to_imgmsg(viz_image, "bgr8")
                    viz_msg.header = msg.header
                    self.detection_viz_publisher.publish(viz_msg)
                except CvBridgeError as e:
                    self.get_logger().error(f"Error converting visualization image: {e}")
            
            if not bird_detected and results and len(results[0].boxes) > 0:
                # Find max confidence
                max_conf = 0
                for result in results:
                    if len(result.boxes.conf) > 0:
                        max_conf = float(result.boxes.conf.max())
                
                # Log appropriate message based on confidence
                if max_conf >= 0.90:
                    self.get_logger().debug(f"High-confidence bird ({max_conf:.2f}) detected, but skipped due to time interval")
                else:
                    self.get_logger().debug(f"Low-confidence bird ({max_conf:.2f}) detected, skipped (threshold: 0.90)")
        
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def get_bird_position_and_normal(self, x1, y1, x2, y2):
        """Calculate 3D position and normal for detected bird using the point cloud data"""
        if self.latest_pointcloud is None:
            self.get_logger().warn("No point cloud data available")
            return None, None
        
        try:
            # Convert the point cloud to a numpy array
            pc_data = pc2.read_points_numpy(
                self.latest_pointcloud['data'],
                field_names=("x", "y", "z")
            ).reshape((self.latest_pointcloud['data'].height, self.latest_pointcloud['data'].width, 3))
            
            # Calculate the center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Get the bird width and height
            width = x2 - x1
            height = y2 - y1
            
            # Collect points from the bird area
            bird_points = []
            
            # Define grid size for sampling
            grid_size = 8
            step_x = width / grid_size
            step_y = height / grid_size
            
            # Sample points in a grid pattern
            for i in range(grid_size):
                for j in range(grid_size):
                    px = int(x1 + i * step_x)
                    py = int(y1 + j * step_y)
                    
                    # Check if point is within bounds
                    if 0 <= px < pc_data.shape[1] and 0 <= py < pc_data.shape[0]:
                        point = pc_data[py, px]
                        
                        # Check if point is valid (not NaN, not infinite)
                        if np.isfinite(point).all() and not np.isnan(point).any():
                            bird_points.append(point)
            
            # If we have enough valid points
            if len(bird_points) >= 3:
                # Convert list to numpy array
                bird_points_array = np.array(bird_points)
                
                # Calculate median position (more robust than mean)
                position_3d = np.median(bird_points_array, axis=0)
                
                # Check if position is valid
                if not np.isfinite(position_3d).all() or np.isnan(position_3d).any():
                    self.get_logger().warn("Invalid 3D position calculated")
                    return None, None
                
                # Calculate normal using PCA
                pca = PCA(n_components=3)
                pca.fit(bird_points_array)
                
                # The normal is perpendicular to the first two principal components
                normal = np.cross(pca.components_[0], pca.components_[1])
                
                # Normalize the normal vector
                normal = normal / np.linalg.norm(normal)
                
                # Ensure the normal points "forward" (usually towards the camera)
                if normal[2] < 0:  # Z is typically depth
                    normal = -normal
                
                self.get_logger().info(f"Bird 3D position: {position_3d}, normal: {normal}")
                
                return position_3d, normal
            else:
                self.get_logger().warn(f"Not enough valid points for bird (found {len(bird_points)})")
                return None, None
                
        except Exception as e:
            self.get_logger().error(f"Error calculating bird position and normal: {e}")
            return None, None

    def publish_bird_marker(self, position_3d, normal, confidence):
        """Publish markers for bird position and normal vector"""
        try:
            # Transform position and normal to map frame
            map_position = self.transform_point_to_map(position_3d)
            map_normal = self.transform_vector_to_map(normal)
            
            if map_position is None or map_normal is None:
                self.get_logger().warn("Could not transform bird position or normal to map frame")
                return
            
            # Check if this bird is already being tracked
            marker_id = None
            is_new_bird = True
            
            # Check against existing birds
            for i, bird in enumerate(self.tracked_birds):
                distance = np.linalg.norm(map_position - bird.position)
                if distance < self.bird_distance_threshold:
                    # Update existing bird
                    self.tracked_birds[i].position = map_position
                    self.tracked_birds[i].normal = map_normal
                    self.tracked_birds[i].last_seen = time.time()
                    marker_id = bird.marker_id
                    is_new_bird = False
                    self.get_logger().info(f"Updated existing bird #{marker_id}, distance: {distance:.2f}m")
                    break
            
            # If it's a new bird, add it to tracking
            if is_new_bird:
                marker_id = self.marker_id
                self.tracked_birds.append(BirdTracker(
                    position=map_position,
                    normal=map_normal,
                    last_seen=time.time(),
                    marker_id=marker_id
                ))
                self.marker_id += 1
                self.get_logger().info(f"New bird detected! Assigned marker ID: {marker_id}")
            
            # Create marker array
            marker_array = MarkerArray()
            
            # Bird position marker (sphere)
            position_marker = Marker()
            position_marker.header.frame_id = "map"
            position_marker.header.stamp = self.get_clock().now().to_msg()
            position_marker.ns = "bird_positions"
            position_marker.id = marker_id
            position_marker.type = Marker.SPHERE
            position_marker.action = Marker.ADD
            
            position_marker.pose.position.x = map_position[0]
            position_marker.pose.position.y = map_position[1]
            position_marker.pose.position.z = map_position[2]
            position_marker.pose.orientation.w = 1.0
            
            # Size based on confidence
            scale = 0.1 + (confidence * 0.1)
            position_marker.scale.x = position_marker.scale.y = position_marker.scale.z = scale
            
            # Color: Green for high confidence, more yellow for lower
            position_marker.color.r = 1.0 - confidence  # Less red for higher confidence
            position_marker.color.g = 1.0
            position_marker.color.b = 0.0
            position_marker.color.a = 0.8
            
            # Set lifetime - a very long lifetime instead of refreshing
            lifetime = 0.0  # 0 means forever
            if lifetime > 0:
                position_marker.lifetime.sec = int(lifetime)
                position_marker.lifetime.nanosec = int((lifetime % 1) * 1e9)
            
            marker_array.markers.append(position_marker)
            
            # Normal direction marker (arrow)
            normal_marker = Marker()
            normal_marker.header.frame_id = "map"
            normal_marker.header.stamp = self.get_clock().now().to_msg()
            normal_marker.ns = "bird_normals"
            normal_marker.id = marker_id
            normal_marker.type = Marker.ARROW
            normal_marker.action = Marker.ADD
            
            normal_marker.pose.position.x = map_position[0]
            normal_marker.pose.position.y = map_position[1]
            normal_marker.pose.position.z = map_position[2]
            
            # Calculate orientation based on normal vector
            reference = np.array([1.0, 0.0, 0.0])
            
            # Handle case where normal is parallel to reference
            if np.allclose(map_normal, reference) or np.allclose(map_normal, -reference):
                if np.allclose(map_normal, reference):
                    normal_marker.pose.orientation.w = 1.0
                else:
                    normal_marker.pose.orientation.z = 1.0
            else:
                rotation_axis = np.cross(reference, map_normal)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(reference, map_normal), -1.0, 1.0))
                
                qx = rotation_axis[0] * np.sin(angle/2)
                qy = rotation_axis[1] * np.sin(angle/2)
                qz = rotation_axis[2] * np.sin(angle/2)
                qw = np.cos(angle/2)
                
                normal_marker.pose.orientation.x = qx
                normal_marker.pose.orientation.y = qy
                normal_marker.pose.orientation.z = qz
                normal_marker.pose.orientation.w = qw
            
            # Arrow dimensions
            normal_marker.scale.x = 0.2
            normal_marker.scale.y = 0.02
            normal_marker.scale.z = 0.02
            
            # Blue color for normal
            normal_marker.color.r = 0.0
            normal_marker.color.g = 0.0
            normal_marker.color.b = 1.0
            normal_marker.color.a = 0.8
            
            # Set lifetime
            if lifetime > 0:
                normal_marker.lifetime.sec = int(lifetime)
                normal_marker.lifetime.nanosec = int((lifetime % 1) * 1e9)
            
            marker_array.markers.append(normal_marker)
            
            # Text marker for bird
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "bird_labels"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = map_position[0]
            text_marker.pose.position.y = map_position[1]
            text_marker.pose.position.z = map_position[2] + 0.15
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.1
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.8
            text_marker.text = f"Bird {marker_id}"
            
            # Set lifetime
            if lifetime > 0:
                text_marker.lifetime.sec = int(lifetime)
                text_marker.lifetime.nanosec = int((lifetime % 1) * 1e9)
            
            marker_array.markers.append(text_marker)
            
            # Publish markers
            self.marker_publisher.publish(marker_array)
            
            if is_new_bird:
                self.get_logger().info(f"Published new bird marker {marker_id}")
            else:
                self.get_logger().info(f"Updated bird marker {marker_id}")
                
        except Exception as e:
            self.get_logger().error(f"Error publishing bird markers: {e}")

def main(args=None):
    rclpy.init(args=args)
    bird_detector = BirdDetector()
    
    try:
        rclpy.spin(bird_detector)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
        
        # Clean up the node
        bird_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()