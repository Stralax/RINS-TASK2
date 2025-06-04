#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import os
import time

from sklearn.decomposition import PCA
import tf2_geometry_msgs as tfg
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import tf_transformations

from sensor_msgs_py import point_cloud2 as pc2
from rclpy.duration import Duration
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Vector3Stamped, PoseArray, Pose
from std_msgs.msg import String

from dataclasses import dataclass
from PIL import Image as PILImage

@dataclass
class BirdTracker:
    position: np.ndarray  # 3D position in map frame
    normal: np.ndarray    # Normal vector
    last_seen: float      # Timestamp
    marker_id: int        # Assigned marker ID
    bird_class: str = ""  # Classified bird species
    confidence: float = 0.0  # Classification confidence

# Bird classifier model from bird_classifier.py
class BirdClassifierModel(nn.Module):
    def __init__(self, num_classes):
        super(BirdClassifierModel, self).__init__()
        # Use ResNet50 as the backbone
        self.model = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

class DetectClassifyBirds(Node):
    def __init__(self):
        super().__init__('detect_classify_birds_node')
        
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
                ('custom_model', False),  # Set to True if using a custom bird model
                ('classifier_model_path', 'models/bird_classifier.pth'),  # Path to bird classifier model
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
        self.classifier_model_path = self.get_parameter('classifier_model_path').get_parameter_value().string_value
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Load YOLO model for detection
        self.get_logger().info(f"Loading YOLO model from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info("YOLO model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise
        
        # Load bird classifier model
        self.get_logger().info(f"Loading bird classifier from {self.classifier_model_path}...")
        try:
            # Load class names first
            class_names_file = os.path.splitext(self.classifier_model_path)[0] + '_classes.txt'
            if os.path.exists(class_names_file):
                with open(class_names_file, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                self.get_logger().info(f"Loaded {len(self.class_names)} bird classes")
            else:
                self.get_logger().error(f"Class names file not found: {class_names_file}")
                self.class_names = ['Unknown']
            
            # Initialize classifier model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.classifier = BirdClassifierModel(len(self.class_names))
            self.classifier.load_state_dict(torch.load(self.classifier_model_path, map_location=self.device))
            self.classifier.eval()
            self.classifier.to(self.device)
            
            # Image preprocessing for classifier
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.get_logger().info("Bird classifier loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load bird classifier: {e}")
            self.classifier = None
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_id = 0
        self.marker_lifetime = 0.0  # Lifetime for markers in seconds (0 = forever)

        # Bird tracking
        self.tracked_birds = []  # List of BirdTracker objects
        self.bird_distance_threshold = 4.0  # Distance threshold to consider it's the same bird (meters)

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
            '/top_camera/rgb/preview/depth/points',
            self.pointcloud_callback,
            qos_profile)
        
        # Add marker publisher
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/bird_markers',
            qos_profile)
            
        # Add classification result publisher
        self.classification_publisher = self.create_publisher(
            String,
            '/bird_classifier/result',
            qos_profile)
        
        # Store the latest point cloud data
        self.latest_pointcloud = None
        
        # Statistics
        self.detections_count = 0
        self.last_detection_time = time.time() - 10  # Initialize to avoid publishing immediately
        self.min_detection_interval = 5.0  # Minimum seconds between detections
        
        self.get_logger().info(f"Bird detector and classifier initialized. Listening on {self.input_topic}")
    
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
            vector_stamped.header.frame_id = "map"  # Adjust frame ID as needed
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

    def classify_bird_image(self, bird_img):
        """Classify a bird image using the loaded model"""
        if self.classifier is None:
            return "Unknown", 0.0
            
        try:
            # Convert OpenCV image (BGR) to PIL image (RGB)
            pil_image = PILImage.fromarray(cv2.cvtColor(bird_img, cv2.COLOR_BGR2RGB))
            
            # Preprocess the image
            input_tensor = self.transform(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Classify the image
            with torch.no_grad():
                output = self.classifier(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Get top prediction
                top_prob, top_class = torch.max(probabilities, 0)
                predicted_class = self.class_names[top_class.item()]
                confidence = top_prob.item()
                
                return predicted_class, confidence
                
        except Exception as e:
            self.get_logger().error(f"Error classifying bird: {e}")
            return "Unknown", 0.0

    def image_callback(self, msg):
        """Process incoming images to detect and classify birds"""
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
                    # Use color based on confidence (red if < 0.9, green if >= 0.9)
                    color = (0, 255, 0) if conf >= 0.85 else (0, 165, 255)
                    
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
                        
                        # Classify the bird
                        bird_class, class_confidence = self.classify_bird_image(bird_img_with_margin)
                        
                        # Create a labeled image for display
                        display_img = bird_img_with_margin.copy()
                        label = f"{bird_class}: {class_confidence:.2f}"
                        
                        # Add label to the image
                        cv2.putText(
                            display_img,
                            label,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0) if class_confidence > 0.7 else (0, 165, 255),
                            2
                        )
                        
                        # Display the cropped bird image with classification
                        cv2.imshow("Detected Bird", display_img)
                        cv2.waitKey(1)  # Update display
                        
                        try:
                            # Convert the cropped image back to ROS Image message
                            bird_msg = self.bridge.cv2_to_imgmsg(display_img, "bgr8")
                            bird_msg.header = msg.header  # Keep original header
                            
                            # Publish the cropped bird image
                            self.bird_image_publisher.publish(bird_msg)
                            
                            # Publish classification result
                            result_msg = String()
                            result_msg.data = f"{bird_class}:{class_confidence:.4f}"
                            self.classification_publisher.publish(result_msg)
                            
                            self.detections_count += 1
                            self.last_detection_time = current_time
                            bird_detected = True
                            
                            # Publish bird markers if position and normal were calculated successfully
                            if position_3d is not None and normal is not None:
                                self.publish_bird_marker(position_3d, normal, conf, bird_class, class_confidence)
                            
                            self.get_logger().info(f"Bird detected: {bird_class} ({class_confidence:.2f})! Total: {self.detections_count}")
                            
                            # Break after publishing first detection to avoid flooding
                            break
                        except CvBridgeError as e:
                            self.get_logger().error(f"Error converting bird crop to ROS message: {e}")
                    elif conf >= 0.85:
                        # If confidence is high but we're still in cooldown period
                        self.get_logger().debug(f"High-confidence bird detected ({conf:.2f}), but still in cooldown period")
            
            # Add classification info to visualization image
            if self.tracked_birds:
                recent_bird = max(self.tracked_birds, key=lambda b: b.last_seen)
                cv2.putText(
                    viz_image,
                    f"{recent_bird.bird_class} ({recent_bird.confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Display the visualization image
            cv2.imshow("Bird Detection & Classification", viz_image)
            cv2.waitKey(1)  # Update display
            
            # Publish visualization image if enabled
            if self.publish_viz:
                try:
                    viz_msg = self.bridge.cv2_to_imgmsg(viz_image, "bgr8")
                    viz_msg.header = msg.header
                    self.detection_viz_publisher.publish(viz_msg)
                except CvBridgeError as e:
                    self.get_logger().error(f"Error converting visualization image: {e}")
        
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

    def publish_bird_marker(self, position_3d, normal, detection_conf, bird_class="Unknown", class_conf=0.0):
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
                    self.tracked_birds[i].bird_class = bird_class
                    self.tracked_birds[i].confidence = class_conf
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
                    marker_id=marker_id,
                    bird_class=bird_class,
                    confidence=class_conf
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
            scale = 0.1 + (detection_conf * 0.1)
            position_marker.scale.x = position_marker.scale.y = position_marker.scale.z = scale
            
            # Color: Green for high confidence, more yellow for lower
            position_marker.color.r = 1.0 - detection_conf  # Less red for higher confidence
            position_marker.color.g = 1.0
            position_marker.color.b = 0.0
            position_marker.color.a = 0.8
            
            # Set lifetime (0 = forever)
            lifetime = 0.0  
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
            
            # Text marker for bird with species info - MODIFIED
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
            text_marker.color.r = 0.0
            text_marker.color.g = 0.0
            text_marker.color.b = 0.0
            text_marker.color.a = 0.8
            
            # CHANGED: Only show bird class name, not ID
            if class_conf > 0.35: # 0.45
                text_marker.text = bird_class  # Just the bird class name
            else:
                text_marker.text = "Unknown Bird"
            
            # Set lifetime
            if lifetime > 0:
                text_marker.lifetime.sec = int(lifetime)
                text_marker.lifetime.nanosec = int((lifetime % 1) * 1e9)
            
            marker_array.markers.append(text_marker)
            
            # Publish markers
            self.marker_publisher.publish(marker_array)
            
            if is_new_bird:
                self.get_logger().info(f"Published new bird marker: {bird_class}")
            else:
                self.get_logger().info(f"Updated bird marker: {bird_class}")
                
        except Exception as e:
            self.get_logger().error(f"Error publishing bird markers: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DetectClassifyBirds()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
        
        # Clean up the node
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()