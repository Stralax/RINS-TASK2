#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Vector3Stamped, PoseArray, Pose

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import tf2_geometry_msgs as tfg
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import tf_transformations

from collections import deque
from dataclasses import dataclass
import time
import json
import os
import subprocess
from datetime import datetime
# Add imports for gender detection
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch

@dataclass
class RGBDetection:
    stamp: rclpy.time.Time
    faces: list  # List of (cx, cy, width, height) for detected faces
    image: np.ndarray = None  # Optional, for visualization

@dataclass
class PointCloudData:
    stamp: rclpy.time.Time
    data: PointCloud2

class FaceData:
    def __init__(self, face_id, position, normal, is_new=True):
        self.face_id = face_id
        self.position = position  # numpy array [x, y, z]
        self.normal = normal      # numpy array [x, y, z]
        self.is_new = is_new      # Flag to track if this is a newly detected face
        self.last_seen = time.time()
        # Add gender information
        self.gender = "Unknown"
        self.gender_confidence = 0.0
        
    def update_position(self, new_position, new_normal, smoothing_factor=0.3):
        """Update position with smoothing and limit updates"""
        if not hasattr(self, 'update_count'):
            self.update_count = 0
        
        if self.update_count < 10:
            self.position = (1 - smoothing_factor) * self.position + smoothing_factor * new_position
            self.normal = (1 - smoothing_factor) * self.normal + smoothing_factor * new_normal
            # Normalize the normal vector
            self.normal = self.normal / np.linalg.norm(self.normal)
            self.last_seen = time.time()
            self.is_new = False
            self.update_count += 1
    
    def update_gender(self, gender, confidence):
        """Update gender information"""
        self.gender = gender
        self.gender_confidence = confidence
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'face_id': int(self.face_id),
            'position': self.position.tolist(),
            'normal': self.normal.tolist(),
            'last_seen': self.last_seen,
            'gender': self.gender,
            'gender_confidence': float(self.gender_confidence)
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create FaceData from dictionary"""
        face = cls(
            face_id=data['face_id'],
            position=np.array(data['position']),
            normal=np.array(data['normal']),
            is_new=False
        )
        face.last_seen = data.get('last_seen', time.time())
        face.gender = data.get('gender', "Unknown")
        face.gender_confidence = data.get('gender_confidence', 0.0)
        return face

class DetectFaces(Node):
    def __init__(self):
        super().__init__('detect_faces')
        
        self.declare_parameters(namespace='', parameters=[
            ('device', ''),
            ('save_file', ''),
            ('save_interval', 5.0),  # Save every 5 seconds
            ('marker_lifetime', 0.0),  # 0 means forever
            ('greeting_text', 'Hello!')
        ])
        
        marker_topic = "/people_marker"
        array_topic = "/people_array"
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.save_file = self.get_parameter('save_file').get_parameter_value().string_value
        self.greeting_text = self.get_parameter('greeting_text').get_parameter_value().string_value
        
        if not self.save_file:
            # Default save file in home directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_file = os.path.expanduser(f"~/colcon_ws/src/dis_tutorial3/detected_faces_{timestamp}.json")
        
        self.save_interval = self.get_parameter('save_interval').get_parameter_value().double_value
        self.marker_lifetime = self.get_parameter('marker_lifetime').get_parameter_value().double_value
        
        self.detection_color = (0, 0, 255)
        
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")

        self.get_logger().info("Loading gender detection model...")
        try:
            model_name = "dima806/fairface_gender_image_detection"
            self.gender_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.gender_model = AutoModelForImageClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.get_logger().info(f"Using device: {self.device}")
            self.gender_model.to(self.device)
            self.gender_model.eval()
            self.gender_labels = self.gender_model.config.id2label if hasattr(self.gender_model.config, "id2label") else {0: "Male", 1: "Female"}
            self.get_logger().info("Gender detection model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load gender detection model: {e}")
            # Continue without gender detection
            self.gender_model = None
        
        # Message synchronization
        self.rgb_buffer = deque(maxlen=30)  # Store recent RGB detections
        self.pointcloud_buffer = deque(maxlen=30)  # Store recent point clouds
        self.max_time_diff = 0.1  # Maximum time difference in seconds to consider messages synchronous
        
        self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", 
                                                     self.rgb_callback, qos_profile_sensor_data)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", 
                                                      self.pointcloud_callback, qos_profile_sensor_data)
        
        self.marker_pub = self.create_publisher(MarkerArray, array_topic, QoSReliabilityPolicy.BEST_EFFORT)
        # self.marker_pub = self.create_publisher(MarkerArray, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
        
        self.positions_pub = self.create_publisher(
            PoseArray, 
            '/detected_people_poses', 
            QoSReliabilityPolicy.BEST_EFFORT
        )

        self.transform_pub = self.create_publisher(
            Float32MultiArray, 
            '/person_transform_data', 
            QoSReliabilityPolicy.BEST_EFFORT    
        )


        # TF2 buffer and listener for transformation
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters
        self.neighborhood_radius = 1.0  # meters for point cloud neighborhood
        self.clustering_distance = 0.8  # meters for DBSCAN clustering
        self.face_distance_threshold = 2.0  # meters for determining if a face is new
        self.min_points_for_face = 5  # Minimum points to represent a valid face
        self.next_face_id = 0  # Counter for face IDs

        # Store state
        self.persistent_faces = {}  # Dictionary of face_id -> FaceData
        self.last_save_time = time.time()
        
        # Create a timer to publish all markers periodically
        self.marker_timer = self.create_timer(0.5, self.publish_all_markers)
        
        # Path to the TTS script
        self.tts_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speak.py")
        
        # Load any previously saved faces
        self.load_faces()
        
        self.get_logger().info(f"Node initialized! Detecting faces and publishing markers to {array_topic}.")
        self.get_logger().info(f"Saving detected faces to {self.save_file}")
        self.get_logger().info(f"Will say '{self.greeting_text}' when a new face is detected")

    def detect_gender(self, face_image):
        """Detect gender from face image"""
        if self.gender_model is None:
            return "Unknown", 0.0
            
        try:
            # Convert OpenCV image (BGR) to PIL image (RGB)
            pil_image = PILImage.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            
            # Process with gender detection model
            inputs = self.gender_feature_extractor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.gender_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_id = probs.argmax()
                pred_label = self.gender_labels[pred_id]
                confidence = float(probs[pred_id])
            
            self.get_logger().debug(f"Gender detection: {pred_label} with confidence {confidence:.4f}")
            return pred_label, confidence
        except Exception as e:
            self.get_logger().error(f"Error in gender detection: {e}")
            return "Unknown", 0.0
    
    def say_greeting(self):
        """Use the text-to-speech script to say greeting"""
        try:
            # Print to terminal
            print(f"{self.greeting_text}")
            
            # Run the TTS script
            subprocess.Popen(["python3", self.tts_script_path, self.greeting_text])
            self.get_logger().info(f"Speaking greeting: {self.greeting_text}")
        except Exception as e:
            self.get_logger().error(f"Error running TTS script: {e}")
    
    def load_faces(self):
        """Load previously saved faces from file"""
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, 'r') as f:
                    data = json.load(f)
                    for face_dict in data:
                        face = FaceData.from_dict(face_dict)
                        self.persistent_faces[face.face_id] = face
                        # Update next_face_id to be higher than any loaded ID
                        self.next_face_id = max(self.next_face_id, face.face_id + 1)
                        
                self.get_logger().info(f"Loaded {len(self.persistent_faces)} faces from {self.save_file}")
            except Exception as e:
                self.get_logger().error(f"Error loading faces from {self.save_file}: {e}")
    
    def save_faces(self):
        """Save detected faces to file"""
        try:
            # Convert faces to list of dictionaries
            faces_data = [face.to_dict() for face in self.persistent_faces.values()]
            
            # Make sure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.save_file)), exist_ok=True)
            
            with open(self.save_file, 'w') as f:
                json.dump(faces_data, f, indent=2)
                
            self.last_save_time = time.time()
            self.get_logger().debug(f"Saved {len(faces_data)} faces to {self.save_file}")
        except Exception as e:
            self.get_logger().error(f"Error saving faces to {self.save_file}: {e}")

    def publish_transform_data(self, center, normal):
        """Publish the transformed center and normal as a Float32MultiArray"""
        if center is None or normal is None:
            return
        
        # Create a Float32MultiArray message
        transform_data = Float32MultiArray()
        
        # Set up the dimensions
        transform_data.layout.dim = [
            MultiArrayDimension(label="data_type", size=2, stride=6),  # 2 data types (center and normal)
            MultiArrayDimension(label="coordinates", size=3, stride=3)  # 3 values per data type (x, y, z)
        ]
        
        # Set the data (center first, then normal)
        transform_data.data = [
            float(center[0]), float(center[1]), float(center[2]),  # center coordinates
            float(normal[0]), float(normal[1]), float(normal[2])   # normal vector
        ]
        
        # Publish the data
        self.transform_pub.publish(transform_data)
        self.get_logger().debug(f"Published transform data: center={center}, normal={normal}")

    def publish_people_positions(self):
        """Publish positions of detected people as a PoseArray"""
        if not self.persistent_faces:
            return
            
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        for face_id, face_data in self.persistent_faces.items():
            # Create a pose for the actual person position
            person_pose = Pose()
            person_pose.position.x = face_data.position[0]
            person_pose.position.y = face_data.position[1]
            person_pose.position.z = face_data.position[2]
            
            # Calculate quaternion from normal vector
            normal = face_data.normal
            reference = np.array([1.0, 0.0, 0.0])
            
            if np.allclose(normal, reference) or np.allclose(normal, -reference):
                rotation_axis = np.array([0.0, 0.0, 1.0])
                angle = 0.0 if np.allclose(normal, reference) else np.pi
            else:
                rotation_axis = np.cross(reference, normal)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(reference, normal))
            
            qx = rotation_axis[0] * np.sin(angle/2)
            qy = rotation_axis[1] * np.sin(angle/2)
            qz = rotation_axis[2] * np.sin(angle/2)
            qw = np.cos(angle/2)
            
            # Add orientation to the pose
            person_pose.orientation.x = qx
            person_pose.orientation.y = qy
            person_pose.orientation.z = qz
            person_pose.orientation.w = qw
            
            # Send only the person's position - let robot_commander calculate approach
            pose_array.poses.append(person_pose)
            
            # self.get_logger().info(f"Publishing person at position={face_data.position[:2]}, normal={normal[:2]}")
        
        # Publish the array
        self.positions_pub.publish(pose_array)
        self.get_logger().debug(f"Published {len(pose_array.poses)} people positions")
        
        
    def rgb_callback(self, data):
        """Process RGB image to detect faces/people"""
        try:
            stamp = data.header.stamp
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Get image height to determine lower half
            img_height = cv_image.shape[0]
            lower_half_y = img_height // 2
            
            # Detect people (class 0 in COCO)
            results = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)
            
            faces = []
            detect_image = cv_image.copy()
            
            # Draw a horizontal line showing the lower half boundary
            # cv2.line(detect_image, (0, lower_half_y), (cv_image.shape[1], lower_half_y), 
            #         (0, 255, 0), 2)
            
            for det in results:
                bbox = det.boxes.xyxy
                if bbox.nelement() == 0:
                    continue
                
                self.get_logger().debug("Person detected")
                
                for i in range(len(bbox)):
                    box = bbox[i]
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    width = int(box[2] - box[0])
                    height = int(box[3] - box[1])
                    
                    # Only consider detections in the lower half of the image
                    if cy >= lower_half_y:
                        # Extract face image for gender detection
                        face_img = cv_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        gender, confidence = self.detect_gender(face_img)
                        
                        # Store detection with size and gender information
                        faces.append((cx, cy, width, height, gender, confidence))
                        
                        # Visualize detection
                        detect_image = cv2.rectangle(detect_image, 
                                            (int(box[0]), int(box[1])), 
                                            (int(box[2]), int(box[3])), 
                                            self.detection_color, 2)
                        detect_image = cv2.circle(detect_image, (cx, cy), 5, self.detection_color, -1)
                        
                        # Add gender and confidence text
                        gender_text = f"{gender}: {confidence:.2f}"
                        text_pos = (int(box[0]), int(box[1]) - 10) if int(box[1]) > 30 else (int(box[0]), int(box[3]) + 20)
                        cv2.putText(detect_image, gender_text, text_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    else:
                        # Draw rejected detection with different color
                        detect_image = cv2.rectangle(detect_image, 
                                            (int(box[0]), int(box[1])), 
                                            (int(box[2]), int(box[3])), 
                                            (128, 128, 128), 1)  # Gray color for rejected detections
            
            # Store detection in buffer
            self.rgb_buffer.append(RGBDetection(stamp=stamp, faces=faces, image=detect_image))
            
            # Try to find a matching point cloud in the buffer
            self.process_synchronized_data()
            
            cv2.imshow("Detections", detect_image)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                self.get_logger().info("Exiting on user request")
                self.save_faces()  # Make sure to save before exiting
                exit()
                
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    def pointcloud_callback(self, data):
        """Store point cloud data in buffer"""
        self.pointcloud_buffer.append(PointCloudData(stamp=data.header.stamp, data=data))
        
        # Try to find a matching RGB detection in the buffer
        self.process_synchronized_data()
    
    def process_synchronized_data(self):
        """Process pairs of time-synchronized RGB and point cloud data"""
        if not self.rgb_buffer or not self.pointcloud_buffer:
            return
        
        # Find the best match by time difference
        best_match = None
        best_time_diff = float('inf')
        matched_rgb_idx = -1
        matched_pc_idx = -1
        
        for rgb_idx, rgb_detection in enumerate(self.rgb_buffer):
            for pc_idx, pc_data in enumerate(self.pointcloud_buffer):
                # Convert stamps to seconds
                rgb_time = rgb_detection.stamp.sec + rgb_detection.stamp.nanosec / 1e9
                pc_time = pc_data.stamp.sec + pc_data.stamp.nanosec / 1e9
                time_diff = abs(rgb_time - pc_time)
                
                if time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_match = (rgb_detection, pc_data)
                    matched_rgb_idx = rgb_idx
                    matched_pc_idx = pc_idx
        
        # Process if the time difference is acceptable
        if best_match and best_time_diff <= self.max_time_diff:
            rgb_detection, pc_data = best_match
            
            # Remove processed items from buffers (and all older items)
            for _ in range(matched_rgb_idx + 1):
                if self.rgb_buffer:
                    self.rgb_buffer.popleft()
            
            for _ in range(matched_pc_idx + 1):
                if self.pointcloud_buffer:
                    self.pointcloud_buffer.popleft()
            
            # Process the synchronized data
            if rgb_detection.faces:
                self.process_detection_with_pointcloud(rgb_detection.faces, pc_data.data)
            
            # Check if it's time to save faces
            if time.time() - self.last_save_time > self.save_interval:
                self.save_faces()
    
    def process_detection_with_pointcloud(self, faces, pointcloud_data):
        """Process synchronized face detection and point cloud data"""
        if not faces:
            return
            
        # Get a timestamp that's close to the actual data (either one would work since they're synchronized)
        pc_stamp = pointcloud_data.header.stamp
        
        # Convert the point cloud to a numpy array
        try:
            pc_array = pc2.read_points_numpy(pointcloud_data, field_names=("x", "y", "z")).reshape((pointcloud_data.height, pointcloud_data.width, 3))
        except Exception as e:
            self.get_logger().error(f"Error reading point cloud: {e}")
            return
        
        # Process each detected face
        for face_idx, face_data in enumerate(faces):
            # Unpack face data, which now includes gender information
            if len(face_data) >= 6:  # If gender info is included
                cx, cy, width, height, gender, gender_confidence = face_data
            else:  # Backward compatibility
                cx, cy, width, height = face_data
                gender, gender_confidence = "Unknown", 0.0

            # Extract a region of interest (ROI) from the point cloud
            # Use a percentage of the bbox size to get a good representation
            roi_width = max(10, int(width * 0.6))
            roi_height = max(10, int(height * 0.4))  # Focus more on upper part for faces
            
            # Define the ROI boundaries ensuring they're within image bounds
            x_start = max(0, cx - roi_width//2)
            x_end = min(pointcloud_data.width-1, cx + roi_width//2)
            y_start = max(0, cy - roi_height//2)
            y_end = min(pointcloud_data.height-1, cy + roi_height//2)
            
            # Extract 3D points from the ROI
            roi_points = []
            for y in range(y_start, y_end+1):
                for x in range(x_start, x_end+1):
                    point = pc_array[y, x, :]
                    if np.isfinite(point).all() and not np.isnan(point).any():
                        roi_points.append(point)
            
            if len(roi_points) < self.min_points_for_face:
                self.get_logger().warn(f"Not enough valid points for face {face_idx}, {len(roi_points)} points found")
                continue
                
            # Convert to numpy array
            roi_points = np.array(roi_points)
            
            # Remove outliers using DBSCAN clustering
            clustering = DBSCAN(eps=self.neighborhood_radius, min_samples=5).fit(roi_points)
            labels = clustering.labels_
            
            # Use the largest cluster
            if len(set(labels)) <= 1 and -1 in labels:  # Only noise points
                self.get_logger().warn(f"No valid clusters found for face {face_idx}")
                continue
                
            unique_labels = np.unique(labels)
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels if label != -1}
            
            if not cluster_sizes:  # No valid clusters
                continue
                
            largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
            cluster_points = roi_points[labels == largest_cluster_label]
            
            # Calculate center point (median is more robust than mean)
            center_3d = np.median(cluster_points, axis=0)
            
            # Calculate normal vector using PCA
            if len(cluster_points) >= 3:
                pca = PCA(n_components=3)
                pca.fit(cluster_points)
                # Normal is perpendicular to the first two principal components
                normal = np.cross(pca.components_[0], pca.components_[1])
                # normal = normal / np.linalg.norm(normal)
                
                # Ensure normal is pointing outward (towards camera), not inward
                if normal[2] < 0:  # Z is typically depth
                    normal = -normal
            else:
                # Default forward-facing normal if we can't compute PCA
                normal = np.array([0, 0, 1])
            
            # Transform to map frame - using latest available transform instead of exact timestamp
            transformed_center = self.transform_point_to_map(center_3d)
            transformed_normal = self.transform_vector_to_map(normal)
            
            if transformed_center is None or transformed_normal is None:
                self.get_logger().warn(f"Transformation failed for face {face_idx}")
                continue
            
            # Check if this face has been seen before
            matched_face_id = None
            for face_id, face_data in self.persistent_faces.items():
                distance = np.linalg.norm(transformed_center - face_data.position)
                if distance < self.face_distance_threshold:
                    matched_face_id = face_id
                    # Update the position of the seen face
                    face_data.update_position(transformed_center, transformed_normal)
                    break
            
            # If no match was found, add a new face
            if matched_face_id is None:
                new_face_id = self.next_face_id
                self.next_face_id += 1
                new_face = FaceData(
                    face_id=new_face_id, 
                    position=transformed_center, 
                    normal=transformed_normal,
                    is_new=True
                )
                # Update gender information
                new_face.update_gender(gender, gender_confidence)
                self.persistent_faces[new_face_id] = new_face
                
                self.get_logger().info(f"New face detected! ID: {new_face_id}, Position: {transformed_center}, Gender: {gender} ({gender_confidence:.2f})")

                self.publish_transform_data(transformed_center, transformed_normal)
                # Say greeting when new face is detected
                # self.say_greeting()
            else:
                # Update existing face
                face_data = self.persistent_faces[matched_face_id]
                face_data.update_position(transformed_center, transformed_normal)
                # Update gender if confidence is higher
                if gender_confidence > face_data.gender_confidence:
                    face_data.update_gender(gender, gender_confidence)
                    self.get_logger().debug(f"Updated gender for face ID {matched_face_id}: {gender} ({gender_confidence:.2f})")
                
                self.get_logger().debug(f"Updated face ID: {matched_face_id}, Position: {transformed_center}")
    
    def transform_point_to_map(self, point):
        """Transform a point from camera frame to map frame using latest transform"""
        try:
            # Create PointStamped object
            point_stamped = PointStamped()
            point_stamped.header.frame_id = "base_link"
            point_stamped.header.stamp = self.get_clock().now().to_msg()  # Use current time
            point_stamped.point.x = float(point[0])
            point_stamped.point.y = float(point[1])
            point_stamped.point.z = float(point[2])
            
            # Get latest transform
            transform = self.tf_buffer.lookup_transform(
                "map", 
                "base_link",
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
        """Transform a direction vector from camera frame to map frame using latest transform"""
        try:
            # Create Vector3Stamped for the normal vector
            vector_stamped = Vector3Stamped()
            vector_stamped.header.frame_id = "base_link"
            vector_stamped.header.stamp = self.get_clock().now().to_msg()  # Use current time
            vector_stamped.vector.x = float(vector[0])
            vector_stamped.vector.y = float(vector[1])
            vector_stamped.vector.z = float(vector[2])
            
            # Get latest transform
            transform = self.tf_buffer.lookup_transform(
                "map", 
                "base_link",
                rclpy.time.Time(),  # Get latest transform
                rclpy.duration.Duration(seconds=1.0)
            )
            
            # Transform the vector
            transformed_vector = tfg.do_transform_vector3(vector_stamped, transform)
            
            # Normalize the transformed vector
            result = np.array([
                transformed_vector.vector.x, 
                transformed_vector.vector.y, 
                transformed_vector.vector.z
            ])
            # result = result / np.linalg.norm(result)
            
            return result
            
        except TransformException as e:
            self.get_logger().warn(f"Could not transform vector: {e}")
            return None
    
    def publish_all_markers(self):
        """Publish markers for all stored faces"""
        if not self.persistent_faces:
            return
            
        marker_array = MarkerArray()
        
        for face_id, face_data in self.persistent_faces.items():
            # Face position marker (sphere)
            face_marker = Marker()
            face_marker.header.frame_id = "map"
            face_marker.header.stamp = self.get_clock().now().to_msg()
            face_marker.ns = "face"  # Use "face" namespace for all face markers
            face_marker.id = face_id
            face_marker.type = Marker.SPHERE
            face_marker.action = Marker.ADD
            face_marker.pose.position.x = face_data.position[0]
            face_marker.pose.position.y = face_data.position[1]
            face_marker.pose.position.z = face_data.position[2]
            face_marker.pose.orientation.w = 1.0
            face_marker.scale.x = face_marker.scale.y = face_marker.scale.z = 0.15
            
            # Different colors based on gender
            if face_data.gender == "Female":
                face_marker.color.r = 1.0
                face_marker.color.g = 0.0
                face_marker.color.b = 1.0  # Purple for female
            elif face_data.gender == "Male":
                face_marker.color.r = 0.0
                face_marker.color.g = 0.0
                face_marker.color.b = 1.0  # Blue for male
            else:  # Unknown
                face_marker.color.r = 0.7
                face_marker.color.g = 0.7
                face_marker.color.b = 0.7  # Gray for unknown
            face_marker.color.a = 0.8
            
            # Store gender as a string in the marker description
            face_marker.text = face_data.gender
            
            # Set lifetime if configured
            if self.marker_lifetime > 0:
                face_marker.lifetime.sec = int(self.marker_lifetime)
                face_marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
            
            marker_array.markers.append(face_marker)
            
            # ID text marker - include gender information
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "face_ids"
            text_marker.id = face_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = face_data.position[0]
            text_marker.pose.position.y = face_data.position[1]
            text_marker.pose.position.z = face_data.position[2] + 0.2  # Above the sphere
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.1  # Text size
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.8
            text_marker.text = f"ID: {face_id} ({face_data.gender})"
            
            if self.marker_lifetime > 0:
                text_marker.lifetime.sec = int(self.marker_lifetime)
                text_marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
            
            marker_array.markers.append(text_marker)

            
            # Direction marker (arrow)
            direction_marker = Marker()
            direction_marker.header.frame_id = "map"
            direction_marker.header.stamp = self.get_clock().now().to_msg()
            direction_marker.ns = "face_directions"
            direction_marker.id = face_id
            direction_marker.type = Marker.ARROW
            direction_marker.action = Marker.ADD
            direction_marker.pose.position.x = face_data.position[0]
            direction_marker.pose.position.y = face_data.position[1]
            direction_marker.pose.position.z = face_data.position[2]
            
            # Create quaternion from normal vector to reference vector [1,0,0]
            normal = face_data.normal
            reference = np.array([1.0, 0.0, 0.0])
            # Handle the case where vectors are parallel
            if np.allclose(normal, reference) or np.allclose(normal, -reference):
                # If parallel, use a different reference
                rotation_axis = np.array([0.0, 0.0, 1.0])
                angle = 0.0 if np.allclose(normal, reference) else np.pi
            else:
                rotation_axis = np.cross(reference, normal)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(reference, normal))
            
            # Convert to quaternion
            qx = rotation_axis[0] * np.sin(angle/2)
            qy = rotation_axis[1] * np.sin(angle/2)
            qz = rotation_axis[2] * np.sin(angle/2)
            qw = np.cos(angle/2)
            
            direction_marker.pose.orientation.x = qx
            direction_marker.pose.orientation.y = qy
            direction_marker.pose.orientation.z = qz
            direction_marker.pose.orientation.w = qw
            
            direction_marker.scale.x = 0.4  # Length
            direction_marker.scale.y = 0.05  # Width
            direction_marker.scale.z = 0.05  # Height
            
            direction_marker.color.r = 0.0
            direction_marker.color.g = 0.0
            direction_marker.color.b = 1.0
            direction_marker.color.a = 0.8
            
            if self.marker_lifetime > 0:
                direction_marker.lifetime.sec = int(self.marker_lifetime)
                direction_marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
            
            marker_array.markers.append(direction_marker)
        
        # Publish the marker array
        self.marker_pub.publish(marker_array)
        self.publish_people_positions()
        self.get_logger().debug(f"Published markers for {len(self.persistent_faces)} faces")

def main():
    print('Face detection node starting.')
    rclpy.init(args=None)
    node = DetectFaces()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()