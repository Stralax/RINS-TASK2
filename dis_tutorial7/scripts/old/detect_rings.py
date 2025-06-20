#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs_py import point_cloud2 as pc2
import subprocess
import os
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import tf2_geometry_msgs as tfg
from geometry_msgs.msg import PointStamped, Vector3Stamped, Quaternion
import rclpy.duration
import time
from collections import deque
from dataclasses import dataclass

from sklearn.decomposition import PCA
import tf_transformations

@dataclass
class RingData:
    position: np.ndarray  # 3D position in map frame [x, y, z]
    radius: float         # Radius in meters
    color_name: str       # Color name (e.g., "red", "blue", "green", "yellow", "black")
    color_bgr: tuple      # BGR color tuple for visualization
    last_seen: float      # Timestamp when last detected
    announced: bool       # Whether this ring has been announced via speech
    normal1: np.ndarray = None  # First normal direction
    normal2: np.ndarray = None  # Second normal direction (opposite)
    
class RingDetector(Node):
    def __init__(self):
        super().__init__('ring_detector')
        self.bridge = CvBridge()
        
        # Subscriptions
        self.image_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/top_camera/rgb/preview/depth", self.depth_callback, 1)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/top_camera/rgb/preview/depth/points", self.pointcloud_callback, 1)
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, "/ring_markers", 10)
        
        # TF2 buffer and listener for transformation
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Data storage
        self.depth_data = None
        self.depth_width = 0
        self.depth_height = 0
        self.pointcloud_data = None
        self.rings = {}  # Dictionary of detected rings by position hash
        
        # Parameters
        self.marker_lifetime = 5.0  # Marker lifetime in seconds
        self.ring_position_threshold = 0.3  # meters, threshold for considering a ring as the same
        # self.ring_timeout = 30.0  # seconds before removing a ring from tracking
        self.announce_cooldown = 5.0  # minimum seconds between announcing the same ring
        
        # Create timers
        self.marker_timer = self.create_timer(0.5, self.publish_ring_markers)
        # self.cleanup_timer = self.create_timer(5.0, self.cleanup_old_rings)
        
        # Add robot position tracking
        self.robot_position = np.array([0.0, 0.0, 0.0])  # Default position

        # Add TF listener for robot position
        self.tf_timer = self.create_timer(0.5, self.update_robot_position)

        # Path to the TTS script (assuming it's in the same directory as this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tts_script_path = os.path.join(script_dir, "speak.py")
        if not os.path.exists(self.tts_script_path):
            self.get_logger().warn(f"TTS script not found at {self.tts_script_path}. Using default path.")
            self.tts_script_path = os.path.expanduser("~/colcon_ws/src/dis_tutorial3/speak.py")
        
        # Create windows
        cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected Rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
        # Add debug windows
        cv2.namedWindow("Ring Debug", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth Points", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Color Debug", cv2.WINDOW_NORMAL)
        
        self.get_logger().info("Ring detector node initialized. Publishing markers to /ring_markers")

    def depth_callback(self, data):
        try:
            # Convert depth image
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            self.depth_data = depth_image  # Store for ring detection
            self.depth_height, self.depth_width = depth_image.shape

            # Create a copy for visualization
            depth_display = depth_image.copy()
            
            # Replace invalid values (inf and nan)
            depth_display = np.nan_to_num(depth_display, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip the depth values to a reasonable range (e.g., 0 to 3 meters)
            max_depth = 3.0  # meters
            depth_display = np.clip(depth_display, 0, max_depth)
            
            # Normalize to 0-255 range for visualization
            depth_normalized = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Apply colormap for better visualization (TURBO gives better depth perception)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
            
            # Display the depth map
            cv2.imshow("Depth window", depth_colormap)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in depth callback: {e}")

    def pointcloud_callback(self, data):
        """Store the latest point cloud data"""
        self.pointcloud_data = data

    def update_robot_position(self):
        """Update the robot's position in the map frame"""
        try:
            # Get transform from base_link to map
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            
            # Extract translation
            self.robot_position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
        except TransformException:
            # Ignore transform errors
            pass

    def calculate_ring_normal(self, ring_points):
        """Calculate the normal vector of a ring using PCA on the 3D points."""
        if len(ring_points) < 3:
            return None, None
        
        try:
            # Convert to numpy array for PCA
            points_array = np.array(ring_points)
            
            # Apply PCA to find normal vector
            pca = PCA(n_components=3)
            pca.fit(points_array)
            
            # The normal is the eigenvector corresponding to the smallest eigenvalue (3rd component)
            normal = pca.components_[2]
            normal = np.cross(pca.components_[0], pca.components_[1])  # Ensure normal is perpendicular to the plane 
            
            # Calculate the ring center (mean of points)
            center = np.mean(points_array, axis=0)
            
            # We need the normal to point outward from the center of the robot
            # If we have the robot position, use that to orient the normal
            # if hasattr(self, 'robot_position'):
            #     # Vector from robot to ring center
            #     robot_to_ring = center - self.robot_position
            #     # If normal is pointing away from robot, flip it
            #     if np.dot(normal, robot_to_ring) < 0:
            #         normal = -normal
            
            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)
            
            # The second normal is the negative of the first
            normal2 = -normal
            
            self.get_logger().debug(f"Ring normal calculated: {normal}")
            return normal, normal2
        except Exception as e:
            self.get_logger().error(f"Error calculating ring normal: {e}")
            return None, None

    def announce_ring_color(self, color_name):
        """Announce the detected ring color using TTS"""
        try:
            message = f"{color_name} ring detected"
            # Print to terminal
            self.get_logger().info(message)
            
            # Run the TTS script
            subprocess.Popen(["python3", self.tts_script_path, message])
        except Exception as e:
            self.get_logger().error(f"Error running TTS script: {e}")

    def position_hash(self, position):
        """Create a simple hash from a position to use as a ring identifier"""
        return f"{position[0]:.2f}_{position[1]:.2f}_{position[2]:.2f}"

    def transform_point_to_map(self, point_3d):
        """Transform a point from camera frame to map frame"""
        try:
            # Create PointStamped object
            point_stamped = PointStamped()
            point_stamped.header.frame_id = "base_link"
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = float(point_3d[0])
            point_stamped.point.y = float(point_3d[1])
            point_stamped.point.z = float(point_3d[2])
            
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

    def get_point_cloud_position(self, x, y, r):
        """Get 3D position of ring center from point cloud data"""
        if self.pointcloud_data is None:
            return None, None
            
        try:
            # Convert point cloud to numpy array
            pc_array = pc2.read_points_numpy(
                self.pointcloud_data, 
                field_names=("x", "y", "z")
            ).reshape((self.pointcloud_data.height, self.pointcloud_data.width, 3))
            
            # Sample points around the ring at two different radii
            ring_points = []
            num_samples = 16  # Increased for better coverage
            
            # Sample at both 80% (inner edge) and 120% (outer edge) of the radius
            sampling_radii = [0.8, 1.0, 1.2]
            
            for radius_factor in sampling_radii:
                for angle in np.linspace(0, 2*np.pi, num_samples, endpoint=False):
                    px = int(x + r * radius_factor * np.cos(angle))
                    py = int(y + r * radius_factor * np.sin(angle))
                    
                    # Check if point is within image bounds
                    if 0 <= px < self.pointcloud_data.width and 0 <= py < self.pointcloud_data.height:
                        point = pc_array[py, px]
                        if np.isfinite(point).all() and not np.isnan(point).any():
                            ring_points.append(point)
            
            # If we have enough points, compute the median position
            if len(ring_points) >= 9:  # Increased minimum number of points
                ring_position = np.median(np.array(ring_points), axis=0)
                return ring_position, ring_points
                
            return None, None
                
        except Exception as e:
            self.get_logger().error(f"Error extracting point cloud data: {e}")
            return None, None
                
        except Exception as e:
            self.get_logger().error(f"Error extracting point cloud data: {e}")
            return None, None

    def is_hollow_ring(self, x, y, r, depth_map):
        """Check if ring is hollow by comparing center depth with perimeter depth."""
        try:
            # Create debug visualization
            debug_img = cv2.cvtColor(depth_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            points_img = np.zeros_like(debug_img)
            
            # Get center depth and mark it
            center_depth = depth_map[y, x]
            cv2.circle(points_img, (x, y), 2, (0, 0, 255), -1)  # Red dot for center
            
            # Get perimeter depths by sampling points around the circle
            num_points = 8
            perimeter_depths = []
            for angle in np.linspace(0, 2*np.pi, num_points, endpoint=False):
                px = int(x + r * np.cos(angle))
                py = int(y + r * np.sin(angle))
                if 0 <= px < self.depth_width and 0 <= py < self.depth_height:
                    depth = depth_map[py, px]
                    if depth > 0 and not np.isnan(depth):
                        perimeter_depths.append(depth)
                        # Mark perimeter points in green
                        cv2.circle(points_img, (px, py), 2, (0, 255, 0), -1)
            
            if len(perimeter_depths) < 4:
                return False
                    
            perimeter_depth = np.mean(perimeter_depths)
            depth_difference = center_depth - perimeter_depth
            
            # Draw the full circle and depth values
            cv2.circle(debug_img, (x, y), r, (255, 255, 255), 1)
            cv2.putText(debug_img, 
                        f"Center: {center_depth:.2f}m", 
                        (x+10, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 255), 
                        1)
            cv2.putText(debug_img, 
                        f"Perim: {perimeter_depth:.2f}m", 
                        (x+10, y+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        1)
            cv2.putText(debug_img, 
                        f"Diff: {depth_difference:.2f}m", 
                        (x+10, y+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1)
            
            # Show debug windows
            cv2.imshow("Ring Debug", debug_img)
            cv2.imshow("Depth Points", points_img)
            cv2.waitKey(1)
            
            min_depth_diff = 0.1
            return depth_difference > min_depth_diff
                
        except Exception as e:
            self.get_logger().error(f"Error checking ring depth: {e}")
            return False

    def get_ring_color(self, frame, x, y, r):
        # Create multiple masks with different thicknesses to sample more effectively
        inner_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        outer_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Use two masks to better capture the ring's color
        cv2.circle(inner_mask, (x, y), int(r-1), 255, 3)  # Inner part of ring
        cv2.circle(outer_mask, (x, y), int(r+1), 255, 3)  # Outer part of ring
        combined_mask = cv2.bitwise_or(inner_mask, outer_mask)
        
        # Create debug visualization
        debug_color = frame.copy()
        cv2.circle(debug_color, (x, y), int(r), (0, 255, 255), 2)
        
        # Sample colors using the mask
        mean_bgr = cv2.mean(frame, mask=combined_mask)[:3]
        b, g, r = mean_bgr
        
        # Calculate RGB differences right away so they're available for all detection methods
        rg_diff = abs(r - g)
        rb_diff = abs(r - b)
        gb_diff = abs(g - b)
        
        # Convert to multiple color spaces for better detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Get color values in different spaces
        mean_hsv = cv2.mean(hsv_frame, mask=combined_mask)[:3]
        mean_lab = cv2.mean(lab_frame, mask=combined_mask)[:3]
        
        h, s, v = mean_hsv
        l, a, b_val = mean_lab
        
        # Show debug info with multiple color spaces
        cv2.putText(debug_color, f"HSV: {h:.0f},{s:.0f},{v:.0f}", 
                    (int(x + r + 10), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(debug_color, f"BGR: {b:.0f},{g:.0f},{r:.0f}", 
                    (int(x + r + 10), int(y+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(debug_color, f"LAB: {l:.0f},{a:.0f},{b_val:.0f}", 
                    (int(x + r + 10), int(y+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Special debug for potentially black rings
        if max(r, g, b) < 100:
            cv2.putText(debug_color, "Potentially black ring", 
                        (int(x + r + 10), int(y+60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        
        # Show the mask and debug info
        cv2.imshow("Color Debug", debug_color)
        
        # Decision making using multiple color spaces
        # More aggressive black detection with multiple approaches
        # Approach 1: Low value in HSV
        if v < 50:  # Increased threshold from adaptive_v_threshold * 1.5
            return "black", (0, 0, 0)

        # Approach 2: Low saturation and value together
        if s < 40 and v < 80:
            return "black", (0, 0, 0)

        # Approach 3: All RGB channels are low and similar
        if max(r, g, b) < 80 and max(rg_diff, rb_diff, gb_diff) < 20:
            return "black", (0, 0, 0)
        
        # BLUE detection (particularly improved)
        # Blue in HSV has high H, and in Lab has negative b component
        if (85 <= h <= 140) and b_val < 128:  # Blue in HSV and Lab
            # Further check in BGR
            if b > max(r, g) + 5:  # B channel is stronger 
                return "blue", (255, 0, 0)
        
        # GREEN detection (use both HSV and Lab)
        if (35 <= h <= 85) and a < 128:  # Green in HSV and Lab
            if g > max(r, b) + 5:  # G channel is stronger
                return "green", (0, 255, 0)
                
        # RED detection (improved)
        if ((0 <= h <= 15) or (165 <= h <= 180)) and a > 128:
            if r > max(g, b) + 5:  # R channel is stronger
                return "red", (0, 0, 255)
                
        # # YELLOW detection (improved)
        # if (15 <= h <= 35) and b_val > 128 and a > 128:
        #     if r > b + 10 and g > b + 10:  # Both R and G channels are stronger than B
        #         return "yellow", (0, 255, 255)
        
        # Fallback to RGB ratio analysis with more robust thresholds
        max_channel = max(r, g, b)
        if max_channel > 30:
            r_ratio = r / max_channel
            g_ratio = g / max_channel
            b_ratio = b / max_channel
            
            # Added feature: channel differences
            rg_diff = abs(r - g)
            rb_diff = abs(r - b)
            gb_diff = abs(g - b)
            
            # Define color by strongest channel with significant difference
            if b_ratio > 0.4 and b > r + 10 and b > g + 10:
                return "blue", (255, 0, 0)
            elif g_ratio > 0.4 and g > r + 10 and g > b + 10:
                return "green", (0, 255, 0)
            elif r_ratio > 0.4 and r > b + 10:
                # if r_ratio > 0.5 and g_ratio > 0.5 and g > b + 10:
                #     return "yellow", (0, 255, 255)
                return "red", (0, 0, 255)
        
        # Final check for dark colors before returning unknown
        if max(r, g, b) < 60:  # Very low RGB values
            return "black", (0, 0, 0)

        # Final aggressive check for dark colors before returning unknown
        if max(r, g, b) < 80:  # Increased from 60
            return "black", (0, 0, 0)

        return "", (128, 128, 128)

    def image_callback(self, msg):
        try:
            # Convert the ROS image message to an OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply median blur for noise reduction
            blurred_image = cv2.medianBlur(gray_image, 5)
            
            # Apply Canny edge detection
            param1 = 50
            edges = cv2.Canny(blurred_image, param1 / 2, param1)
            cv2.imshow("Canny", edges)
            
            # Check if depth data is available
            if self.depth_data is None:
                self.get_logger().warn("No depth data available")
                return
                
            # Get the current depth map
            depth_map = self.depth_data
            
            # Use Hough Gradient Alternative method to detect circles
            circles = cv2.HoughCircles(
                blurred_image,
                cv2.HOUGH_GRADIENT_ALT,
                dp=1.5,
                minDist=10,
                param1=param1,
                param2=0.9,
                minRadius=5,
                maxRadius=100
            )
            
            # If circles are detected, process them
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    x, y, r = circle  # Extract circle center (x, y) and radius (r)
                    
                    # Check if this is a hollow ring
                    if self.is_hollow_ring(x, y, r, depth_map):
                        # Get ring color
                        color_name, color_bgr = self.get_ring_color(frame, x, y, r)
                        
                        # Skip if we couldn't determine a color
                        if not color_name:
                            continue
                        
                        # Draw the detected hollow ring in its detected color
                        cv2.circle(frame, (x, y), r, color_bgr, 2)  # Circle outline
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Center point in red
                        
                        # Add color label below the ring
                        cv2.putText(frame, 
                                    color_name, 
                                    (x - 20, y + r + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color_bgr,
                                    2)
                        
                        # Get 3D position from point cloud
                        position_3d, perimeter_points_3d = self.get_point_cloud_position(x, y, r)
                        if position_3d is not None:
                            # Transform to map frame
                            map_position = self.transform_point_to_map(position_3d)
                            
                            if map_position is not None:
                                # Calculate normals if we have enough perimeter points
                                if perimeter_points_3d and len(perimeter_points_3d) >= 3:
                                    # Transform all perimeter points to map frame
                                    map_perimeter_points = []
                                    for point in perimeter_points_3d:
                                        map_point = self.transform_point_to_map(point)
                                        if map_point is not None:
                                            map_perimeter_points.append(map_point)
                                    
                                    normal1, normal2 = self.calculate_ring_normal(map_perimeter_points)
                                else:
                                    normal1, normal2 = None, None
                                
                                self.get_logger().info(
                                    f"{color_name.upper()} hollow ring detected at "
                                    f"({x}, {y}) with radius {r}, map position: {map_position}"
                                    f", normals: {normal1}, {normal2}"
                                )
                                
                                # Store the ring data with normals
                                self.update_ring(map_position, r, color_name, color_bgr, normal1, normal2)
                        else:
                            self.get_logger().info(
                                f"{color_name.upper()} hollow ring detected at "
                                f"({x}, {y}) with radius {r}, but no 3D position determined"
                            )
                    else:
                        # Draw non-hollow circles in gray
                        cv2.circle(frame, (x, y), r, (128, 128, 128), 1)
            
            # Show the detected circles
            cv2.imshow("Detected Rings", frame)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")

    def update_ring(self, position, radius_px, color_name, color_bgr, normal1=None, normal2=None):
        """Update ring data in storage, create new entry if needed"""
        # Filter out rings where y-coordinate is greater than -1.00 in map frame
        if position[1] < -1.00 or position[0] < -4.50:
            self.get_logger().info(f"Rejecting {color_name} ring at {position}")
            return
        
        # Check if this ring is already in our dictionary by checking if it's near an existing ring
        matched_hash = None
        for ring_hash, ring_data in self.rings.items():
            distance = np.linalg.norm(position - ring_data.position)
            if distance < self.ring_position_threshold:
                matched_hash = ring_hash
                break
        
        current_time = time.time()
        
        # If matching ring found, update its data
        if matched_hash:
            # Update position with some smoothing
            smoothing = 0.3
            self.rings[matched_hash].position = (1 - smoothing) * self.rings[matched_hash].position + smoothing * position
            self.rings[matched_hash].last_seen = current_time
            
            # Update normals if available
            if normal1 is not None and normal2 is not None:
                self.rings[matched_hash].normal1 = normal1
                self.rings[matched_hash].normal2 = normal2
            
            # Check if we should announce this ring again
            if (self.rings[matched_hash].color_name != color_name or 
            (not self.rings[matched_hash].announced) or
            (current_time - self.rings[matched_hash].last_seen > self.announce_cooldown)):
                self.announce_ring_color(color_name)
                self.rings[matched_hash].announced = True
            
            # Update color if it changed
            if self.rings[matched_hash].color_name != color_name:
                self.rings[matched_hash].color_name = color_name
                self.rings[matched_hash].color_bgr = color_bgr
        else:
            # Create new ring entry
            pos_hash = self.position_hash(position)
            self.rings[pos_hash] = RingData(
                position=position,
                radius=radius_px * 0.002,  # Convert pixels to approximate meters
                color_name=color_name,
                color_bgr=color_bgr,
                last_seen=current_time,
                announced=False,
                normal1=normal1,
                normal2=normal2
            )
            # Announce new ring
            self.announce_ring_color(color_name)
            self.rings[pos_hash].announced = True
            
            self.get_logger().info(f"Added new {color_name} ring at {position} with Y={position[1]}")

    def cleanup_old_rings(self):
        """Remove rings that haven't been seen recently"""
        current_time = time.time()
        keys_to_remove = []
        
        for ring_hash, ring_data in self.rings.items():
            if current_time - ring_data.last_seen > self.ring_timeout:
                keys_to_remove.append(ring_hash)
        
        for key in keys_to_remove:
            self.rings.pop(key)
            
        if keys_to_remove:
            self.get_logger().debug(f"Removed {len(keys_to_remove)} old rings")

    def publish_ring_markers(self):
        """Publish markers for all tracked rings"""
        if not self.rings:
            return
            
        marker_array = MarkerArray()
        
        for ring_hash, ring_data in self.rings.items():
            # Ring position marker (sphere)
            ring_marker = Marker()
            ring_marker.header.frame_id = "map"
            ring_marker.header.stamp = self.get_clock().now().to_msg()
            ring_marker.ns = "ring_positions"
            ring_marker.id = hash(ring_hash) % 10000  # Use hash for ID
            ring_marker.type = Marker.SPHERE
            ring_marker.action = Marker.ADD
            ring_marker.pose.position.x = ring_data.position[0]
            ring_marker.pose.position.y = ring_data.position[1]
            ring_marker.pose.position.z = ring_data.position[2]
            ring_marker.pose.orientation.w = 1.0
            ring_marker.scale.x = ring_marker.scale.y = ring_marker.scale.z = ring_data.radius * 10  # Diameter
            
            # Set color from BGR to RGB
            b, g, r = ring_data.color_bgr
            ring_marker.color.r = float(r) / 255.0
            ring_marker.color.g = float(g) / 255.0
            ring_marker.color.b = float(b) / 255.0
            ring_marker.color.a = 0.8
            
            # Set lifetime
            if self.marker_lifetime > 0:
                ring_marker.lifetime.sec = int(self.marker_lifetime)
                ring_marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
            
            marker_array.markers.append(ring_marker)
            
            # Text marker for color label
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "ring_colors"
            text_marker.id = hash(ring_hash) % 10000  # Use hash for ID
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = ring_data.position[0]
            text_marker.pose.position.y = ring_data.position[1]
            text_marker.pose.position.z = ring_data.position[2] + 0.1  # Above the sphere
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.20 # Text size
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.8
            text_marker.text = ring_data.color_name.upper()
            
            if self.marker_lifetime > 0:
                text_marker.lifetime.sec = int(self.marker_lifetime)
                text_marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
            
            marker_array.markers.append(text_marker)
            
            # Add normal vector markers if available
            if ring_data.normal1 is not None and ring_data.normal2 is not None:
                # Helper function to create a quaternion from normal vector
                def create_quat_from_normal(normal):
                    # We'll use the normal vector as the z-axis
                    z_axis = normal
                    # Choose an arbitrary x-axis perpendicular to z
                    x_axis = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    # Y-axis completes the right-handed coordinate system
                    y_axis = np.cross(z_axis, x_axis)
                    
                    # Create rotation matrix [x_axis, y_axis, z_axis]
                    rot_matrix = np.zeros((4, 4))
                    rot_matrix[:3, 0] = x_axis
                    rot_matrix[:3, 1] = y_axis
                    rot_matrix[:3, 2] = z_axis
                    rot_matrix[3, 3] = 1.0
                    
                    # Convert rotation matrix to quaternion
                    q = tf_transformations.quaternion_from_matrix(rot_matrix)
                    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                
                # Create markers for both normal directions
                for i, (normal, color) in enumerate([
                    (ring_data.normal1, (0.0, 1.0, 0.0)), # Green for normal1
                    (ring_data.normal2, (1.0, 0.0, 0.0))  # Red for normal2
                ]):
                    normal_marker = Marker()
                    normal_marker.header.frame_id = "map"
                    normal_marker.header.stamp = self.get_clock().now().to_msg()
                    normal_marker.ns = "ring_normals"
                    normal_marker.id = hash(ring_hash) % 10000 + i  # Use hash for ID + offset for direction
                    normal_marker.type = Marker.ARROW
                    normal_marker.action = Marker.ADD
                    
                    # Set position to ring center
                    normal_marker.pose.position.x = ring_data.position[0]
                    normal_marker.pose.position.y = ring_data.position[1]
                    normal_marker.pose.position.z = ring_data.position[2]
                    
                    # Set orientation based on normal direction
                    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
                    # normal = -normal
                    normal_marker.pose.orientation = create_quat_from_normal(normal)
                    
                    # Set arrow dimensions
                    arrow_length = 0.5# 10cm
                    arrow_width = 0.05  # 1cm
                    normal_marker.scale.x = arrow_length  # length
                    normal_marker.scale.y = arrow_width   # width
                    normal_marker.scale.z = arrow_width   # height
                    
                    # Set color
                    normal_marker.color.r = color[0]
                    normal_marker.color.g = color[1]
                    normal_marker.color.b = color[2]
                    normal_marker.color.a = 1.0
                    
                    # Set lifetime
                    if self.marker_lifetime > 0:
                        normal_marker.lifetime.sec = int(self.marker_lifetime)
                        normal_marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
                    
                    marker_array.markers.append(normal_marker)
        
        # Publish the marker array
        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    detector = RingDetector()
    rclpy.spin(detector)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()