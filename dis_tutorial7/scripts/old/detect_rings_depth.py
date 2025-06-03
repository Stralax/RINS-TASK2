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
import tf_transformations
from nav_msgs.msg import OccupancyGrid
from sklearn.decomposition import PCA


@dataclass
class RingData:
    position: np.ndarray  # 3D position in map frame [x, y, z]
    radius: float         # Radius in meters
    color_name: str       # Color name (e.g., "red", "blue", "green", "yellow", "black")
    color_bgr: tuple      # BGR color tuple for visualization
    last_seen: float      # Timestamp when last detected
    announced: bool       # Whether this ring has been announced via speech
    normal: np.ndarray = None    # Normal vector of the ring plane

class RingDetector(Node):
    def __init__(self):
        super().__init__('ring_detector')
        self.bridge = CvBridge()
        
        # Subscriptions
        self.costmap_data = None
        self.costmap_metadata = None
        # Create a one-time subscription to get the costmap
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, 
            "/global_costmap/costmap",  # This is the standard topic for global costmap
            self._costmap_callback,
            1
        )
        # self.get_logger().info("Subscribed to global costmap")

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
            return None
            
        try:
            # Convert point cloud to numpy array
            pc_array = pc2.read_points_numpy(
                self.pointcloud_data, 
                field_names=("x", "y", "z")
            ).reshape((self.pointcloud_data.height, self.pointcloud_data.width, 3))

            

            # Sample points around the ring
            ring_points = []
            num_samples = 8
            for angle in np.linspace(0, 2*np.pi, num_samples, endpoint=False):
                px = int(x + r * 0.8 * np.cos(angle))  # Sample at 80% of radius to get on the ring
                py = int(y + r * 0.8 * np.sin(angle))
                
                # Check if point is within image bounds
                if 0 <= px < self.pointcloud_data.width and 0 <= py < self.pointcloud_data.height:
                    point = pc_array[py, px]
                    if np.isfinite(point).all() and not np.isnan(point).any():
                        ring_points.append(point)
            
            # If we have enough points, compute the median position
            if len(ring_points) >= 3:
                ring_position = np.median(np.array(ring_points), axis=0)
                return ring_position, ring_points
                
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
            # elif r_ratio > 0.4 and r > b + 10:
            #     if r_ratio > 0.5 and g_ratio > 0.5 and g > b + 10:
            #         return "yellow", (0, 255, 255)
            #     return "red", (0, 0, 255)
        
        # Final check for dark colors before returning unknown
        if max(r, g, b) < 60:  # Very low RGB values
            return "black", (0, 0, 0)

        # Final aggressive check for dark colors before returning unknown
        if max(r, g, b) < 80:  # Increased from 60
            return "black", (0, 0, 0)

        return "", (128, 128, 128)
    
    def calculate_ring_normal(self, points_3d):
        """Calculate the normal vector of the ring based on 3D points using PCA"""
        if points_3d is None or len(points_3d) < 3:
            self.get_logger().warn("Not enough points to calculate normal")
            return None
        
        try:
            # Apply PCA to find principal components
            pca = PCA(n_components=3)
            pca.fit(points_3d)
            
            # The normal is the eigenvector corresponding to the smallest eigenvalue (3rd component)
            normal = pca.components_[2]
            
            # Normalize the vector
            normal_length = np.linalg.norm(normal)
            if normal_length > 0:
                normal = normal / normal_length
            
            # Make sure the normal points "up" in camera frame (z-axis is forward)
            # This heuristic assumes the camera is roughly above the rings
            if normal[2] < 0:  # If z component is negative, flip the normal
                normal = -normal

            normal[2] = 0     
            self.get_logger().info(f"Ring normal calculated: {normal}")
            return normal
            
        except Exception as e:
            self.get_logger().error(f"Error calculating ring normal: {e}")
            return None

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
                        
                        # Get 3D points and position using depth
                        ring_points_3d, position_3d = self.get_ring_depth_points(x, y, r, depth_map)
                        
                        if position_3d is not None and len(ring_points_3d) >= 3:
                            # Calculate normal from the 3D points in camera frame
                            normal = self.calculate_ring_normal(ring_points_3d)
                            
                            # Transform position and normal to map frame
                            map_position = self.transform_point_to_map(position_3d)
                            
                            if map_position is not None:
                                # Transform the normal vector to map frame if we have a valid normal
                                map_normal = None
                                if normal is not None:
                                    map_normal = self.transform_vector_to_map(normal)
                                    
                                self.get_logger().info(
                                    f"{color_name.upper()} hollow ring detected at "
                                    f"({x}, {y}) with radius {r}, map position: {map_position}"
                                )
                                
                                # Store the ring data with normal
                                self.update_ring(map_position, r, color_name, color_bgr, map_normal)
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

    def update_ring(self, position, radius_px, color_name, color_bgr, normal):
        """Update ring data in storage, create new entry if needed"""
        # Check if this ring is already in our dictionary by checking if it's near an existing ring
        if position[1] < -1.00 or position[0] < -4.50:
            self.get_logger().info(f"Rejecting {color_name} ring at {position}")
            return
        
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
            
            # Check if we should announce this ring again (only if color changed or sufficient time passed)
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
                normal=normal
            )
            # Announce new ring
            self.announce_ring_color(color_name)
            self.rings[pos_hash].announced = True

    # Add this function to the RingDetector class
    def get_ring_depth_points(self, x, y, r, depth_map):
        """
        Get 3D points around ring perimeter using the depth camera data.
        Returns both the 3D points and the calculated centroid.
        """
        if depth_map is None:
            self.get_logger().warn("No depth data available for ring points")
            return None, None

        try:
            # Camera intrinsic parameters (from your top camera info)
            fx = 277.0  # Focal length x
            fy = 277.0  # Focal length y
            cx = 160.0  # Principal point x
            cy = 120.0  # Principal point y
            
            # Sample points around the ring perimeter
            ring_points_3d = []
            num_points = 8  # Sample 8 points for the normal calculation
            debug_img = np.zeros_like(depth_map, dtype=np.float32)
            
            for angle in np.linspace(0, 2*np.pi, num_points, endpoint=False):
                # Sample at 80% of radius to ensure we're on the ring and not background
                px = int(x + r * 0.8 * np.cos(angle))
                py = int(y + r * 0.8 * np.sin(angle))
                
                # Check if point is within image bounds
                if 0 <= px < self.depth_width and 0 <= py < self.depth_height:
                    depth = depth_map[py, px]
                    
                    # Check for valid depth value
                    if depth > 0 and not np.isnan(depth) and not np.isinf(depth):
                        # Convert from depth image to 3D point (backprojection)
                        # Z is the depth value
                        z = depth
                        # X = (u - cx) * Z / fx
                        x_3d = (px - cx) * z / fx
                        # Y = (v - cy) * Z / fy
                        y_3d = (py - cy) * z / fy
                        
                        # Add point to our collection
                        ring_points_3d.append([x_3d, y_3d, z])
                        
                        # Mark this point in the debug image
                        debug_img[py, px] = depth
            
            # Show sampled points in debug window
            if ring_points_3d:
                # Normalize for display
                debug_img = np.nan_to_num(debug_img, nan=0.0, posinf=0.0, neginf=0.0)
                debug_img = cv2.normalize(debug_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                debug_colored = cv2.applyColorMap(debug_img, cv2.COLORMAP_JET)
                
                # Draw circles to show where we sampled from
                for angle in np.linspace(0, 2*np.pi, num_points, endpoint=False):
                    px = int(x + r * 0.8 * np.cos(angle))
                    py = int(y + r * 0.8 * np.sin(angle))
                    if 0 <= px < self.depth_width and 0 <= py < self.depth_height:
                        cv2.circle(debug_colored, (px, py), 2, (255, 255, 255), -1)
                
                cv2.imshow("Depth Points", debug_colored)
            
            # Calculate the centroid (average position) of the ring
            if len(ring_points_3d) >= 3:  # Need at least 3 points to define a plane
                ring_points_3d = np.array(ring_points_3d)
                centroid = np.mean(ring_points_3d, axis=0)
                return ring_points_3d, centroid
            else:
                self.get_logger().debug("Not enough valid depth points for ring")
                return None, None
    
        except Exception as e:
            self.get_logger().error(f"Error getting ring depth points: {e}")
            return None, None

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

    def valid_position(self, pos1, pos2):
        """Check if the position is not in the danger zone using global costmap data"""
        try:
            # Subscribe to the global costmap if we haven't already
            if not hasattr(self, 'costmap_data'):
                # Initialize costmap data storage
                self.costmap_data = None
                self.costmap_metadata = None
                # Create a one-time subscription to get the costmap
                self.costmap_sub = self.create_subscription(
                    OccupancyGrid, 
                    "/global_costmap/costmap",  # This is the standard topic for global costmap
                    self._costmap_callback,
                    1
                )
                self.get_logger().info("Subscribed to global costmap")
            
            # If we don't have costmap data yet, use the distance-based fallback
            if self.costmap_data is None:
                self.get_logger().warn("No costmap data available yet, using distance-based fallback")
                return self._fallback_valid_position(pos1, pos2)
            
            # Check validity of both positions using the costmap
            pos1_valid = self._check_costmap_position(pos1)
            pos2_valid = self._check_costmap_position(pos2)
            
            # Log which positions are valid
            self.get_logger().debug(f"Position 1 ({pos1[:2]}) valid: {pos1_valid}")
            self.get_logger().debug(f"Position 2 ({pos2[:2]}) valid: {pos2_valid}")
            
            # Return the valid position or fallback if needed
            if pos1_valid and not pos2_valid:
                return pos1
            elif pos2_valid and not pos1_valid:
                return pos2
            elif pos1_valid and pos2_valid:
                # If both are valid, prefer position 2 (or you could choose based on other criteria)
                return pos2
            else:
                # If both positions are invalid, try to find a nearby valid position
                self.get_logger().warn("Both positions are invalid, finding closest valid position")
                
                # Try to find a valid position by stepping back from the ring
                ring_pos = (pos1 + pos2) / 2.0  # Approximate ring position
                
                # Try different directions from the ring
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    for distance in [0.5, 0.7, 1.0, 1.5]:
                        test_pos = ring_pos.copy()
                        test_pos[0] += np.cos(angle) * distance
                        test_pos[1] += np.sin(angle) * distance
                        
                        if self._check_costmap_position(test_pos):
                            self.get_logger().info(f"Found valid fallback position at ({test_pos[0]:.2f}, {test_pos[1]:.2f})")
                            return test_pos
                
                # If all else fails, return position 1 as fallback
                self.get_logger().warn(f"Could not find valid position, returning pos1 as fallback")
                return pos1
        
        except Exception as e:
            self.get_logger().error(f"Error in valid_position: {e}")
            return pos1

    def _costmap_callback(self, msg):
        """Store the latest costmap data"""
        self.costmap_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.costmap_metadata = msg.info
        self.get_logger().info(f"Received costmap: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}")
        
        # Unsubscribe after receiving the first costmap to save resources
        # We can resubscribe later if needed for updates
        self.destroy_subscription(self.costmap_sub)
        delattr(self, 'costmap_sub')

    def _check_costmap_position(self, position):
        """Check if a position is valid according to the costmap"""
        if self.costmap_data is None or self.costmap_metadata is None:
            return True  # No costmap data, assume valid
        
        # Convert world coordinates to costmap grid coordinates
        grid_x = int((position[0] - self.costmap_metadata.origin.position.x) / self.costmap_metadata.resolution)
        grid_y = int((position[1] - self.costmap_metadata.origin.position.y) / self.costmap_metadata.resolution)
        
        # Check if coordinates are within the map bounds
        if (0 <= grid_x < self.costmap_metadata.width and 
            0 <= grid_y < self.costmap_metadata.height):
            
            # Get the cell value
            cell_value = self.costmap_data[grid_y, grid_x]
            
            # In ROS costmaps:
            # 0-254: Indicates the cost of occupancy (0 = free, 100+ = occupied, 254 = lethal)
            # -1/255: Unknown
            
            # Free space has value 0, threshold is typically 50-60 for navigation
            FREE_THRESHOLD = 50  # Values below this are considered free space
            
            # Position is valid if it's free space (below threshold)
            is_valid = cell_value < FREE_THRESHOLD
            
            if not is_valid:
                self.get_logger().debug(f"Position ({position[0]:.2f}, {position[1]:.2f}) has costmap value {cell_value}")
            
            return is_valid
        
        # Outside the map bounds
        self.get_logger().warn(f"Position ({position[0]:.2f}, {position[1]:.2f}) is outside costmap bounds")
        return False

    def _fallback_valid_position(self, pos1, pos2):
        """Fallback method using distance-based heuristics if costmap is unavailable"""
        try:
            # Use distances from current robot position as a simple heuristic
            transform = self.tf_buffer.lookup_transform(
                "base_link", 
                "map",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Get robot position in map frame
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            
            # Calculate distance to robot for each position
            dist1 = np.sqrt((pos1[0] - robot_x)**2 + (pos1[1] - robot_y)**2)
            dist2 = np.sqrt((pos2[0] - robot_x)**2 + (pos2[1] - robot_y)**2)
            
            # Simple check: position is valid if it's not too close to the robot
            # and not too far (which might indicate it's outside the map or in an obstacle)
            MIN_DIST = 0.5  # Minimum safe distance from robot in meters
            MAX_DIST = 3.0  # Maximum reasonable distance in meters
            
            pos1_valid = MIN_DIST <= dist1 <= MAX_DIST
            pos2_valid = MIN_DIST <= dist2 <= MAX_DIST
            
            if pos1_valid and not pos2_valid:
                return pos1
            elif pos2_valid and not pos1_valid:
                return pos2
            elif pos1_valid and pos2_valid:
                # If both are valid, prefer the one farther from robot for safety
                return pos1 if dist1 > dist2 else pos2
            else:
                # If both invalid, use pos2 (consistent with original implementation)
                return pos2
                
        except TransformException:
            # If we can't get the transform, just return pos2
            self.get_logger().warn("Could not get transform for position check, using pos2")
            return pos2
        

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
            
            # Add point at ring_position + normal * 0.8 if normal is available
            if ring_data.normal is not None and np.linalg.norm(ring_data.normal) > 0:
                # Normalize the normal vector
                normal = ring_data.normal / np.linalg.norm(ring_data.normal)
                
                # Create markers for both normal directions
                for i, direction in enumerate([1.0, -1.0]):  # Positive and negative directions
                    # Calculate point position for this direction
                    point_position = ring_data.position + direction * normal * 0.2  # 20cm along normal
                    
                    # Create arrow marker
                    normal_marker = Marker()
                    normal_marker.header.frame_id = "map"
                    normal_marker.header.stamp = self.get_clock().now().to_msg()
                    normal_marker.ns = "ring_normals"
                    normal_marker.id = (hash(ring_hash) % 10000) + i  # Use hash for ID + offset for direction
                    normal_marker.type = Marker.ARROW
                    normal_marker.action = Marker.ADD
                    
                    # Set start position (at ring center)
                    normal_marker.pose.position.x = ring_data.position[0]
                    normal_marker.pose.position.y = ring_data.position[1]
                    normal_marker.pose.position.z = ring_data.position[2]
                    
                    # Set orientation using a quaternion to point in the normal direction
                    # We need to calculate a rotation from the default arrow direction (x-axis)
                    # to our normal direction
                    
                    # Create a quaternion for this rotation
                    # First, get the cross product of the default direction [1,0,0] with our normal
                    # to find the rotation axis
                    rotation_axis = np.cross([1.0, 0.0, 0.0], direction * normal)
                    rotation_axis_norm = np.linalg.norm(rotation_axis)
                    
                    if rotation_axis_norm < 1e-6:  # If close to zero, normal is aligned with x-axis
                        # Default quaternion (no rotation) or 180 degree if negative
                        if direction * normal[0] > 0:
                            quat = [0.0, 0.0, 0.0, 1.0]  # No rotation
                        else:
                            quat = [0.0, 0.0, 1.0, 0.0]  # 180 degree around z
                    else:
                        # Normalize the axis
                        rotation_axis = rotation_axis / rotation_axis_norm
                        
                        # Get the angle between the vectors (using dot product)
                        dot_product = np.dot([1.0, 0.0, 0.0], direction * normal)
                        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                        
                        # Create quaternion [x,y,z,w] for this axis-angle rotation
                        sin_half_angle = np.sin(angle / 2.0)
                        quat = [
                            rotation_axis[0] * sin_half_angle,
                            rotation_axis[1] * sin_half_angle,
                            rotation_axis[2] * sin_half_angle,
                            np.cos(angle / 2.0)
                        ]
                    
                    normal_marker.pose.orientation.x = quat[0]
                    normal_marker.pose.orientation.y = quat[1]
                    normal_marker.pose.orientation.z = quat[2]
                    normal_marker.pose.orientation.w = quat[3]
                    
                    # Set arrow dimensions
                    arrow_length = 0.2  # 20cm
                    arrow_width = 0.02  # 2cm
                    normal_marker.scale.x = arrow_length
                    normal_marker.scale.y = arrow_width
                    normal_marker.scale.z = arrow_width
                    
                    # Set color (green for positive, red for negative)
                    if direction > 0:
                        normal_marker.color.r = 0.0
                        normal_marker.color.g = 1.0
                        normal_marker.color.b = 0.0
                    else:
                        normal_marker.color.r = 1.0
                        normal_marker.color.g = 0.0
                        normal_marker.color.b = 0.0
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