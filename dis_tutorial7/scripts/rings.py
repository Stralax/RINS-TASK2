#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros
import time
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from rclpy.duration import Duration

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String, Bool
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs as tfg
from nav2_msgs.action import Spin, NavigateToPose
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

from geometry_msgs.msg import Point, Quaternion, PoseArray, Vector3Stamped
from std_msgs.msg import Int32MultiArray, MultiArrayDimension

from rclpy.action import ActionClient
from collections import defaultdict
from tf_transformations import quaternion_from_euler, euler_from_quaternion

# For map data
from nav_msgs.msg import OccupancyGrid

import math
import tf_transformations  # For quaternion conversions

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')


        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.checking_frames = 0
        timer_frequency = 2
        timer_period = 1/timer_frequency

        self.depth_elipses = []

        self.bridge = CvBridge()

        self.marker_array = MarkerArray()
        self.marker_num = 1


        self.latest_ellipse_mask = None
        # self.crna_mapa = cv2.imread("/home/alpha/Desktop/task2_new/src/dis_tutorial3/maps/map34.pgm", cv2.IMREAD_GRAYSCALE)

        self.circle_positions = defaultdict(list)  # circle_id -> list of positions

        self.detected_points = []
        self.image_height = None
        self.image_width = None
        self.current_target = None
        self.nav_in_progress = False
        self.pointcloud_data = None
        self.detected_circles = []
        
        self.ring_colors = {}  # {ring_id: {"red": count, "green": count, ...}}
        self.color_history_max_size = 10  
        
        self.circle_frame_counts = {} 
        self.frames_to_confirm = 20 
        self.confirmed_circles = []  

        # Map related attributes
        self.map_np = None
        self.map_data = {"map_load_time": None,
                        "resolution": None,
                        "width": None,
                        "height": None,
                        "origin": None} 
        
        self.costmap_np = None
        self.costmap_info = None
        self.costmap_data = {"resolution": None,
                            "width": None,
                            "height": None,
                            "origin_x": None,
                            "origin_y": None,
                            "orientation": None}


        self.cst_map_np = None
        self.cst_map_data = {
            "map_load_time": None,
            "resolution": None,
            "width": None,
            "height": None,
            "origin": None,
        }
        

        self.map_y_temp = 0.0
        self.map_x_temp = 0.0

        self.black_threshold = 35
        self.color_min_diff = 20
        self.saturation_threshold = 30  
        self.value_threshold = 60      
        
        self.color_history = []
        self.color_history_max_size = 3

        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, 
            "/oakd/rgb/preview/depth/points", 
            self.pointcloud_callback, 
            qos_profile_sensor_data
        )

        self.occupancy_grid_sub = self.create_subscription(
            OccupancyGrid, 
            "/map", 
            self.map_callback, 
            qos_profile
        )
        self.global_costmap_sub = self.create_subscription(
            OccupancyGrid, 
            "/global_costmap/costmap", 
            self.costmap_callback, 
            qos_profile
        )



        self.ring_detected_sub = self.create_subscription(
            Bool,
            "/ring_detected_color_response",
            self.ring_detected_callback,
            10
        )

        # Create necessary publishers
        self.audio_pub = self.create_publisher(
            String,
            "/robot/speech",
            10
        )
        self.ring_detected_pub = self.create_publisher(
            Bool,
            "/ring_detected_at_top",
            10
        )
        self.circle_goal_pub = self.create_publisher(
            PoseStamped,
            "/circle_navigation_goal",
            10
        )
        self.circle_color_pub = self.create_publisher(
            String,
            "/circle_color",
            10
        )
        self.marker_pub = self.create_publisher(
            Marker,
            "/circle_marker",
            10
        )
        self.marker_array_pub = self.create_publisher(
            MarkerArray, 
            "/circle_points", 
            10
        )
        self.costmap_sub = self.create_subscription(OccupancyGrid, "/global_costmap/costmap", self.global_cost_callback, qos_profile_sensor_data)

        self.offset_marker_pub = self.create_publisher(Marker, "/offset_pos_marker", 10)

        # Set up timers
        self.check_timer = self.create_timer(0.1, self.check_top_position)
        self.circle_pub_timer = self.create_timer(1.0, self.publish_confirmed_circles)

        self.flag_response_ring = False
        self.raw_depth = None

        # Optional debug windows
        # Comment these lines if you don't need them
        # cv2.namedWindow("Color Calibration", cv2.WINDOW_NORMAL)
        # cv2.createTrackbar("Black Threshold", "Color Calibration", self.black_threshold, 100, self.update_black_threshold)
        # cv2.createTrackbar("Color Difference", "Color Calibration", self.color_min_diff, 100, self.update_color_diff)
        # cv2.createTrackbar("Saturation Threshold", "Color Calibration", self.saturation_threshold, 100, self.update_saturation_threshold)
        # cv2.createTrackbar("Value Threshold", "Color Calibration", self.value_threshold, 100, self.update_value_threshold)

        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)

    def update_black_threshold(self, value):
        self.black_threshold = value

    def update_color_diff(self, value):
        self.color_min_diff = value
        
    def update_saturation_threshold(self, value):
        self.saturation_threshold = value
        
    def update_value_threshold(self, value):
        self.value_threshold = value

    def ring_detected_callback(self, data):
        self.flag_response_ring = data.data

    def pointcloud_callback(self, data):
        self.pointcloud_data = data

    def costmap_callback(self, msg):
        # Convert to numpy array
        self.costmap_np = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        # Flip vertically for OpenCV compatibility
        self.costmap_np = np.flipud(self.costmap_np)
        
        # Store costmap metadata
        self.costmap_info = {
            "resolution": msg.info.resolution,
            "width": msg.info.width,
            "height": msg.info.height,
            "origin_x": msg.info.origin.position.x,
            "origin_y": msg.info.origin.position.y,
            "orientation": [
                msg.info.origin.orientation.x,
                msg.info.origin.orientation.y,
                msg.info.origin.orientation.z,
                msg.info.origin.orientation.w
            ]
        }

    def global_cost_callback(self, msg):
        # reshape the message vector back into a map
        self.map_np = np.asarray(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width
        )
        # fix the direction of Y (origin at top for OpenCV, origin at bottom for ROS2)
        self.map_np = np.flipud(self.map_np)
        # change the colors so they match with the .pgm image
        unique_values, counts = np.unique(self.map_np, return_counts=True)

        self.cst_map_data["map_load_time"] = msg.info.map_load_time
        self.cst_map_data["resolution"] = msg.info.resolution
        self.cst_map_data["width"] = msg.info.width
        self.cst_map_data["height"] = msg.info.height
        quat_list = [
            msg.info.origin.orientation.x,
            msg.info.origin.orientation.y,
            msg.info.origin.orientation.z,
            msg.info.origin.orientation.w,
        ]
        self.cst_map_data["origin"] = [
            msg.info.origin.position.x,
            msg.info.origin.position.y,
            euler_from_quaternion(quat_list)[-1],
        ]

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.cv_image = cv_image  # Store for later use
            self.image_height = cv_image.shape[0]

            blue = cv_image[:,:,0]
            green = cv_image[:,:,1]
            red = cv_image[:,:,2]

            # Tranform image to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)

            skeleton = skeletonize(cv2.bitwise_not(thresh)).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            dilated = cv2.dilate(skeleton, kernel, iterations=1)

            # Extract contours
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Fit elipses to all extracted contours
            image_elispes = []
            for cnt in contours:
                if cnt.shape[0] >= 7:
                    ellipse = cv2.fitEllipse(cnt)
                    image_elispes.append(ellipse)

            # Code for finding pairs of ellipses
            image_elispes2 = set()
            for n in range(len(image_elispes)):
                for m in range(n + 1, len(image_elispes)):
                    e1 = image_elispes[n]
                    e2 = image_elispes[m]
                    dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                    angle_diff = np.abs(e1[2] - e2[2])

                    if dist >= 5:
                        continue

                    if angle_diff > 4:
                        continue

                    e1_minor_axis = e1[1][0]
                    e1_major_axis = e1[1][1]

                    e2_minor_axis = e2[1][0]
                    e2_major_axis = e2[1][1]

                    if e1_major_axis >= e2_major_axis and e1_minor_axis >= e2_minor_axis: 
                        le = e1 
                        se = e2 
                    elif e2_major_axis >= e1_major_axis and e2_minor_axis >= e1_minor_axis:
                        le = e2 
                        se = e1 
                    else:
                        continue # if one ellipse does not contain the other, it is not a ring
                    
                    border_major = (le[1][1]-se[1][1])/2
                    border_minor = (le[1][0]-se[1][0])/2
                    border_diff = np.abs(border_major - border_minor)

                    if border_diff > 5:
                        continue
                        
                    image_elispes2.add(e1)
                    image_elispes2.add(e2)

            candidates = []
            tolerance = 3  

            for depth_ellipse in self.depth_elipses:
                center_d = depth_ellipse[0]  # (x, y)

                for image_ellipse in image_elispes:
                    center_i = image_ellipse[0]  # (x, y)    

                    dx = center_i[0] - center_d[0]
                    dy = center_i[1] - center_d[1]
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    
                    if dist <= tolerance:
                        candidates.append((depth_ellipse, image_ellipse))

            self.depth_elipses.clear()

            for c in candidates:
                e1 = c[0]
                e2 = c[1]

                if e2[1][0] > e1[1][0] and e2[1][1] > e1[1][1]:  # Compare major and minor axes
                    outer_ellipse = e2
                    inner_ellipse = e1
                else:
                    outer_ellipse = e1
                    inner_ellipse = e2
                
                center_x = int(outer_ellipse[0][0])
                center_y = int(outer_ellipse[0][1])
                radius = max(outer_ellipse[1]) / 2
                
                mask = np.zeros_like(gray)
                cv2.ellipse(mask, outer_ellipse, 255, -1) 
                cv2.ellipse(mask, inner_ellipse, 0, -1)   
                
                ring_mask = mask > 0


                b_mean = np.mean(blue[ring_mask])
                g_mean = np.mean(green[ring_mask])
                r_mean = np.mean(red[ring_mask])




                if np.any(ring_mask):
                    r_pixels = red[ring_mask]
                    g_pixels = green[ring_mask]
                    b_pixels = blue[ring_mask]
                    
                    ref_r, ref_g, ref_b = 178, 178, 178
                    
                    distinctive_pixels = (r_pixels != ref_r) | (g_pixels != ref_g) | (b_pixels != ref_b)
                    
                    black_threshold = 30
                    black_pixels = (r_pixels <= black_threshold) & (g_pixels <= black_threshold) & (b_pixels <= black_threshold)
                    
                    # distinctive_pixels = distinctive_pixels & ~black_pixels
                    
                    mask_distinctive = np.zeros_like(gray)
                    if np.any(distinctive_pixels):
                        y_indices, x_indices = np.where(ring_mask)
                        distinctive_coords = [(x_indices[i], y_indices[i]) for i in range(len(y_indices)) if distinctive_pixels[i]]
                        
                        if distinctive_coords:
                            distinctive_x = sum(coord[0] for coord in distinctive_coords) / len(distinctive_coords)
                            distinctive_y = sum(coord[1] for coord in distinctive_coords) / len(distinctive_coords)
                            
                            for x, y in distinctive_coords:
                                mask_distinctive[y, x] = 255
                            
                            center_x = int(distinctive_x)
                            center_y = int(distinctive_y)
                            # print(f"Using distinctive pixel average: ({center_x}, {center_y}), original center: ({int(outer_ellipse[0][0])}, {int(outer_ellipse[0][1])})")
                            
                            # Debug display of the distinctive pixels mask
                            # cv2.imshow("Distinctive pixels", mask_distinctive)
                            # cv2.waitKey(1)
                    
                    if np.any(distinctive_pixels):
                        r_mean = np.mean(r_pixels[distinctive_pixels])
                        g_mean = np.mean(g_pixels[distinctive_pixels])
                        b_mean = np.mean(b_pixels[distinctive_pixels])
                        
                        # print(f"Mean RGB: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
                 
                
                # if np.any(ring_mask):
                #     b_mean = np.mean(blue[ring_mask])
                #     g_mean = np.mean(green[ring_mask])
                #     r_mean = np.mean(red[ring_mask])
                # else:
                #     b_mean, g_mean, r_mean = 0, 0, 0  # Default values



















                
                # --- Extract 3D points from ellipse mask ---
                points_3d = []
                if self.pointcloud_data is not None and self.latest_ellipse_mask is not None:
                    pc_array = pc2.read_points_numpy(
                        self.pointcloud_data, field_names=("x", "y", "z")
                    ).reshape((self.pointcloud_data.height, self.pointcloud_data.width, 3))

                    mask = self.latest_ellipse_mask > 0
                    y_indices, x_indices = np.where(mask)

                    for y, x in zip(y_indices, x_indices):
                        if 0 <= y < pc_array.shape[0] and 0 <= x < pc_array.shape[1]:
                            point = pc_array[y, x]
                            if np.all(np.isfinite(point)):
                                points_3d.append(point)

                # do ovde gi zemam tockite od maskata







                points_3d = np.array(points_3d)
                if len(points_3d) >= 10:
                    pca = PCA(n_components=3)
                    pca.fit(points_3d)
                    normal = pca.components_[2]
                    normal /= np.linalg.norm(normal)
                    centroid = np.mean(points_3d, axis=0)
                    view_vector = -centroid
                    if np.dot(normal, view_vector) < 0:
                        normal = -normal

                    # Create a vector for transformation
                    normal_msg = Vector3Stamped()
                    normal_msg.header.frame_id = "base_link"
                    normal_msg.vector.x = float(normal[0])
                    normal_msg.vector.y = float(normal[1])
                    normal_msg.vector.z = float(normal[2])

                    # Transform the vector to map frame
                    try:
                        # First lookup the transform from base_link to map
                        time_now = rclpy.time.Time()
                        timeout = Duration(seconds=0.1)
                        trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)
                        
                        # Then transform the normal vector using that transform
                        normal_map_frame = tfg.do_transform_vector3(normal_msg, trans)
                        normal = np.array([normal_map_frame.vector.x, normal_map_frame.vector.y, normal_map_frame.vector.z])
                    except Exception as e:
                        self.get_logger().error(f"Failed to transform normal: {e}")


                    self.normal_normalized_new = normal
                    """
                    offset = centroid + normal * 0.3
                    qx, qy, qz, qw = self.calculate_orientation_quaternion(float(offset[0]), float(offset[1]), 
                                                                        float(centroid[0]), float(centroid[1]))
                    
                    # Convert numpy values to native Python floats
                    offset_float = (float(offset[0]), float(offset[1]), float(offset[2]))
                    centroid_float = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
                    normal_float = (float(normal[0]), float(normal[1]), float(normal[2]))

                    #print(f"Offset: {offset_float}, Centroid: {centroid_float}, Normal: {normal_float}, Quaternion: {(qx, qy, qz, qw)}")
                    self.publish_offset_marker(offset_float, centroid_float, normal_float, (qx, qy, qz, qw))
                    """       
                    self.get_logger().info(f"[PCA] Estimated normal: {normal}")
                else:
                    self.get_logger().warn("Too few valid points for PCA.")

                

                





















                color_name = self.detect_color(r_mean, g_mean, b_mean, (center_x, center_y))
                color_bgr = (b_mean, g_mean, r_mean)
                
                # Use the top point of the ring for navigation
                #top_ring_point = center_y - int(radius)
                self.current_target = (color_name, center_x, center_y)


                if color_name == "red":
                    ellipse_color = (0, 0, 255)
                elif color_name == "green":
                    ellipse_color = (0, 255, 0)
                elif color_name == "blue":
                    ellipse_color = (255, 0, 0)
                else:  
                    ellipse_color = (128, 128, 128) 
                
                cv2.ellipse(cv_image, outer_ellipse, ellipse_color, 2)
                cv2.ellipse(cv_image, inner_ellipse, ellipse_color, 2)
                

                text_pos = (center_x + 20, center_y)
                cv2.putText(cv_image, color_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, ellipse_color, 2)
                




                
                
                # Start navigation to the target
                self.start_navigation(center_x, center_y, color_name, color_bgr)

            cv2.imshow("Detected rings", cv_image)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Processing Error: {e}")

    def detect_color(self, r, g, b, ring_position=None):
        """
        Enhanced color detection with multiple approaches and per-ring history tracking
        Args:
            r, g, b: Mean color values in the ring region
            ring_position: (x,y) tuple of the ring position for history tracking
        Returns:
            String representing the detected color ('red', 'green', 'blue', or 'black')
        """

        r, g, b = max(0, r), max(0, g), max(0, b)
        rgb = np.array([[[r, g, b]]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv
        
        brightness = (r + g + b) / 3.0
        
        if s <= self.saturation_threshold and v <= self.value_threshold:
            detected_color = "black"
        elif brightness < self.black_threshold:
            detected_color = "black"
        else:
            colors = np.array([
                [0, 0, 0],       
                [255, 0, 0],     
                [0, 255, 0],     
                [0, 0, 255],     
                [r, g, b]        
            ])
            
            colors_normalized = colors / 255.0
            
            clustering = DBSCAN(eps=0.3, min_samples=1).fit(colors_normalized)
            labels = clustering.labels_
            
            current_label = labels[4]
            
            if current_label == labels[0]:
                detected_color = "black"
            else:
                max_channel = max(r, g, b)
                
                if r == max_channel and r > g + self.color_min_diff and r > b + self.color_min_diff:
                    detected_color = "red"
                elif g == max_channel and g > r + self.color_min_diff and g > b + self.color_min_diff:
                    detected_color = "green"
                elif b == max_channel and b > r + self.color_min_diff and b > g + self.color_min_diff:
                    detected_color = "blue"
                else:
                    if r >= g and r >= b:
                        detected_color = "red"
                    elif g >= r and g >= b:
                        detected_color = "green"
                    else:
                        detected_color = "blue"
        
        if ring_position is not None:
            map_position = self.transform_point_to_world(ring_position[0], ring_position[1])
            if map_position[0] is not None and map_position[1] is not None:
                ring_id = self.circle_idx_matching((map_position[0], map_position[1]))
                
                if ring_id is True:  # New ring
                    ring_id = len(self.detected_circles)
                
                if ring_id not in self.ring_colors:
                    self.ring_colors[ring_id] = {"red": 0, "green": 0, "blue": 0, "black": 0}
                
                self.ring_colors[ring_id][detected_color] += 1
                
                most_common_color = max(self.ring_colors[ring_id], key=self.ring_colors[ring_id].get)
                return most_common_color
        
        return detected_color
            
    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            
            self.raw_depth = np.nan_to_num(depth_image, nan=-1.0, posinf=0.0, neginf=0.0)
            
            depth_image[depth_image==np.inf] = 0

            depth_image[depth_image >= 3.0] = 0

            image_1 = depth_image / 65536.0 * 255
            if np.max(image_1) > 0:
                image_1 = image_1 / np.max(image_1) * 255

            image_viz = np.array(image_1, dtype=np.uint8)

            mask = (depth_image > 0).astype(np.uint8) * 255

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            elipse_mask = np.zeros_like(depth_image, dtype=np.uint8)
            depth_copy = np.copy(image_viz)

            for cnt in contours:
                if len(cnt) >= 7:
                    ellipse = cv2.fitEllipse(cnt)
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))  # (x, y)

                    major_axis = max(ellipse[1])
                    if major_axis > 50:
                        continue

                    # Check if the center has valid depth
                    if 0 < center[0] < depth_image.shape[1] and 0 < center[1] < depth_image.shape[0]:
                        center_depth = depth_image[center[1], center[0]]

                        if center_depth > 0:  # Valid depth at center → it's not hollow
                            continue  # skip this, it's not a ring
                        
                        # Check if the center is too low in the image frame (likely on the floor)
                        if center[1] > 120:  # Adjust threshold as needed
                            continue  # skip this, it's likely on the floor

                        # No depth at center → likely a hollow ring
                        self.depth_elipses.append(ellipse)
                        cv2.ellipse(image_viz, ellipse, (255, 0, 0), 2)



                        inflated_ellipse = (
                            ellipse[0],  # center stays the same
                            (ellipse[1][0] + 10, ellipse[1][1] + 10),  # increase both axes
                            ellipse[2]  # angle stays the same
                        )
                        elipse_mask = np.zeros_like(depth_image, dtype=np.uint8)
                        cv2.ellipse(elipse_mask, inflated_ellipse, 255, -1)

            ellipse_mask = depth_copy * elipse_mask
            ellipse_mask = np.where(ellipse_mask > 0, 255, 0).astype(np.uint8)
            self.latest_ellipse_mask = ellipse_mask.copy()
            # cv2.imshow("Ellipse mask", ellipse_mask)
            # cv2.imshow("Depth window", image_viz)
            # cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion error: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in depth processing: {str(e)}")



    def calculate_orientation_quaternion(self, from_x, from_y, to_x, to_y):
        dx = to_x - from_x
        dy = to_y - from_y
        yaw = math.atan2(dy, dx)
        q = tf_transformations.quaternion_from_euler(0, 0, yaw)
        return q[0], q[1], q[2], q[3]   


    def publish_offset_marker(self, offset_world, ring_world, normal, orientation, marker_id=0):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.5  # Arrow length
        marker.scale.y = 0.1  # Arrow width
        marker.scale.z = 0.2  # Arrow height
        marker.color.r = 0.8
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.position.x = offset_world[0]
        marker.pose.position.y = offset_world[1]
        marker.pose.position.z = offset_world[2]
        marker.pose.orientation.x = orientation[0]
        marker.pose.orientation.y = orientation[1]
        marker.pose.orientation.z = orientation[2]
        marker.pose.orientation.w = orientation[3]
        self.offset_marker_pub.publish(marker)

    def transform_point_to_world(self, center_x, center_y_pixel):
        try:
            # Check if pointcloud data exists
            if self.pointcloud_data is None:
                return None, None, None
                
            height = self.pointcloud_data.height
            width = self.pointcloud_data.width

            pc_array = pc2.read_points_numpy(self.pointcloud_data, field_names=("x", "y", "z"))
            pc_array = pc_array.reshape((height, width, 3))
            point3d = pc_array[center_y_pixel, center_x, :]
            
            if not np.all(np.isfinite(point3d)):
                return None, None, None  # Explicitly return None values

            pos_x = float(point3d[0])
            pos_y = float(point3d[1])
            pos_z = float(point3d[2])

            camera_point = PointStamped()
            camera_point.header.frame_id = "base_link"  # Remove leading slash
            camera_point.header.stamp = self.pointcloud_data.header.stamp
            
            camera_point.point.x = float(pos_x)
            camera_point.point.y = float(pos_y)
            camera_point.point.z = float(pos_z)
            
            # Use the pointcloud timestamp for transform lookup
            time_now = rclpy.time.Time()
            timeout = Duration(seconds=0.1)
            
            try:
                # Option 1: Use current time instead of pointcloud timestamp
                trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().error(f"Transform lookup failed: {str(e)}")
                return None, None, None
    
            point_in_map_frame = tfg.do_transform_point(camera_point, trans)

            map_x = point_in_map_frame.point.x
            map_y = point_in_map_frame.point.y
            map_z = point_in_map_frame.point.z
            self.map_y_temp = map_y
            self.map_x_temp = map_x
            return (map_x, map_y, map_z)
            
        except Exception as e:
            self.get_logger().error(f"Transform failed: {str(e)}")
            return None, None, None  # Return three None values

    def world_to_map_pixel(self, world_x, world_y, world_theta=0.2):
        ### Convert a real world location to a pixel in a numpy image
        ### Works only for theta=0
        if self.map_data["resolution"] is None:
            return 0, 0
        
        # Apply resolution, change of origin, and translation
        # x is the first coordinate, which in opencv (numpy) that is the matrix row - vertical
        x = int((world_x - self.map_data["origin"][0])/self.map_data["resolution"])
        y = int(self.map_data["height"] - (world_y - self.map_data["origin"][1])/self.map_data["resolution"])

        return x, y
    
    def map_callback(self, msg):
        # reshape the message vector back into a map
        self.map_np = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        # fix the direction of Y (origin at top for OpenCV, origin at bottom for ROS2)
        self.map_np = np.flipud(self.map_np)
        # change the colors so they match with the .pgm image
        self.map_np[self.map_np==0] = 127
        self.map_np[self.map_np==100] = 0
        # load the map parameters
        self.map_data["map_load_time"]=msg.info.map_load_time
        self.map_data["resolution"]=msg.info.resolution
        self.map_data["width"]=msg.info.width
        self.map_data["height"]=msg.info.height
        quat_list = [msg.info.origin.orientation.x,
                        msg.info.origin.orientation.y,
                        msg.info.origin.orientation.z,
                        msg.info.origin.orientation.w]
        self.map_data["origin"]=[msg.info.origin.position.x,
                                    msg.info.origin.position.y,
                                    tf_transformations.euler_from_quaternion(quat_list)[-1]]

    def check_top_position(self):
        if self.current_target and self.image_height:
            color_name, target_x, target_y = self.current_target

            if target_y <= 10:
                # Publish the Boolean flag indicating a ring is detected at the top
                flag_msg = Bool()
                flag_msg.data = True
                self.ring_detected_pub.publish(flag_msg)
                
                # Send audio message with the color
                if self.flag_response_ring:
                    audio_msg = String()
                    audio_msg.data = f"Ring detected with color {color_name}"
                    self.audio_pub.publish(audio_msg)
                    self.flag_response_ring = False

                # Reset target
                self.current_target = None
                self.nav_in_progress = False

    def start_navigation(self, center_x, y_point, color_name="unknown", color_bgr=(0,0,0)):

        # self.checking_frames += 1
        # if self.checking_frames < 30:
            # return
        # self.checking_frames = 0
        center_x, center_y, center_z = self.transform_point_to_world(center_x, y_point)
        
        if center_x is None or center_y is None or center_z is None or center_z > 0.75:
            return
                                
        offset_length = 0.5  # Length of the offset vector
        offsetLevo = np.array([center_x, center_y, center_z]) + self.normal_normalized_new * offset_length
        qx, qy, qz, qw = self.calculate_orientation_quaternion(float(offsetLevo[0]), float(offsetLevo[1]), 
                                                                float(center_x), float(center_y))
        offsetDesno = np.array([center_x, center_y, center_z]) + -self.normal_normalized_new * offset_length
        qx, qy, qz, qw = self.calculate_orientation_quaternion(float(offsetDesno[0]), float(offsetDesno[1]), 
                                                                float(center_x), float(center_y))
        
        print (f"Offset left: {offsetLevo}, Offset right: {offsetDesno}")
        
        px_levo, py_levo = self.world_to_map_pixel(offsetLevo[0], offsetLevo[1])
        px_desno, py_desno = self.world_to_map_pixel(offsetDesno[0], offsetDesno[1])
        if (0 <= px_levo < self.map_data["width"] and 0 <= py_levo < self.map_data["height"] and
            0 <= px_desno < self.map_data["width"] and 0 <= py_desno < self.map_data["height"]):
            
            cost_levo = self.costmap_np[py_levo, px_levo]
            cost_desno = self.costmap_np[py_desno, px_desno]

            if cost_levo > cost_desno:
                self.normal_normalized_new = -self.normal_normalized_new
        else:
            self.get_logger().warn("Offset points are outside the map boundaries.")

        offset = np.array([center_x, center_y, center_z]) + self.normal_normalized_new * 0.5
        qx, qy, qz, qw = self.calculate_orientation_quaternion(float(offset[0]), float(offset[1]), 
                                                                float(center_x), float(center_y))

        offset_float = (float(offset[0]), float(offset[1]), float(offset[2]))
        center_float = (float(center_x), float(center_y), float(center_z))
        normal_float = (float(self.normal_normalized_new[0]), float(self.normal_normalized_new[1]), float(self.normal_normalized_new[2]))

        self.publish_offset_marker(offset_float, center_float, normal_float, (qx, qy, qz, qw))
        circle_pose = (offset[0], offset[1], qx, qy, qz, qw)
        
        # Check if circle is already tracked
        circle_match = self.circle_idx_matching((center_x,center_y))
        
        if circle_match is True:  # Circle is new
            circle_id = len(self.detected_circles)
            self.detected_circles.append((center_x, center_y))
            self.circle_frame_counts[circle_id] = 1
            self.circle_positions[circle_id] = [circle_pose]  # Initialize with first pose
            self.get_logger().info(f"Tracking new circle ID {circle_id} at ({circle_pose[0]:.2f}, {circle_pose[1]:.2f}) with color {color_name}")
        elif circle_match >= 0:  # Circle exists, increment frame count
            circle_id = circle_match
            self.circle_frame_counts[circle_id] += 1
            self.circle_positions[circle_id].append(circle_pose)

            # Check if we've seen this circle enough times to confirm it
            if self.circle_frame_counts[circle_id] >= self.frames_to_confirm or (self.map_y_temp > 4.4 and self.map_y_temp < 5.5 and self.map_x_temp < -2.5 and self.map_x_temp > -3.5 and self.circle_frame_counts[circle_id] >= self.frames_to_confirm - 16):
                # Check if this circle is already confirmed
                print(f"map_y : {self.map_y_temp}")
                already_confirmed = False
                for confirmed in self.confirmed_circles:
                    if confirmed['id'] == circle_id:
                        confirmed['color'] = color_name
                        confirmed['color_bgr'] = color_bgr
                        already_confirmed = True
                        break
            # elif self.map_y_temp > 3.1 and self.circle_frame_counts[circle_id] >= self.frames_to_confirm - 11:
            #     # Check if this circle is already confirmed
            #     print("DAAAAAAAAAAAAA")
            #     already_confirmed = False
            #     for confirmed in self.confirmed_circles:
            #         if confirmed['id'] == circle_id:
            #             confirmed['color'] = color_name
            #             confirmed['color_bgr'] = color_bgr
            #             already_confirmed = True
            #             break

                if not already_confirmed:
                    positions = np.array(self.circle_positions[circle_id])  # shape (N, 3)
                    pos_data = positions[:, :3]  # x, y, z positions (offset positions)
                    quat_data = positions[:, 3:]

                    if len(pos_data) >= 4:
                        # DBSCAN parameters - adjust based on your detection accuracy
                        eps_distance = 0.15  # 15cm tolerance for clustering
                        min_cluster_size = max(3, int(len(positions) * 0.25))  # At least 25% of frames must agree
                        
                        clustering = DBSCAN(eps=eps_distance, min_samples=min_cluster_size).fit(pos_data)
                        labels = clustering.labels_
                        
                        # Find the largest cluster (ignore noise points labeled as -1)
                        valid_labels = labels[labels >= 0]
                        if len(valid_labels) > 0:
                            unique_labels, counts = np.unique(valid_labels, return_counts=True)
                            largest_cluster_label = unique_labels[np.argmax(counts)]
                            cluster_mask = labels == largest_cluster_label
                            
                            # Use only the largest cluster (removes outliers)
                            pos_data_clean = pos_data[cluster_mask]
                            quat_data_clean = quat_data[cluster_mask]
                            
                            outliers_removed = len(positions) - len(pos_data_clean)
                            self.get_logger().info(f"Circle {circle_id}: Used {len(pos_data_clean)} points, filtered out {outliers_removed} outliers")
                            
                            # Calculate mean of the clean data
                            avg_position = np.mean(pos_data_clean, axis=0)
                            avg_quaternion = np.mean(quat_data_clean, axis=0)
                            
                            avg_pose = tuple(np.concatenate([avg_position, avg_quaternion]).tolist())
                        else:
                            # Fallback: all points are noise, use simple mean
                            self.get_logger().warn(f"Circle {circle_id}: All points detected as outliers, using simple mean")
                            avg_position = np.mean(pos_data, axis=0)
                            avg_quaternion = np.mean(quat_data, axis=0)
                            avg_pose = tuple(np.concatenate([avg_position, avg_quaternion]).tolist())
                    else:
                        # Fallback: too few points for clustering, use simple mean
                        avg_position = np.mean(pos_data, axis=0)
                        avg_quaternion = np.mean(quat_data, axis=0)
                        avg_pose = tuple(np.concatenate([avg_position, avg_quaternion]).tolist())

                    self.get_logger().info(f"Circle ID {circle_id} confirmed after {self.circle_frame_counts[circle_id]} frames")
                    self.confirmed_circles.append({
                        'id': circle_id,
                        'position': avg_pose,  # <--- Use the clustered/average pose!
                        'color': color_name,
                        'color_bgr': color_bgr,
                        'published': False
                    })
                    print(f"NEW POSE PUBLISHED: {avg_pose} AND ID: {circle_id}")
                    # Create a persistent marker for the confirmed circle
                    self.create_confirmed_circle_marker(circle_id, (center_x, center_y, center_z), color_name, color_bgr)
                    # Optionally clear history
                    # del self.circle_positions[circle_id]
        
    def create_confirmed_circle_marker(self, circle_id, circle_pos, color_name="unknown", color_bgr=(0,0,0)):
        """Create a MarkerArray for a confirmed circle and its approaching point"""
        marker_array = MarkerArray()

        # Marker for the ring position (sphere)
        ring_marker = Marker()
        ring_marker.header.frame_id = "map"
        ring_marker.header.stamp = self.get_clock().now().to_msg()
        ring_marker.id = 1000 + circle_id  # Unique ID for ring marker
        ring_marker.type = Marker.SPHERE
        ring_marker.action = Marker.ADD
        
        # Set ring marker size
        ring_marker.scale.x = 0.3  # Diameter
        ring_marker.scale.y = 0.3  # Diameter
        ring_marker.scale.z = 0.3  # Height
        
        # Set marker color based on detected color (with fallback to yellow)
        if color_name in ["red", "green", "blue", "black"]:
            b, g, r = color_bgr
            ring_marker.color.r = float(r)/255.0
            ring_marker.color.g = float(g)/255.0
            ring_marker.color.b = float(b)/255.0
        else:
            ring_marker.color.r = 1.0
            ring_marker.color.g = 1.0
            ring_marker.color.b = 0.0
        
        ring_marker.color.a = 1.0  # Full opacity
        
        # Set ring marker position
        ring_marker.pose.position.x = circle_pos[0]
        ring_marker.pose.position.y = circle_pos[1]
        ring_marker.pose.position.z = circle_pos[2]
        
        # Set marker lifetime (0 for persistent)
        ring_marker.lifetime.sec = 0
        
        # Calculate approaching point (offset using normal)
        offset_length = 0.5  # Same as in start_navigation
        offset = np.array([circle_pos[0], circle_pos[1], circle_pos[2]]) + self.normal_normalized_new * offset_length
        qx, qy, qz, qw = self.calculate_orientation_quaternion(
            float(offset[0]), float(offset[1]), 
            float(circle_pos[0]), float(circle_pos[1])
        )
        
        # Marker for the approaching point (arrow)
        approach_marker = Marker()
        approach_marker.header.frame_id = "map"
        approach_marker.header.stamp = self.get_clock().now().to_msg()
        approach_marker.id = 2000 + circle_id  # Unique ID for approach marker
        approach_marker.type = Marker.ARROW
        approach_marker.action = Marker.ADD
        
        # Set approach marker size
        approach_marker.scale.x = 0.5  # Arrow length
        approach_marker.scale.y = 0.1  # Arrow width
        approach_marker.scale.z = 0.2  # Arrow height
        
        # Set approach marker color (red to distinguish from ring)
        approach_marker.color.r = 0.8
        approach_marker.color.g = 0.0
        approach_marker.color.b = 0.0
        approach_marker.color.a = 1.0
        
        # Set approach marker position and orientation
        approach_marker.pose.position.x = float(offset[0])
        approach_marker.pose.position.y = float(offset[1])
        approach_marker.pose.position.z = float(offset[2])
        approach_marker.pose.orientation.x = qx
        approach_marker.pose.orientation.y = qy
        approach_marker.pose.orientation.z = qz
        approach_marker.pose.orientation.w = qw
        
        # Set marker lifetime (0 for persistent)
        approach_marker.lifetime.sec = 0
        
        # Add both markers to the MarkerArray
        marker_array.markers.append(ring_marker)
        marker_array.markers.append(approach_marker)
        
        # Publish the MarkerArray
        self.marker_array_pub.publish(marker_array)
    
    # # Add a text marker with the color name
    # text_marker = Marker()
    # text_marker.header.frame_id = "map"
    # text_marker.header.stamp = self.get_clock().now().to_msg()
    # text_marker.id = 2000 + circle_id  # Different ID range for text markers
    # text_marker.type = Marker.TEXT_VIEW_FACING
    # text_marker.action = Marker.ADD
    
    # # Set text marker size and color
    # text_marker.scale.z = 0.15  # Text height
    # text_marker.color.r = 1.0
    # text_marker.color.g = 1.0
    # text_marker.color.b = 1.0
    # text_marker.color.a = 1.0
    
    # # Set text content and position
    # text_marker.text = color_name.upper()
    # text_marker.pose.position.x = circle_pos[0]
    # text_marker.pose.position.y = circle_pos[1]
    # text_marker.pose.position.z = 0.2  # Position above the cylinder
    
    # # Set marker lifetime
    # text_marker.lifetime.sec = 0
    
    # # Publish the text marker
    # self.marker_pub.publish(text_marker)

    def publish_confirmed_circles(self):
        """Publish confirmed circles that haven't been published yet"""
        if not self.confirmed_circles:
            return
            
        for circle in self.confirmed_circles:
            if not circle.get('published', False):
                # Create a PoseStamped message for the circle
                goal_pose = PoseStamped()
                goal_pose.header.frame_id = 'map'
                goal_pose.header.stamp = self.get_clock().now().to_msg()
                
                # Set position
                goal_pose.pose.position.x = circle['position'][0]
                goal_pose.pose.position.y = circle['position'][1]
                goal_pose.pose.position.z = float(circle.get('id', 0))
                
                # Store the circle ID in quaternion x component
                # This is a hack but allows us to send ID without creating custom messages
                goal_pose.pose.orientation.x = circle['position'][2]  # Use x as a placeholder
                goal_pose.pose.orientation.y = circle['position'][3]  # Use z as a placeholder
                goal_pose.pose.orientation.z = circle['position'][4]
                goal_pose.pose.orientation.w = circle['position'][5]
                
                # Publish the circle goal with embedded ID
                self.circle_goal_pub.publish(goal_pose)
                
                # Add debug log with clear ID info
                self.get_logger().info(f"Publishing circle ID {circle.get('id', 0)} to /circle_navigation_goal topic")
                
                # Publish the color information
                color_msg = String()
                color_info = f"{circle.get('id', 0)}:{circle.get('color', 'unknown')}"
                color_msg.data = color_info
                self.get_logger().info(f"Publishing color '{color_info}' to /circle_color topic")
                self.circle_color_pub.publish(color_msg)
                
                # Mark as published
                circle['published'] = True
                
                self.get_logger().info(f"Published circle ID {circle.get('id', 0)} with color {circle.get('color', 'unknown')} at ({circle['position'][0]:.2f}, {circle['position'][1]:.2f})")
                
                # Only publish one circle at a time to avoid confusion
                break
            elif circle.get('published', True):
                # Continue publishing color information even after the position is published
                # This ensures the robot_commander always has the latest color information
                color_msg = String()
                
                # Add circle ID to the message so robot_commander can match colors to positions
                color_info = f"{circle.get('id', 0)}:{circle.get('color', 'unknown')}"
                color_msg.data = color_info
                
                self.circle_color_pub.publish(color_msg)

    def circle_idx_matching(self, circle_pose):
        """
        Check if a circle position matches any already detected circle
        
        Args:
            circle_pose: (x, y) tuple of circle position
            
        Returns:
            True if it's a new circle
            int >= 0 (index) if it matches an existing circle
            False if there's an error
        """
        try:
            for circle_idx, circle_data in enumerate(self.detected_circles):
                distance = math.sqrt((circle_pose[0] - circle_data[0])**2 + 
                                    (circle_pose[1] - circle_data[1])**2)
                
                if distance < 1:  # Same threshold as in detect_people_normal.py
                    return circle_idx  # Return the matching circle index
            
            return True  # No match found, it's a new circle
                
        except Exception as e:
            self.get_logger().error(f"Error in circle_idx_matching: {e}")
            return False

def main():
    rclpy.init(args=None)
    rd_node = RingDetector()
    rclpy.spin(rd_node)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()