#!/usr/bin/env python3

import rclpy
import math
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from tf_transformations import quaternion_from_euler
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import time

class BridgeSegmenter:
    def __init__(self, node):
        self.node = node
        
        # Bridge color range (HSV)
        self.bridge_lower = np.array([15, 50, 50])  # Adjust for your bridge
        self.bridge_upper = np.array([35, 255, 255])
        
        # Grass exclusion range (HSV)
        self.grass_lower = np.array([35, 50, 50])   # Green color range
        self.grass_upper = np.array([85, 255, 255])
        
        # Morphological parameters
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        
        # ROI parameters
        self.roi_bottom = 0.7  # Focus on bottom 70% of image

    def segment(self, image):
        """Bridge detection with grass exclusion"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 1. Detect bridge color
            bridge_mask = cv2.inRange(hsv, self.bridge_lower, self.bridge_upper)
            
            # 2. Detect and exclude grass
            grass_mask = cv2.inRange(hsv, self.grass_lower, self.grass_upper)
            no_grass = cv2.bitwise_not(grass_mask)
            
            # 3. Combine bridge detection with grass exclusion
            combined = cv2.bitwise_and(bridge_mask, no_grass)
            
            # 4. Apply ROI
            h, w = combined.shape
            roi = np.zeros_like(combined)
            roi[int(h*(1-self.roi_bottom)):h, :] = 255
            masked = cv2.bitwise_and(combined, roi)
            
            # 5. Morphological processing
            cleaned = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self.kernel_open)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel_close)
            
            return cleaned
            
        except Exception as e:
            self.node.get_logger().error(f"Segmentation failed: {str(e)}")
            return np.zeros(image.shape[:2], dtype=np.uint8)

    def find_bridge_center(self, mask):
        """Find center with shape validation"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Find largest contour that meets bridge shape criteria
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Minimum bridge area
                continue
            # Check aspect ratio (bridge should be long and narrow)
            rect = cv2.minAreaRect(cnt)
            (_,_), (w,h), _ = rect
            aspect_ratio = max(w,h)/max(min(w,h), 1)
            if aspect_ratio > 2.5:
                valid_contours.append(cnt)
        
        if not valid_contours:
            return None
            
        largest = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] > 0 else None

class BridgeFollower(Node):
    def __init__(self):
        super().__init__('bridge_follower')
        
        # Initialize segmenter with reference to this node
        self.segmenter = BridgeSegmenter(self)

        # Navigation client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_command_pub = self.create_publisher(String, '/arm_command', 10)
        
        # Vision setup
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/top_camera/rgb/preview/image_raw',
            self.image_callback,
            10
        )
        
        # Control parameters
        self.Kp = 0.01
        self.speed = 0.15
        self.image_width = 640
        self.bridge_center = None
        self.at_start_point = False

        # Red detection parameters
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        #self.red_found_and_stopped = False

        self.going_to_red = False
        self.red_spot_center = None
        self.red_spot_area = 0
        self.task_complete = False


    def go_to_start_point(self, x, y, face_x, face_y):
        """Navigate to initial position before bridge following"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        
        # Calculate orientation to face
        yaw = math.atan2(face_y - y, face_x - x)
        q = quaternion_from_euler(0, 0, yaw)
        goal_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        
        # Send navigation goal
        self.get_logger().info(f"Going to start point ({x}, {y})")
        goal_msg = NavigateToPose.Goal(pose=goal_pose)
        
        self.nav_to_pose_client.wait_for_server()
        
        # Send goal and wait for it to be accepted
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected!")
            return False
        
        # Wait for the navigation to complete
        self.get_logger().info("Waiting for navigation to complete...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        if result_future.result().status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Navigation successful!")
            self.send_arm_command("look_for_bridge")
            self.at_start_point = True
            return True
        else:
            self.get_logger().error("Failed to reach start point!")
            return False

    def detect_red_spot(self, image):
        """Detect the largest red area and its center in the image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological clean-up (optional)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, 0

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 150:  # Ignore tiny red spots
            return False, None, 0

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return False, None, 0
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return True, (cx, cy), area


    def image_callback(self, msg):
        if not self.at_start_point or self.task_complete:
            return
            
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_width = cv_image.shape[1]
            image_height = cv_image.shape[0]

            # --- RED SPOT LOGIC ---
            found_red, red_center, red_area = self.detect_red_spot(cv_image)
            if found_red:
                self.going_to_red = True
                self.red_spot_center = red_center
                self.red_spot_area = red_area
            else:
                self.going_to_red = False
                self.red_spot_center = None
                self.red_spot_area = 0

            if self.going_to_red and self.red_spot_center is not None:
                error_x = self.red_spot_center[0] - self.image_width // 2
                error_y = (image_height - self.red_spot_center[1])
                # If red spot is centered and close, move a bit forward then stop
                if abs(error_x) < 30 and error_y < 70:
                    self.get_logger().info("Arrived at red spot! Nudging forward before final stop.")
                    cmd = Twist()
                    cmd.linear.x = 3.0  # Forward speed (m/s), adjust as needed
                    cmd.angular.z = 0.0
                    self.cmd_vel_pub.publish(cmd)
                    time.sleep(6.0)  # Move forward for 0.7 seconds; tune as needed
                    self.stop_robot()
                    self.going_to_red = False
                    self.task_complete = True
                    return

                else:
                    cmd = Twist()
                    cmd.linear.x = 7.5 if error_y > 60 else 0.05
                    cmd.angular.z = -self.Kp * error_x
                    self.get_logger().info(f"Approaching red spot. Error_x: {error_x}, Error_y: {error_y}, Speed: {cmd.linear.x}, Turn: {cmd.angular.z}")
                    self.cmd_vel_pub.publish(cmd)
                    return

            # If not going to red, proceed with bridge following
            mask = self.segmenter.segment(cv_image)
            self.bridge_center = self.find_bridge_center(mask)
            
            if self.bridge_center is not None:
                self.follow_bridge()
                self.show_detection(cv_image, mask)
            else:
                self.get_logger().warn("Bridge not detected!")
                self.stop_robot()
                
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")
            self.stop_robot()


    def find_bridge_center(self, mask):
        """Find the center point of the bridge in the image"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
            
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def follow_bridge(self):
        """Generate control commands to follow the bridge"""
        error = self.bridge_center[0] - self.image_width // 2
        
        cmd = Twist()
        cmd.linear.x = self.speed + 0.30 # ADD SPEED
        cmd.angular.z = -self.Kp * error
        self.get_logger().info(f"Following bridge. Error: {error}, Speed: {cmd.linear.x}, Turn: {cmd.angular.z}")
        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    # def show_detection(self, image, mask):
    #     """Show detection results (for debugging)"""
    #     cv2.circle(image, self.bridge_center, 10, (0, 255, 0), -1)
    #     cv2.line(image, (self.image_width//2, 0), 
    #             (self.image_width//2, image.shape[0]), 
    #             (255, 0, 0), 2)
    #     cv2.line(image, (self.image_width//2, self.bridge_center[1]),
    #             self.bridge_center, (0, 0, 255), 2)
        
    #     cv2.imshow("Camera View", image)
    #     cv2.imshow("Bridge Mask", mask)
    #     cv2.waitKey(1)

    def show_detection(self, image, mask):
        debug_img = image.copy()
        
        # Highlight grass areas in red
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, self.segmenter.grass_lower, self.segmenter.grass_upper)
        debug_img[grass_mask > 0] = [0, 0, 255]  # Mark grass red
        
        # Highlight bridge in green
        debug_img[mask > 0] = [0, 255, 0]  # Mark bridge green
        
        # Show bridge center point
        if self.bridge_center:
            cv2.circle(debug_img, self.bridge_center, 10, (255, 0, 0), -1)
        # Show red spot center
        if self.red_spot_center:
            cv2.circle(debug_img, self.red_spot_center, 10, (0, 0, 255), 2)
        
        cv2.imshow("Detection Debug", debug_img)
        cv2.waitKey(1)


    def send_arm_command(self, command):
        msg = String()
        msg.data = command
        self.arm_command_pub.publish(msg)
        self.get_logger().info(f"Sent arm command: {command}")


def main(args=None):
    rclpy.init(args=args)
    follower = BridgeFollower()
    
    try:
        if follower.go_to_start_point(0.0, -0.8, 0.0, -0.9):
            follower.get_logger().info("Starting bridge following...")
            rclpy.spin(follower)
    except Exception as e:
        follower.get_logger().error(f"Error: {str(e)}")
    finally:
        follower.stop_robot()
        cv2.destroyAllWindows()
        follower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



