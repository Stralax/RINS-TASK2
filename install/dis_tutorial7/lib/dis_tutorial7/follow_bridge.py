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
import os
from fpdf import FPDF  # For PDF generation
from datetime import datetime

class BridgeSegmenter:
    def __init__(self, node):
        self.node = node
        
        # Bridge color range (HSV)
        self.bridge_lower = np.array([15, 50, 50])  # Adjust for your bridge
        self.bridge_upper = np.array([35, 255, 255])
        
        # Enhanced grass detection parameters
        # Primary grass color range (HSV)
        self.grass_lower = np.array([30, 40, 40])   # Wider green color range for better detection
        self.grass_upper = np.array([90, 255, 255])  # Increased upper bound to catch more variations
        
        # Secondary grass color range (yellowish-green variants)
        self.grass_lower2 = np.array([25, 30, 40])
        self.grass_upper2 = np.array([40, 255, 255])
        
        # Texture parameters for grass detection
        self.texture_kernel_size = 5
        self.texture_threshold = 20  # Threshold for texture variation
        
        # Morphological parameters
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        self.grass_kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.grass_kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        
        # ROI parameters
        self.roi_bottom = 0.7  # Focus on bottom 70% of image

    def detect_texture(self, gray_img):
        """Detect texture variation in the image using Laplacian"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # Apply Laplacian filter to detect edges/texture
        laplacian = cv2.Laplacian(blurred, cv2.CV_8U, ksize=self.texture_kernel_size)
        
        # Threshold the Laplacian to get high texture areas
        _, texture_mask = cv2.threshold(laplacian, self.texture_threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up the texture mask with morphology
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_OPEN, self.grass_kernel_open)
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, self.grass_kernel_close)
        
        return texture_mask

    def segment(self, image):
        """Enhanced bridge detection with improved grass exclusion for textured areas"""
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Detect bridge color
            bridge_mask = cv2.inRange(hsv, self.bridge_lower, self.bridge_upper)
            
            # 2. Enhanced grass detection using color and texture
            # 2.1. Primary color detection for grass
            grass_mask1 = cv2.inRange(hsv, self.grass_lower, self.grass_upper)
            
            # 2.2. Secondary color detection for grass (yellowish variants)
            grass_mask2 = cv2.inRange(hsv, self.grass_lower2, self.grass_upper2)
            
            # 2.3. Combine both color masks
            grass_color_mask = cv2.bitwise_or(grass_mask1, grass_mask2)
            
            # 2.4. Detect texture-rich areas that could be grass
            texture_mask = self.detect_texture(gray)
            
            # 2.5. Combine color and texture info to get final grass mask
            # Consider an area as grass if it has both green color AND texture
            # OR if it has very strong green signal
            strong_green = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([70, 255, 255]))
            texture_and_color = cv2.bitwise_and(texture_mask, grass_color_mask)
            grass_mask = cv2.bitwise_or(texture_and_color, strong_green)
            
            # 2.6. Apply morphological operations to clean up the grass mask
            grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, self.grass_kernel_open)
            grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, self.grass_kernel_close)
            
            # 3. Create inverse grass mask (non-grass areas)
            no_grass = cv2.bitwise_not(grass_mask)
            
            # 4. Combine bridge detection with grass exclusion
            combined = cv2.bitwise_and(bridge_mask, no_grass)
            
            # 5. Apply ROI
            h, w = combined.shape
            roi = np.zeros_like(combined)
            roi[int(h*(1-self.roi_bottom)):h, :] = 255
            masked = cv2.bitwise_and(combined, roi)
            
            # 6. Morphological processing for final bridge mask
            cleaned = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self.kernel_open)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Store the last grass mask for visualization
            self.last_grass_mask = grass_mask
            
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

        # Image saving setup
        self.image_save_folder = os.path.expanduser("/home/beta/Desktop/RINS-TASK2/img/")
        os.makedirs(self.image_save_folder, exist_ok=True)
        self.should_save_image = False
        self.images_saved = False  # Flag to track if we've already saved images

    def create_images_pdf(self):
        """Create a PDF file from all images in the /img folder"""
        try:
            # Get list of all image files in the directory
            image_files = [f for f in os.listdir(self.image_save_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                self.get_logger().warn("No images found to create PDF")
                return

            # Sort images by name (which includes timestamp)
            image_files.sort()
            
            # Create PDF
            pdf = FPDF()
            
            for img_file in image_files:
                img_path = os.path.join(self.image_save_folder, img_file)
                
                # Add a page for each image
                pdf.add_page()
                
                # Add image title (filename without extension)
                title = os.path.splitext(img_file)[0]
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=title, ln=1, align='C')
                
                # Add the image
                pdf.image(img_path, x=10, y=20, w=180)
            
            # Save PDF with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pdf_filename = os.path.join(self.image_save_folder, f"Birds_catalog.pdf")
            pdf.output(pdf_filename)
            
            self.get_logger().info(f"Created PDF with images: {pdf_filename}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to create PDF: {str(e)}")


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
        """Detect the red cross in the image with improved precision."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for red (combining two ranges)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological clean-up for better shape detection
        kernel_open = np.ones((3,3), np.uint8)
        kernel_close = np.ones((9,9), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create debug visualization image
        debug_img = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(debug_img, "Red Cross Detection", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if not contours:
            cv2.imshow("Red Cross Detection", debug_img)
            cv2.waitKey(1)
            return False, None, 0

        # Find the largest red object
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Filter by minimum area
        if area < 150:
            cv2.imshow("Red Cross Detection", debug_img)
            cv2.waitKey(1)
            return False, None, 0
            
        # Get the center using moments (most reliable for finding true center)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            cv2.imshow("Red Cross Detection", debug_img)
            cv2.waitKey(1)
            return False, None, 0
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Draw contour and center on debug image
        cv2.drawContours(debug_img, [largest], -1, (0, 255, 0), 2)
        
        # Draw the center point
        cv2.circle(debug_img, (cx, cy), 10, (0, 0, 255), -1)
        
        # Draw crosshairs on center point
        cv2.line(debug_img, (cx - 20, cy), (cx + 20, cy), (255, 255, 0), 2)
        cv2.line(debug_img, (cx, cy - 20), (cx, cy + 20), (255, 255, 0), 2)
        
        # Show target position (image center)
        h, w = debug_img.shape[:2]
        center_x = w // 2
        center_y = h // 2
        cv2.circle(debug_img, (center_x, center_y), 10, (255, 0, 255), 2)
        
        # Draw line from image center to red cross center
        cv2.line(debug_img, (center_x, center_y), (cx, cy), (0, 255, 255), 2)
        
        # Display error values
        error_x = cx - center_x
        error_y = cy - center_y
        cv2.putText(debug_img, f"Error X: {error_x}px", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Error Y: {error_y}px", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Check if it's likely a cross shape (optional shape validation)
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = float(w) / float(h) if h > 0 else 0
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if 0.7 <= aspect_ratio <= 1.3:  # Cross should be roughly square in bounding box
            # Likely a cross - this is what we're looking for
            cv2.putText(debug_img, "CROSS SHAPE DETECTED", (10, 120), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.get_logger().info(f"Found red cross at ({cx},{cy}), area={area}")
        else:
            cv2.putText(debug_img, f"Aspect ratio: {aspect_ratio:.2f}", (10, 120), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Display the debug window
        cv2.imshow("Red Cross Detection", debug_img)
        cv2.waitKey(1)
        return True, (cx, cy), area


    def image_callback(self, msg):
        if not self.at_start_point or self.task_complete:
            return
            
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_width = cv_image.shape[1]
            image_height = cv_image.shape[0]

            # --- RED CROSS DETECTION AND PARKING LOGIC ---
            found_red, red_center, red_area = self.detect_red_spot(cv_image)
            # --- ADD LOGS TO DEBUG found_red ---
            self.get_logger().info("Starting red cross detection...")
            self.get_logger().info(f"Red cross detection result: found_red={found_red}, red_center={red_center}, red_area={red_area}")

            if found_red:
                self.get_logger().info("Red cross detected. Proceeding with movement logic.")
                self.send_arm_command("look_for_red_cross")
            else:
                self.get_logger().info("Red cross not detected. Continuing bridge following.")

            if found_red:
                # We've detected the red cross - switch to precise parking mode
                self.going_to_red = True
                self.red_spot_center = red_center
                self.red_spot_area = red_area
                
                # --- IMPROVED PRECISE PARKING STRATEGY ---
                # Initialize cmd as an instance of Twist
                cmd = Twist()
                
                # Calculate errors from the robot's center to the red cross center
                robot_center_x = self.image_width // 2  # Assuming robot's center aligns with image center
                robot_center_y = image_height // 2
                error_x = self.red_spot_center[0] - robot_center_x
                error_y = self.red_spot_center[1] - robot_center_y

                # Log detailed positioning information
                self.get_logger().info(f"RED CROSS: pos=({red_center[0]}, {red_center[1]}), " +
                                      f"errors=({error_x}, {error_y}), area={red_area:.1f}")

                # Define stricter thresholds for considering the robot centered
                centered_x_threshold = 3  # Must be within 3 pixels horizontally (very strict)
                centered_y_threshold = 3  # Must be within 3 pixels vertically (very strict)

                # Check if the robot's center is precisely aligned with the red cross
                if abs(error_x) < centered_x_threshold and abs(error_y) < centered_y_threshold:
                    self.task_complete = True
                    self.create_images_pdf()
                    self.get_logger().info("Robot is centered on the red cross. Task complete!")
                    return

                # Adjust motion control for precise positioning
                angular_kp = 0.003  # Base proportional gain for rotation
                linear_kp = 0.002  # Base proportional gain for linear motion

                # Increase gains for larger errors
                if abs(error_x) > 30:
                    angular_kp = 0.005
                elif abs(error_x) < 10:
                    angular_kp = 0.002

                if abs(error_y) > 30:
                    linear_kp = 0.003
                elif abs(error_y) < 10:
                    linear_kp = 0.001

                # Calculate angular and linear speeds
                angular_speed = min(0.15, abs(error_x) * angular_kp)
                linear_speed = min(0.08, abs(error_y) * linear_kp)

                # Determine direction for rotation and movement
                cmd.angular.z = -np.sign(error_x) * angular_speed
                cmd.linear.x = -np.sign(error_y) * linear_speed

                # Apply near-zero deadband for fine adjustments
                if abs(error_x) < 2:
                    cmd.angular.z = 0.0

                if abs(error_y) < 2:
                    cmd.linear.x = 0.0

                # Send the motion command
                self.cmd_vel_pub.publish(cmd)
                return
            else:
                self.going_to_red = False
                self.red_spot_center = None
                self.red_spot_area = 0



            # If not going to red, proceed with bridge following
            mask = self.segmenter.segment(cv_image)
            self.bridge_center = self.find_bridge_center(mask)  # Only pass the mask parameter
            
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
        
        # Get the enhanced grass mask for visualization
        if hasattr(self.segmenter, 'last_grass_mask'):
            grass_mask = self.segmenter.last_grass_mask
        else:
            # Fall back to simple HSV-based detection if texture detection hasn't run yet
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            grass_mask1 = cv2.inRange(hsv, self.segmenter.grass_lower, self.segmenter.grass_upper)
            grass_mask2 = cv2.inRange(hsv, self.segmenter.grass_lower2, self.segmenter.grass_upper2)
            grass_mask = cv2.bitwise_or(grass_mask1, grass_mask2)
        
        # Create a semi-transparent overlay for grass
        overlay = debug_img.copy()
        overlay[grass_mask > 0] = [0, 0, 255]  # Mark grass with red color
        
        # Apply semi-transparency
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, debug_img, 1 - alpha, 0, debug_img)
        
        # Highlight bridge in green (fully opaque)
        debug_img[mask > 0] = [0, 255, 0]  # Mark bridge green
        
        # Add text to show what's being detected
        cv2.putText(debug_img, 
                  "GREEN: Bridge Path", 
                  (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(debug_img, 
                  "RED: Grass/Obstacles", 
                  (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show bridge center point
        if self.bridge_center:
            cv2.circle(debug_img, self.bridge_center, 10, (255, 0, 0), -1)
            cv2.putText(debug_img, 
                      "Navigation Target", 
                      (self.bridge_center[0] - 70, self.bridge_center[1] - 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Show red spot center
        if self.red_spot_center:
            cv2.circle(debug_img, self.red_spot_center, 10, (0, 255, 255), 2)
            cv2.putText(debug_img, 
                      "RED CROSS", 
                      (self.red_spot_center[0] - 40, self.red_spot_center[1] - 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Enhanced Bridge Detection", debug_img)
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



