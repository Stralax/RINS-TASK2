#!/usr/bin/python3

import rclpy
import rclpy.duration
import math
from rclpy.node import Node

from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from action_msgs.msg import GoalStatus

from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import datetime
import os
import tf2_ros
from geometry_msgs.msg import TransformStamped


# primer uporabe za visino uzeti z: -99.3
# ros2 topic pub --once /target_point geometry_msgs/msg/Point "{x: 1.61, y: 0.0, z: -99.3}"





# QoS profile for reliable communication
qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class ArmMoverAction(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Image saving setup
        self.image_save_folder = os.path.expanduser("~/Desktop/RINS-TASK2/img")
        os.makedirs(self.image_save_folder, exist_ok=True)
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_sub = self.create_subscription(
            Image,
            '/top_camera/rgb/preview/image_raw', 
            self.image_callback,
            10
        )
        
        # Add a subscriber for bird name
        self.bird_name = None
        self.bird_name_sub = self.create_subscription(
            String,
            '/target_bird_name',
            self.bird_name_callback,
            10
        )
        self.should_save_image = False

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.robot_base_frame = "base_link"
        self.camera_frame = "top_camera_rgb_camera_optical_frame"
        self.world_frame = "odom"

        # Robot arm properties
        self.link_lengths = [0.1, 0.2, 0.2, 0.1]  # Adjust these based on your robot's dimensions

        # State variables
        self.new_command_arrived = False
        self.executing_command = False
        
        # Predefined positions
        self.joint_names = ['arm_base_joint', 'arm_shoulder_joint', 'arm_elbow_joint', 'arm_wrist_joint']
        self.arm_poses = {
            'look_for_parking': [0., 0.6, 1.5, 1.2],
            'look_for_bridge': [0., 0., 1.5, 1.0],
            'look_for_qr': [0., 0.6, 0.5, 2.0],
            'garage': [0., -0.45, 2.8, -0.8],
            'look_for_red_cross': [0., 0., 0.3 , 2.0],
            'up': [0., 0., 0., 0.],
            'manual': None
        }

        # Communication setup
        self.arm_command_sub = self.create_subscription(String, "/arm_command", self.arm_command_callback, 1)
        self.target_sub = self.create_subscription(Point, "/target_point", self.target_callback, 1)
        self.arm_position_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

        # Timer for periodic processing
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.get_logger().info("Initialized the Arm Mover node! Waiting for commands...")

    def bird_name_callback(self, msg):
        """Store the bird name for image saving"""
        self.bird_name = msg.data
        self.get_logger().info(f"Received bird name: {self.bird_name}")

    def image_callback(self, msg):
        """Store the latest camera image"""
        self.latest_image = msg

    def get_camera_position(self):
        """Get current camera position in world frame using TF"""
        try:
            # Request transform for current time (but allow latest if current not available)
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            
            pos = transform.transform.translation
            return [pos.x, pos.y, pos.z]
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            
            # Fallback to default position
            self.get_logger().warn("Using default camera position")
            return [0.0, 0.0, 0.5]  # Default fallback position

    def simple_inverse_kinematics(self, x, y, z):
        """
        Simple inverse kinematics for 4-DOF arm
        Returns joint positions or None if position is unreachable
        """
        try:
            # Base rotation (joint 0)
            theta_base = math.atan2(y, x)
            
            # Project onto vertical plane
            planar_distance = math.sqrt(x**2 + y**2)
            
            # Shoulder and elbow joints (joints 1-2)
            # Using simple geometric solution for 2-link planar arm
            l1 = self.link_lengths[1]  # Shoulder to elbow
            l2 = self.link_lengths[2]  # Elbow to wrist
            
            # Target in arm plane (x is forward, z is up)
            target_dist = math.sqrt(planar_distance**2 + z**2)
            
            # Check if target is reachable
            if target_dist > (l1 + l2) or target_dist < abs(l1 - l2):
                self.get_logger().warn(f"Target at distance {target_dist} is unreachable with links {l1}, {l2}")
                return None
                
            # Calculate angles
            cos_theta2 = (target_dist**2 - l1**2 - l2**2) / (2 * l1 * l2)
            theta2 = math.acos(cos_theta2)
            
            theta1 = math.atan2(z, planar_distance) - math.atan2(l2 * math.sin(theta2), 
                                                          l1 + l2 * math.cos(theta2))
            
            # Wrist joint (joint 3) - keep it simple for now
            theta_wrist = 0.0  # Or calculate based on desired end effector orientation
            
            return [theta_base, theta1, theta2, theta_wrist]
            
        except Exception as e:
            self.get_logger().error(f"IK calculation failed: {str(e)}")
            return None

    def target_callback(self, msg):
        """Callback for receiving target point in world coordinates"""
        self.get_logger().info(f"Received target point: x={msg.x}, y={msg.y}, z={msg.z}")
        
        # Set flag to save image only if z coordinate is -99.3
        self.should_save_image = (abs(msg.z - (-99.3)) < 0.1)  # Using small epsilon for float comparison
        
        # Calculate look-at angles
        joint_positions = self.calculate_look_at_angles(msg.x, msg.y, msg.z)
        
        if joint_positions is not None:
            self.current_command = f"manual:{joint_positions}"
            self.new_command_arrived = True
            #self.save_camera_image()
        else:
            self.get_logger().error("Failed to calculate joint positions for target point")

    def set_arm_position(self, command_string):
        """Send goal to position the arm"""
        self.executing_command = True

        while not self.arm_position_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("'Arm controller' action server not available, waiting...")

        point = JointTrajectoryPoint()

        if ':' in command_string:
            command_type, command_value = command_string.split(':', 1)
            
            if command_type == "manual":
                point.positions = eval(command_value)
            elif command_type == "coord":
                coords = eval(command_value)
                point.positions = self.calculate_look_at_angles(*coords) or [0., 0., 0., 0.]
        else:
            point.positions = self.arm_poses[command_string]
            
        point.time_from_start = rclpy.duration.Duration(seconds=3.).to_msg()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.goal_time_tolerance = rclpy.duration.Duration(seconds=3.).to_msg()
        goal_msg.trajectory.joint_names = self.joint_names
        goal_msg.trajectory.points.append(point)

        self.get_logger().info(f'Sending goal: {point.positions}')
        self.send_goal_future = self.arm_position_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_accepted_callback)

        self.new_command_arrived = False
    
    def goal_accepted_callback(self, future):
        """Callback when goal is accepted/rejected"""
        goal_handle = future.result()

        if goal_handle.accepted: 
            self.get_logger().info('Arm controller ACCEPTED the goal.')
            self.result_future = goal_handle.get_result_async()
            self.result_future.add_done_callback(self.get_result_callback)
        else:
            self.get_logger().error('Arm controller REJECTED the goal.')
            self.executing_command = False

    def get_result_callback(self, future):
        """Callback when goal completes"""
        status = future.result().status

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Arm controller says GOAL FAILED: {status}')
        else:
            self.get_logger().info(f'Arm controller says GOAL REACHED.')
            # Only save image if flag is set
            if self.should_save_image:
                self.save_camera_image()
                self.should_save_image = False  # Reset flag

        self.executing_command = False

    def save_camera_image(self):
        """Save the current camera image to disk using bird name if available"""
        if self.latest_image is None:
            self.get_logger().warn("No image received yet, cannot save picture.")
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
            
            if self.bird_name:
                # Use bird name in the filename
                # Clean up bird name to make it suitable for a filename
                clean_bird_name = self.bird_name
                # clean_bird_name = self.bird_name.replace(" ", "_").replace(".", "").replace("/", "")
                filename = f"{clean_bird_name}.jpg"  # Add timestamp to avoid overwrites
                self.get_logger().info(f"Saving image with bird name: {clean_bird_name}")
            else:
                # Fallback to date-based filename
                now = datetime.datetime.now()
                filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
                self.get_logger().info(f"No bird name available, using date-based filename")
                
            full_path = os.path.join(self.image_save_folder, filename)
            cv2.imwrite(full_path, cv_image)
            self.get_logger().info(f"Saved image as {full_path}")
            
            # Reset the bird name after saving the image
            self.bird_name = None
            
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")

    def timer_callback(self):
        """Periodic timer callback"""
        if self.new_command_arrived and not self.executing_command:
            self.set_arm_position(self.current_command)
            self.get_logger().info(f"Will set a new position for the arm joints: {self.current_command}")
            self.previous_command = self.current_command
            self.new_command_arrived = False

    def arm_command_callback(self, msg):
        self.get_logger().info(f"Received arm command: {msg.data}")
        """Callback for arm commands"""
        command_string = msg.data.strip().lower()
        
        # Check if command is a coordinate command
        if command_string.startswith("coord:"):
            coord_str = command_string.split("coord:")[1].strip()
            try:
                # Convert string to list of floats
                coords = eval(coord_str)  # e.g. "[0.3, 0.0, 0.5]"
                if len(coords) != 3:
                    self.get_logger().error("Coordinate command must have exactly 3 values.")
                    return

                x, y, z = coords
                self.get_logger().info(f"Received coordinate command: x={x}, y={y}, z={z}")

                # Compute joint positions via look-at function
                joint_positions = self.calculate_look_at_angles(x, y, z)
                if joint_positions is None:
                    self.get_logger().error("No solution found for given coordinates.")
                    return

                # Format as manual command string
                self.current_command = f"manual:{joint_positions}"
                self.new_command_arrived = True

                #if self.should_save_image:
                #self.save_camera_image()
                #    self.should_save_image = False  # Reset flag

                return

            except Exception as e:
                self.get_logger().error(f"Failed to parse coordinate command: {e}")
                return

        # Otherwise, treat as existing position command
        command_test = command_string.split(":")[0] if ":" in command_string else command_string
        if command_test not in list(self.arm_poses.keys()):
            self.get_logger().error(f"Invalid command: {command_string}")
            return

        self.current_command = command_string
        self.new_command_arrived = True
        self.get_logger().info(f"Got a new command for the arm configuration: {command_string}")


    def get_camera_transform(self):
        """Get current camera transform in world frame"""
        try:
            # Get transform from world to camera frame
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=1.0))
            
            # Debug output
            self.get_logger().info(f"Camera transform: "
                                f"x={transform.transform.translation.x:.2f}, "
                                f"y={transform.transform.translation.y:.2f}, "
                                f"z={transform.transform.translation.z:.2f}")
            return transform
            
        except tf2_ros.LookupException as e:
            self.get_logger().error(f"TF LookupException: {str(e)}")
        except tf2_ros.ConnectivityException as e:
            self.get_logger().error(f"TF ConnectivityException: {str(e)}")
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().error(f"TF ExtrapolationException: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {str(e)}")
        return None

    def calculate_look_at_angles(self, target_x, target_y, target_z):
        """
        Calculate joint angles to make camera look at target point in world frame
        Returns joint positions or None if calculation fails
        """
        try:
            # Get current transform from odom to base_link
            try:
                base_tf = self.tf_buffer.lookup_transform(
                    self.world_frame,
                    self.robot_base_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0))
            except Exception as e:
                self.get_logger().error(f"Failed to get base transform: {str(e)}")
                return None

            # Transform target point to base_link frame
            target_in_base = Point()
            target_in_base.x = target_x - base_tf.transform.translation.x
            target_in_base.y = target_y - base_tf.transform.translation.y
            target_in_base.z = target_z #- base_tf.transform.translation.z

            # Account for robot rotation (yaw only for simplicity)
            robot_yaw = math.atan2(
                2.0 * (base_tf.transform.rotation.w * base_tf.transform.rotation.z + 
                    base_tf.transform.rotation.x * base_tf.transform.rotation.y),
                1.0 - 2.0 * (base_tf.transform.rotation.y**2 + base_tf.transform.rotation.z**2))
            
            # Rotate target point to align with robot's forward direction
            dx = math.cos(robot_yaw) * target_in_base.x + math.sin(robot_yaw) * target_in_base.y
            dy = -math.sin(robot_yaw) * target_in_base.x + math.cos(robot_yaw) * target_in_base.y
            dz = target_in_base.z

            # Get current camera position relative to base
            try:
                cam_tf = self.tf_buffer.lookup_transform(
                    self.robot_base_frame,
                    self.camera_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0))
            except Exception as e:
                self.get_logger().error(f"Failed to get camera transform: {str(e)}")
                return None

            # Vector from camera to target (in base frame)
            dx_cam_to_target = -dx - cam_tf.transform.translation.x
            dy_cam_to_target = -dy - cam_tf.transform.translation.y
            dz_cam_to_target = dz - cam_tf.transform.translation.z

            # Debug output
            self.get_logger().info(f"Camera to target vector: dx={dx_cam_to_target:.2f}, dy={dy_cam_to_target:.2f}, dz={dz_cam_to_target:.2f}")

            # 1. Calculate base joint rotation (pan) - REVERSED DIRECTION
            pan = math.atan2(-dy_cam_to_target, -dx_cam_to_target)
            
            # 2. Calculate shoulder joint (tilt) - REVERSED DIRECTION
            distance_xy = math.sqrt(dx_cam_to_target**2 + dy_cam_to_target**2)
            tilt = math.atan2(-dz_cam_to_target, distance_xy)
            
            # 3. Calculate elbow to maintain camera orientation
            elbow = -1.9 #tilt * -1  # (1 ali -1) Changed sign #STELOVANJE
            
            # 4. Calculate wrist to keep camera level
            wrist = tilt #-elbow + tilt  # Adjusted calculation
            
            # Debug output
            self.get_logger().info(f"Calculated angles: pan={pan:.2f}, tilt={tilt:.2f}, elbow={elbow:.2f}, wrist={wrist:.2f}")
            
            return [pan, tilt, elbow, wrist]
            
        except Exception as e:
            self.get_logger().error(f"Look-at calculation failed: {str(e)}")
            return None

def main():
    rclpy.init(args=None)
    rd_node = ArmMoverAction()
    rclpy.spin(rd_node)

if __name__ == '__main__':
    main()



