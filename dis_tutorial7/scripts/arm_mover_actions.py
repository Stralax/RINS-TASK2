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
        self.image_save_folder = os.path.expanduser("~/RINS-TASK2/img")
        os.makedirs(self.image_save_folder, exist_ok=True)
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_sub = self.create_subscription(
            Image,
            '/top_camera/rgb/preview/image_raw', 
            self.image_callback,
            10
        )

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.robot_base_frame = "base_link"
        self.camera_frame = "top_camera_rgb_camera_optical_frame"
        self.world_frame = "map"

        # Robot arm properties
        self.link_lengths = [0.1, 0.2, 0.2, 0.1]  # Adjust these based on your robot's dimensions

        # State variables
        self.new_command_arrived = False
        self.executing_command = False
        
        # Predefined positions
        self.joint_names = ['arm_base_joint', 'arm_shoulder_joint', 'arm_elbow_joint', 'arm_wrist_joint']
        self.arm_poses = {
            'look_for_parking': [0., 0.4, 1.5, 1.2],
            'look_for_bridge': [0., 0.4, 0.3, 2.0],
            'look_for_qr': [0., 0.6, 0.5, 2.0],
            'garage': [0., -0.45, 2.8, -0.8],
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
        
        # Calculate look-at angles
        joint_positions = self.calculate_look_at_angles(msg.x, msg.y, msg.z)
        
        if joint_positions is not None:
            self.current_command = f"manual:{joint_positions}"
            self.new_command_arrived = True
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
            self.save_camera_image()

        self.executing_command = False

    def save_camera_image(self):
        """Save the current camera image to disk"""
        if self.latest_image is None:
            self.get_logger().warn("No image received yet, cannot save picture.")
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
            now = datetime.datetime.now()
            filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
            full_path = os.path.join(self.image_save_folder, filename)
            cv2.imwrite(full_path, cv_image)
            self.get_logger().info(f"Saved image as {full_path}")
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
            # Get current camera transform in world frame
            camera_tf = self.get_camera_transform()
            if camera_tf is None:
                self.get_logger().error("Failed to get camera transform")
                return None

            # Get camera position and orientation
            cam_x = camera_tf.transform.translation.x
            cam_y = camera_tf.transform.translation.y
            cam_z = camera_tf.transform.translation.z
            
            # Debug output
            self.get_logger().info(f"Camera position: x={cam_x:.2f}, y={cam_y:.2f}, z={cam_z:.2f}")
            self.get_logger().info(f"Target position: x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}")

            # Vector from camera to target
            dx = target_x - cam_x
            dy = target_y - cam_y
            dz = target_z - cam_z
            
            # Debug output
            self.get_logger().info(f"Vector to target: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")

            # 1. Calculate base joint rotation (pan)
            pan = math.atan2(dy, dx)
            
            # 2. Calculate shoulder joint (tilt)
            distance_xy = math.sqrt(dx**2 + dy**2)
            tilt = math.atan2(dz, distance_xy)
            
            # 3. Keep elbow and wrist at reasonable fixed angles
            elbow = 0.5  # Slightly bent
            wrist = 1.2  # Slightly angled down
            
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



