#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from nav2_msgs.action import NavigateToPose
from tf_transformations import quaternion_from_euler
from std_msgs.msg import String
from action_msgs.msg import GoalStatus

class Nav2Commander(Node):
    def __init__(self):
        super().__init__('nav2_commander')
        # Navigation client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Arm command publisher (QoS profile for reliability)
        self.arm_publisher = self.create_publisher(
            String, 
            '/arm_command', 
            qos_profile=rclpy.qos.qos_profile_system_default
        )
        
    def calculate_yaw_to_point(self, current_pos, target_pos):
        """Calculate yaw angle to face target point"""
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        return math.atan2(dy, dx)
    
    def send_arm_command(self, command):
        """Publish command to arm"""
        msg = String()
        msg.data = command
        self.arm_publisher.publish(msg)
        self.get_logger().info(f"Sent arm command: {command}")
        # Ensure message is sent before continuing
        rclpy.spin_once(self, timeout_sec=0.1)
    
    def go_to_point_and_face(self, goal_x, goal_y, face_x, face_y):
        """Navigate to (goal_x,goal_y) oriented toward (face_x,face_y)"""
        # Create goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set goal position
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        
        # Calculate orientation to face the target point
        current_pos = Point(x=goal_x, y=goal_y)
        face_pos = Point(x=face_x, y=face_y)
        yaw = self.calculate_yaw_to_point(current_pos, face_pos)
        
        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, yaw)
        goal_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        
        # Send navigation goal
        self.get_logger().info(f"Navigating to ({goal_x}, {goal_y}) facing ({face_x}, {face_y})")
        goal_msg = NavigateToPose.Goal(pose=goal_pose)
        
        # Wait for server and send goal
        self.nav_to_pose_client.wait_for_server()
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected!")
            return False
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        if result_future.result().status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Navigation successful!")
            self.send_arm_command("look_for_bridge")
            return True
        else:
            self.get_logger().error("Navigation failed!")
            return False

def main(args=None):
    rclpy.init(args=args)
    commander = Nav2Commander()
    
    try:
        # Your specified coordinates
        success = commander.go_to_point_and_face(
            goal_x=0.0, 
            goal_y=-0.8, 
            face_x=0.0, 
            face_y=-0.9
        )
        
        if success:
            commander.get_logger().info("Mission completed successfully!")
        else:
            commander.get_logger().error("Mission failed!")
            
        # Ensure arm command is published before shutdown
        rclpy.spin_once(commander, timeout_sec=1.0)
        
    finally:
        commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()