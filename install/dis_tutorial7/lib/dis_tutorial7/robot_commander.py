#! /usr/bin/env python3
# Modified from Samsung Research America
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import time
import subprocess
import os
import numpy as np

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from nav_msgs.msg import Path
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import MarkerArray

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        
        # Person detection tracking
        self.detected_person_markers = None
        self.greeting_distance_threshold = 1.5  # meters
        self.greeted_faces = set()  # Keep track of which faces we've already greeted
        self.person_to_greet = None  # Will hold information about person we're approaching
        self.current_waypoint_idx = 0
        self.tts_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "speak.py")  # Use direct path from current script
        self.greeting_text = "Hello there!"
        self.current_task = None  # None, 'waypoint', or 'greeting'
 
        # ROS2 subscribers
        self.create_subscription(DockStatus,
                                 'dock_status',
                                 self._dockCallback,
                                 qos_profile_sensor_data)
        
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                              'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)
        
        # Subscribe to person markers
        self.people_marker_sub = self.create_subscription(
            MarkerArray,
            '/people_array',
            self._peopleMarkerCallback,
            10
        )
        
        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                                      'initialpose',
                                                      10)
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.get_logger().info(f"Robot commander has been initialized!")
        
    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()     

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        self.debug(f"goToPose: Goal accepted, result_future created: {self.result_future}")
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        self.debug(f"spin: Goal accepted, result_future created: {self.result_future}")
        return True
    
    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task in cancelTask.')
        if self.result_future:
            self.info("cancelTask: result_future exists, attempting to cancel goal.")
            try:
                self.info("cancelTask: Calling cancel_goal_async...")
                future = self.goal_handle.cancel_goal_async()
                
                # Add timeout to prevent hanging
                self.info("cancelTask: Waiting for cancel_goal_async to complete (with timeout)...")
                timeout_sec = 2.0
                # Just check if succeeded instead of checking for TimeoutException
                success = rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
                
                if not success:
                    self.error(f"cancelTask: cancel_goal_async timed out after {timeout_sec} seconds!")
                else:
                    try:
                        cancel_result = future.result()
                        self.info(f"cancelTask: cancel_goal_async completed with result: {cancel_result}")
                    except Exception as e:
                        self.error(f"cancelTask: Error getting cancel result: {e}")
                        
            except Exception as e:
                self.error(f"cancelTask: Exception occurred during cancellation: {e}")
                # Continue even if there was an error
                
            # Reset the variables regardless of success or failure
            self.info("cancelTask: Resetting result_future and goal_handle")
            self.result_future = None
            self.goal_handle = None
            self.info("cancelTask: result_future and goal_handle reset to None.")
        else:
            self.info("cancelTask: No result_future, nothing to cancel.")
        
        # Force a callback processing opportunity to flush events
        rclpy.spin_once(self, timeout_sec=0.1)
        self.info("cancelTask: Task cancellation procedure completed.")
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            self.debug("isTaskComplete: No result_future, task is complete.")
            return True
        
        try:
            rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
            
            # Make sure result_future is still valid (might have been reset by another callback)
            if not self.result_future:
                self.debug("isTaskComplete: result_future was reset during spin, task is complete.")
                return True
                
            if self.result_future.done():
                result = self.result_future.result()
                if result:
                    self.status = result.status
                    if self.status != GoalStatus.STATUS_SUCCEEDED:
                        self.debug(f'Task failed with status code: {self.status}')
                        return True
                    else:
                        self.debug('Task succeeded!')
                        return True
            else:
                # Timed out, still processing, not complete yet
                self.debug("isTaskComplete: result_future timed out, task not complete.")
                return False
        except Exception as e:
            self.error(f"isTaskComplete: Exception occurred: {e}")
            # If there's an error, consider the task complete to avoid getting stuck
            return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return
    
    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def _peopleMarkerCallback(self, msg: MarkerArray):
        """Handle incoming people markers"""
        self.detected_person_markers = msg
        self.info(f"Received people markers: {len(msg.markers)}")
        
        # Log the details of each marker
        for marker in msg.markers:
            self.debug(f"_peopleMarkerCallback: Received marker - NS: {marker.ns}, ID: {marker.id}, Pose: {marker.pose}")
    
    
    def sayGreeting(self, text=None):
        """Use text-to-speech to say greeting"""
        if text is None:
            text = self.greeting_text
            
        try:
            # Print to terminal
            self.info(f"Speaking: {text}")
            
            # Check if file exists before running
            if os.path.exists(self.tts_script_path):
                self.info(f"Running TTS script: {self.tts_script_path}")
                subprocess.Popen(["python3", self.tts_script_path, text])
            else:
                self.error(f"TTS script not found at: {self.tts_script_path}")
                # Just print the text as fallback
                print(f"ROBOT SAYS: {text}")
        except Exception as e:
            self.error(f"Error running TTS script: {e}")

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return
    
    def navigate_to_person(self, goal_pos):
        """Navigate to a person's goal position"""
        self.current_task = 'greeting'
        
        # Create the goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_pose.pose.position.x = goal_pos[0]
        goal_pose.pose.position.y = goal_pos[1]
        goal_pose.pose.position.z = 0.0
        
        # Calculate orientation to face the person
        if self.person_to_greet['person_pos'] is not None:
            # Calculate direction vector from goal position to person position
            direction = self.person_to_greet['person_pos'] - goal_pos
            # Calculate yaw angle
            yaw = np.arctan2(direction[1], direction[0])
            goal_pose.pose.orientation = self.YawToQuaternion(yaw)
        else:
            # Default orientation if we don't have person position
            goal_pose.pose.orientation.w = 1.0
        
        # Navigate to the goal position
        self.info(f"Navigating to greet person (face ID: {self.person_to_greet['face_id']})")
        return self.goToPose(goal_pose)
def main(args=None):
    
    rclpy.init(args=args)
    rc = RobotCommander()

    rc.waitUntilNav2Active()

    waypoints = []

    def global_path_callback(msg):
        """Callback to process the global path and extract waypoints."""
        nonlocal waypoints
        waypoints = [(pose.pose.position.x, pose.pose.position.y, 1.57) for pose in msg.poses]
        rc.info(f"Received {len(waypoints)} waypoints from /global_path")

    # Subscribe to the /global_path topic
    global_path_sub = rc.create_subscription(
        Path,
        '/global_path',
        global_path_callback,
        QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
    )

    # Wait until waypoints are received
    rc.info("Waiting for waypoints from /global_path...")
    while not waypoints:
        rclpy.spin_once(rc, timeout_sec=0.5)
    
    # Phase 1: Traverse all waypoints and collect face locations
    rc.info("Phase 1: Starting navigation through waypoints...")
    detected_faces = {}  # Dictionary to store face_id -> position mapping
    
    rc.current_waypoint_idx = 0
    
    while rc.current_waypoint_idx < len(waypoints):
        # Process any pending callbacks
        rclpy.spin_once(rc, timeout_sec=0.1)
        
        # Get current waypoint
        x, y, yaw = waypoints[rc.current_waypoint_idx]
        
        rc.info(f"Navigating to waypoint {rc.current_waypoint_idx + 1}/{len(waypoints)}: ({x}, {y})")
        
        # Create the goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = rc.get_clock().now().to_msg()
        
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation = rc.YawToQuaternion(yaw)
        
        # Set current task
        rc.current_task = 'waypoint'
        
        # Send navigation command
        rc.goToPose(goal_pose)
        
        # Wait for completion or interruption
        while not rc.isTaskComplete():
            rclpy.spin_once(rc, timeout_sec=0.5)
            
            # While moving, collect face information
            if rc.detected_person_markers:
                for marker in rc.detected_person_markers.markers:
                    # Check if it's a face marker
                    if marker.ns == "face":
                        face_id = marker.id
                        face_pos = np.array([marker.pose.position.x, marker.pose.position.y])
                        # Store face position in map coordinates
                        detected_faces[face_id] = face_pos
                        rc.info(f"Detected face ID {face_id} at position {face_pos}")
            
            time.sleep(0.5)
        
        rc.info(f"Reached waypoint {rc.current_waypoint_idx + 1}!")
        
        # Optional: spin at each waypoint to look around
        if rc.current_waypoint_idx < len(waypoints) - 1:  # Don't spin at the last waypoint
            rc.info("Spinning to look around...")
            rc.spin(6.28)  # Spin 360 degrees
            while not rc.isTaskComplete():
                rclpy.spin_once(rc, timeout_sec=0.5)
                # Continue collecting face information during spin
                if rc.detected_person_markers:
                    for marker in rc.detected_person_markers.markers:
                        if marker.ns == "face":
                            face_id = marker.id
                            face_pos = np.array([marker.pose.position.x, marker.pose.position.y])
                            detected_faces[face_id] = face_pos
                            rc.info(f"Detected face ID {face_id} at position {face_pos}")
                time.sleep(0.5)

        rc.current_waypoint_idx += 1
        rc.current_task = None

    rc.info("Phase 1 completed: Traversed all waypoints!")
    rc.info(f"Detected {len(detected_faces)} unique faces during traversal.")
    
    # Phase 2: Visit and greet each detected face
    rc.info("Phase 2: Starting face greeting sequence...")
    
    for face_id, face_position in detected_faces.items():
        if face_id in rc.greeted_faces:
            rc.info(f"Already greeted face ID {face_id}, skipping.")
            continue
            
        rc.info(f"Moving to greet face ID {face_id} at position {face_position}")
        
        # Calculate approach position (slightly before the face)
        approach_distance = 1.0  # meters
        
        # If we have current robot position from AMCL
        if hasattr(rc, 'current_pose'):
            robot_pos = np.array([rc.current_pose.position.x, rc.current_pose.position.y])
            direction = face_position - robot_pos
            direction_norm = direction / np.linalg.norm(direction)
            # Calculate position 1 meter away from the face
            goal_pos = face_position - direction_norm * approach_distance
        else:
            # Fallback: just go 1 meter in front of face along X axis
            goal_pos = face_position.copy()
            goal_pos[0] -= approach_distance
        
        # Set up person to greet
        rc.person_to_greet = {
            'face_id': face_id,
            'person_pos': face_position
        }
        
        # Create the goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = rc.get_clock().now().to_msg()
        
        goal_pose.pose.position.x = goal_pos[0]
        goal_pose.pose.position.y = goal_pos[1]
        goal_pose.pose.position.z = 0.0
        
        # Calculate orientation to face the person
        direction = face_position - goal_pos
        yaw = np.arctan2(direction[1], direction[0])
        goal_pose.pose.orientation = rc.YawToQuaternion(yaw)
        
        # Set current task
        rc.current_task = 'greeting'
        
        # Navigate to the greeting position
        rc.goToPose(goal_pose)
        
        # Wait for completion
        while not rc.isTaskComplete():
            rclpy.spin_once(rc, timeout_sec=0.5)
            time.sleep(0.5)
        
        # Greet the person
        rc.sayGreeting(f"Hello there! Nice to meet you, person number {face_id}!")
        rc.greeted_faces.add(face_id)
        rc.info(f"Greeted face ID {face_id}")
        
        # Wait a moment after greeting
        time.sleep(2.0)
    
    # Return to the starting waypoint (waypoint 0)
    if waypoints:
        rc.info("Returning to starting position (waypoint 0)...")
        x, y, yaw = waypoints[0]
        
        # Create the goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = rc.get_clock().now().to_msg()
        
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation = rc.YawToQuaternion(yaw)
        
        # Set current task
        rc.current_task = 'waypoint'
        
        # Send navigation command
        rc.goToPose(goal_pose)
        
        # Wait for completion
        while not rc.isTaskComplete():
            rclpy.spin_once(rc, timeout_sec=0.5)
            time.sleep(0.5)
        
        rc.info("Returned to starting position!")
    
    rc.info("Mission completed successfully!")
    
    rc.destroyNode()
    
# And a simple example
if __name__=="__main__":
    main()