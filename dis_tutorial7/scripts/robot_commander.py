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

from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid
import tf_transformations
import cv2


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

        self.detected_rings = {}  # Dictionary to track detected rings by ID
        self.ring_markers = None  # Latest ring markers message
        self.ring_approach_distance = 0.5  # Distance to stay from ring in meters
        self.global_costmap = None  # Store the global costmap
        self.ring_approach_timeout = 60.0  # Timeout for approaching a ring
        self.current_ring_id = None  # Currently targeted ring ID
        self.already_approached_rings = set()  # Track which rings we've approached
        self.expected_ring_count = 4  # We expect to find 4 rings total


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

        self.ring_markers_sub = self.create_subscription(
            MarkerArray,
            '/ring_markers',
            self._ringMarkersCallback,
            10
        )
        
        # Subscribe to global costmap
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self._costmapCallback,
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
    
    def _ringMarkersCallback(self, msg):
        """Handle incoming ring markers"""
        self.ring_markers = msg
        
        # Process markers to update ring dictionary
        for marker in msg.markers:
            # Only process position markers with sphere type
            if marker.ns == "ring_positions" and marker.type == Marker.SPHERE:
                ring_id = marker.id
                
                # Get ring position
                ring_pos = np.array([
                    marker.pose.position.x,
                    marker.pose.position.y,
                    marker.pose.position.z
                ])
                
                # Find the corresponding text marker to get the color
                ring_color = "Unknown"
                for text_marker in msg.markers:
                    if text_marker.ns == "ring_colors" and text_marker.id == ring_id:
                        ring_color = text_marker.text
                        break
                
                # Find the normal vectors
                normals = []
                for normal_marker in msg.markers:
                    if normal_marker.ns == "ring_normals" and (normal_marker.id == ring_id or normal_marker.id == ring_id + 1):
                        # Extract normal direction from arrow marker
                        q = normal_marker.pose.orientation
                        # Convert quaternion to rotation matrix
                        matrix = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
                        # The arrow points along the X axis after rotation
                        normal = matrix[:3, 0]  # First column is the X axis
                        normals.append(normal)
                
                # Store ring data with both normals if available
                if len(normals) >= 1:
                    self.detected_rings[ring_id] = {
                        'position': ring_pos,
                        'color': ring_color,
                        'normals': normals,
                        'last_seen': self.get_clock().now()
                    }
                    self.info(f"Updated ring {ring_id}: {ring_color} at {ring_pos}")

    def _costmapCallback(self, msg):
        """Store the latest costmap data"""
        # Log the costmap values for visualization
        # self.info(f"width and height of costmap: {msg.info.width} x {msg.info.height}")
        costmap_data = np.array(msg.data).reshape(
            (msg.info.height, msg.info.width)
        )
        # self.info(f"maximum cost in costmap: {np.max(costmap_data)}")

        # self.info(f"Global costmap:\n{costmap_data}")
        self.global_costmap = msg

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
            # Extract gender from marker text if available
            marker_gender = "Unknown"
            if hasattr(marker, 'text') and marker.text:
                self.debug(f"Marker text: {marker.text}")
                # The marker text format is expected to be the gender string (Male/Female/Unknown)
                marker_gender = marker.text
                
            self.debug(f"_peopleMarkerCallback: Received marker - NS: {marker.ns}, ID: {marker.id}, Gender: {marker_gender}, Pose: {marker.pose}")    
    
    def check_point_in_costmap(self, x, y):
        """Check if a point is valid in the costmap (not occupied)"""
        if self.global_costmap is None:
            self.info("No costmap available")
            return False
        
        # Convert world coordinates to costmap cell coordinates
        origin_x = self.global_costmap.info.origin.position.x
        origin_y = self.global_costmap.info.origin.position.y
        resolution = self.global_costmap.info.resolution
        
        mx = int((x - origin_x) / resolution)
        my = int((y - origin_y) / resolution)
        
        # Check if coordinates are within costmap boundaries
        width = self.global_costmap.info.width
        height = self.global_costmap.info.height
        

        if 0 <= mx < width and 0 <= my < height:
            # Get the costmap value at this cell
            index = my * width + mx
            cost = self.global_costmap.data[index]
            
            # Visualize the costmap and the index point using OpenCV

            # Convert costmap data to a numpy array
            costmap_data = np.array(self.global_costmap.data).reshape(
                (self.global_costmap.info.height, self.global_costmap.info.width)
            )

            # Normalize costmap values for visualization
            normalized_costmap = np.clip(costmap_data, 0, 100).astype(np.uint8)
            normalized_costmap = cv2.applyColorMap(normalized_costmap, cv2.COLORMAP_JET)

            # Draw the index point on the costmap
            if 0 <= mx < width and 0 <= my < height:
                cv2.circle(normalized_costmap, (mx, my), 5, (0, 255, 0), -1)

            # Display the costmap
            cv2.imshow("Costmap Visualization", normalized_costmap)
            cv2.waitKey(1)

            # Return True if the cost is below the lethal threshold (less than 99)
            # 100 is fully occupied, -1 is unknown
            if cost < 50:
                return True
        
        return False

    def approach_ring(self, ring_id):
        """Approach a ring along its normal vector"""
        if ring_id not in self.detected_rings:
            self.error(f"Cannot approach ring {ring_id}: not in detected rings")
            return False
        
        ring_data = self.detected_rings[ring_id]
        ring_pos = ring_data['position']
        normals = ring_data['normals']
        
        if not normals:
            self.error(f"Cannot approach ring {ring_id}: no normal vectors available")
            return False
        
        self.info(f"Planning approach to {ring_data['color']} ring {ring_id}")
        
        # Calculate the two possible approach positions
        approach_positions = []
        
        # We want to make sure we get the correct normal for approach
        # Assuming normal[0] points outward from the ring
        for i, normal in enumerate(normals):
            # Get current robot position
            if hasattr(self, 'current_pose'):
                robot_pos = np.array([
                    self.current_pose.position.x,
                    self.current_pose.position.y,
                    self.current_pose.position.z
                ])
                
                # Calculate which normal points more toward the robot
                vec_to_robot = robot_pos - ring_pos
                dot_product = np.dot(normal, vec_to_robot)
                
                # If dot product is positive, this normal points more toward the robot
                normal_for_approach = normal if dot_product > 0 else -normal
            else:
                # If we don't have robot position, just use the normals as they are
                normal_for_approach = normal
            
            # Calculate approach position
            approach_pos = ring_pos + normal_for_approach * self.ring_approach_distance
            valid = self.check_point_in_costmap(approach_pos[0], approach_pos[1])
            approach_positions.append({
                'position': approach_pos,
                'normal': normal_for_approach,
                'valid': valid
            })
        
        # Log the approach options
        for i, ap in enumerate(approach_positions):
            self.info(f"Approach option {i}: pos={ap['position']}, valid={ap['valid']}")
        
        # Choose the valid approach position, prefer the first one if both are valid
        valid_approaches = [ap for ap in approach_positions if ap['valid']]
        
        if not valid_approaches:
            self.error(f"Cannot approach ring {ring_id}: no valid approach positions")
            return False
        
        # Use the first valid approach
        approach = valid_approaches[0]
        approach_pos = approach['position']
        approach_normal = approach['normal']
        
        # Create the goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_pose.pose.position.x = approach_pos[0]
        goal_pose.pose.position.y = approach_pos[1]
        goal_pose.pose.position.z = 0.0  # Set Z to 0 for ground navigation
        
        # Calculate orientation to face the ring
        direction = -approach_normal  # Face toward the ring
        yaw = np.arctan2(direction[1], direction[0])
        goal_pose.pose.orientation = self.YawToQuaternion(yaw)
        
        # Set current task
        self.current_task = 'ring_approach'
        self.current_ring_id = ring_id
        
        # Log the approach
        self.info(f"Approaching {ring_data['color']} ring {ring_id} at {approach_pos}")
        
        # Navigate to the approach position
        return self.goToPose(goal_pose)

    def find_nearest_ring(self, current_position):
        """Find the nearest ring that hasn't been approached yet"""
        if not self.detected_rings:
            return None
        
        nearest_ring = None
        min_distance = float('inf')
        
        for ring_id, ring_data in self.detected_rings.items():
            # Skip rings we've already approached
            if ring_id in self.already_approached_rings:
                continue
            
            ring_pos = ring_data['position']
            distance = np.linalg.norm(current_position[:2] - ring_pos[:2])  # 2D distance
            
            if distance < min_distance:
                min_distance = distance
                nearest_ring = ring_id
        
        return nearest_ring

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
    
    # Phase 1: Traverse waypoints, approach rings, and collect face locations
    rc.info("Phase 1: Starting navigation through waypoints...")
    detected_faces = {}  # Dictionary to store face_id -> data mapping
    
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
                    if marker.ns == "face":
                        face_id = marker.id
                        face_pos = np.array([marker.pose.position.x, marker.pose.position.y])
                        
                        # Extract gender from marker.text if available
                        gender = "Unknown"
                        if hasattr(marker, 'text') and marker.text:
                            gender = marker.text
                        
                        # Store face position and gender
                        detected_faces[face_id] = {
                            'position': face_pos,
                            'gender': gender
                        }
                        rc.info(f"Detected face ID {face_id} at position {face_pos}, gender: {gender}")
            
            time.sleep(0.5)
        
        rc.info(f"Reached waypoint {rc.current_waypoint_idx + 1}!")
        # rc.spin(6.28)  # Spin to look around

        # Look for rings at each waypoint
        # After reaching each waypoint, check for nearby rings before moving to the next waypoint
        rc.info(f"Checking for rings at waypoint {rc.current_waypoint_idx + 1}...")

        # Wait a moment for ring markers to update
        time.sleep(1.0)
        rclpy.spin_once(rc, timeout_sec=0.5)

        # Process any new rings from the latest ring markers
        if rc.ring_markers:
            rclpy.spin_once(rc, timeout_sec=0.5)  # Process callbacks again after waiting
    
        # Spin around to detect all rings at this waypoint
        rc.info("Spinning to look for rings...")
        rc.spin(-1.57/2)  # 360 degrees
        rc.spin(3.14/2)  # 180 degrees
        while not rc.isTaskComplete():
            rclpy.spin_once(rc, timeout_sec=0.1)
            time.sleep(0.1)

        # Now check for rings to approach
        rings_to_approach = []
        if rc.detected_rings:
            # Make a list of all rings not yet approached
            for ring_id, ring_data in rc.detected_rings.items():
                if ring_id not in rc.already_approached_rings:
                    rings_to_approach.append((ring_id, ring_data))
            
            rc.info(f"Found {len(rings_to_approach)} unapproached rings at this waypoint")
            
            # Keep approaching rings until none are left
            while rings_to_approach and hasattr(rc, 'current_pose'):
                robot_pos = np.array([rc.current_pose.position.x, rc.current_pose.position.y, 0.0])
                
                # Find the nearest ring
                nearest_ring_id = None
                nearest_ring_data = None
                min_distance = float('inf')
                
                for ring_id, ring_data in rings_to_approach:
                    ring_pos = ring_data['position']
                    distance = np.linalg.norm(robot_pos[:2] - ring_pos[:2])  # 2D distance
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_ring_id = ring_id
                        nearest_ring_data = ring_data
                
                if nearest_ring_id is not None:
                    rc.info(f"Approaching {nearest_ring_data['color']} ring {nearest_ring_id}, distance: {min_distance:.2f}m")
                    
                    # Approach the ring
                    if rc.approach_ring(nearest_ring_id):
                        # Wait for approach to complete
                        approach_start_time = time.time()
                        while not rc.isTaskComplete():
                            rclpy.spin_once(rc, timeout_sec=0.1)
                            
                            # Timeout check
                            if time.time() - approach_start_time > rc.ring_approach_timeout:
                                rc.error("Ring approach timed out, canceling")
                                rc.cancelTask()
                                break
                            
                            time.sleep(0.1)
                        
                        # Mark the ring as approached
                        rc.already_approached_rings.add(nearest_ring_id)
                        rc.info(f"Successfully approached ring {nearest_ring_id}")
                        
                        # Remove this ring from the list to approach
                        rings_to_approach = [(r_id, r_data) for r_id, r_data in rings_to_approach 
                                            if r_id != nearest_ring_id]
                        
                        # Spin after approaching to look for more rings or faces
                        rc.info("Spinning to look around...")
                        rc.spin(6.28)  # 360 degrees
                        while not rc.isTaskComplete():
                            rclpy.spin_once(rc, timeout_sec=0.1)
                            # Collect face information during spin
                            if rc.detected_person_markers:
                                for marker in rc.detected_person_markers.markers:
                                    if marker.ns == "face":
                                        face_id = marker.id
                                        face_pos = np.array([marker.pose.position.x, marker.pose.position.y])
                                        gender = "Unknown"
                                        if hasattr(marker, 'text') and marker.text:
                                            gender = marker.text
                                        detected_faces[face_id] = {
                                            'position': face_pos,
                                            'gender': gender
                                        }
                                        rc.info(f"Detected face ID {face_id} at position {face_pos}, gender: {gender}")
                            time.sleep(0.1)
                        
                        # Update list of rings to approach (might have found new ones during spin)
                        rings_to_approach = []
                        for ring_id, ring_data in rc.detected_rings.items():
                            if ring_id not in rc.already_approached_rings:
                                rings_to_approach.append((ring_id, ring_data))
                    else:
                        rc.error(f"Failed to approach ring {nearest_ring_id}")
                        # Remove this ring from the list to avoid endless retrying
                        rings_to_approach = [(r_id, r_data) for r_id, r_data in rings_to_approach 
                                            if r_id != nearest_ring_id]
                else:
                    break  # No more rings to approach
        else:
            rc.info(f"No rings detected at waypoint {rc.current_waypoint_idx + 1}")
            # rc.spin(6.28)  # Spin to look around
            # rc.spin(3.14)
            
        # Check if we've found all 4 rings
        if len(rc.already_approached_rings) >= 4:
            rc.info("All 4 rings have been approached! Skipping remaining waypoints.")
            # Skip to the next phase
            break
        # Move to the next waypoint
        rc.current_waypoint_idx += 1
        rc.current_task = None

    rc.info("Phase 1 completed: Traversed all waypoints!")
    rc.info(f"Detected {len(detected_faces)} unique faces and approached {len(rc.already_approached_rings)} rings.")

    
    # Phase 2: Visit and greet each detected face
    rc.info("Phase 2: Starting face greeting sequence...")
    
    for face_id, face_position in detected_faces.items():
        if face_id in rc.greeted_faces:
            rc.info(f"Already greeted face ID {face_id}, skipping.")
            continue
            
        # Extract position and gender from the face data
        face_position = face_data['position']
        face_gender = face_data.get('gender', 'Unknown')

        rc.info(f"Moving to greet face ID {face_id} at position {face_position}, gender: {face_gender}")

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
            'person_pos': face_position,
            'gender': face_gender
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

        # Greet the person using gender information
        greeting_text = f"Hello {face_gender}! Nice to meet you, person number {face_id}!"
        rc.sayGreeting(greeting_text)
        rc.greeted_faces.add(face_id)
        rc.info(f"Greeted face ID {face_id} as {face_gender}")

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