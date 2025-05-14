#! /usr/bin/env python3
# Mofidied from Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
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
        
        # Face/People detection variables
        self.face_detected = False
        self.face_position = None
        self.global_plan_active = False
        self.current_waypoint_index = 0
        self.last_waypoint_before_face = None
        self.people_positions = []

        # ROS2 subscribers
        self.create_subscription(DockStatus,
                                 'dock_status',
                                 self._dockCallback,
                                 qos_profile_sensor_data)
        
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                              'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)
        
        self.people_sub = self.create_subscription(
            MarkerArray, '/people_array', self.people_callback, QoSReliabilityPolicy.BEST_EFFORT
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
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug('Task succeeded!')
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
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def people_callback(self, msg: MarkerArray):
        """Callback for people detection messages from MarkerArray"""
        if self.global_plan_active and not self.face_detected:
            detected_person_position = None
            for marker in msg.markers:
                if marker.ns == "face_positions" and marker.type == marker.SPHERE:
                    detected_person_position = (
                        marker.pose.position.x,
                        marker.pose.position.y,
                        marker.pose.position.z
                    )
                    self.info(f"Person detected via MarkerArray at position: {detected_person_position}")
                    break

            if detected_person_position:
                self.face_position = detected_person_position
                self.face_detected = True
                self.navigateToFace()

    def navigateToFace(self):
        """Navigate to the detected face/person position"""
        if not self.face_position:
            self.warn("No face/person position available")
            return
            
        if self.current_waypoint_index < len(waypoints):
             self.last_waypoint_before_face = self.current_waypoint_index
        else:
             self.last_waypoint_before_face = max(0, len(waypoints) - 1)

        if self.goal_handle and not self.isTaskComplete():
            self.cancelTask()
            time.sleep(0.5) 
            self.info("Cancelled current waypoint navigation to approach person")
        
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.pose_frame_id
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        x, y, z = self.face_position
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation = self.YawToQuaternion(0.0)
        
        self.global_plan_active = False
        
        self.info(f"Navigating to person at: ({x:.2f}, {y:.2f})")
        nav_started = self.goToPose(goal_pose)

        if nav_started:
            while not self.isTaskComplete():
                 if not self.face_detected:
                     self.warn("Face interaction cancelled externally.")
                     self.cancelTask()
                     break
                 self.info("Approaching person...")
                 rclpy.spin_once(self, timeout_sec=0.5)

            if self.face_detected:
                 task_result = self.getResult()
                 if task_result == TaskResult.SUCCEEDED:
                     self.info("Reached person's vicinity.")
                 else:
                     self.error(f"Failed to navigate to person. Status: {task_result}")
                     self.resumeGlobalPlan()
        else:
            self.error("Failed to start navigation towards person.")
            self.resumeGlobalPlan()

    def resumeGlobalPlan(self):
        """Resume the global waypoint plan after face interaction"""
        self.info("Attempting to resume global plan...")
        self.face_detected = False
        self.face_position = None
        if self.last_waypoint_before_face is not None:
            self.info(f"Resuming global plan from waypoint index {self.last_waypoint_before_face}")
            self.current_waypoint_index = self.last_waypoint_before_face
            self.global_plan_active = True
        else:
            self.warn("No last waypoint saved, cannot resume global plan automatically.")
            self.global_plan_active = False

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
    
def main(args=None):
    
    rclpy.init(args=args)
    global waypoints 
    waypoints = [
        (0.94, -2.0, 1.57),
        (2.0, 0.0,  1.57),
        (0.45, 1.86, 1.57),
    ]

    rc = RobotCommander()

    rc.waitUntilNav2Active()

    rc.info("Starting navigation sequence...")
    rc.global_plan_active = True
    rc.current_waypoint_index = 0
    
    while rclpy.ok():
        rclpy.spin_once(rc, timeout_sec=0.1) 

        if rc.global_plan_active:
            if rc.current_waypoint_index >= len(waypoints):
                rc.info("Completed all waypoints!")
                break

            if rc.goal_handle is None or rc.isTaskComplete():
                if rc.goal_handle is not None and rc.getResult() == TaskResult.SUCCEEDED:
                     rc.info(f"Reached waypoint {rc.current_waypoint_index}!")
                     if rc.current_waypoint_index < len(waypoints):
                         rc.info("Spinning to look around...")
                         rc.spin(3.14)
                         spin_start_time = time.time()
                         while not rc.isTaskComplete():
                             if not rc.global_plan_active:
                                 rc.info("Spin interrupted by person detection.")
                                 break
                             if time.time() - spin_start_time > 15:
                                 rc.warn("Spin timed out.")
                                 rc.cancelTask()
                                 break
                             rclpy.spin_once(rc, timeout_sec=0.2)
                         
                         if not rc.global_plan_active: continue

                     rc.current_waypoint_index += 1
                     if rc.current_waypoint_index >= len(waypoints):
                         rc.info("Finished last waypoint.")
                         break

                if rc.global_plan_active and rc.current_waypoint_index < len(waypoints):
                    x, y, yaw = waypoints[rc.current_waypoint_index]
                    rc.info(f"Navigating to waypoint {rc.current_waypoint_index + 1}/{len(waypoints)}: ({x:.2f}, {y:.2f})")
                    
                    goal_pose = PoseStamped()
                    goal_pose.header.frame_id = rc.pose_frame_id
                    goal_pose.header.stamp = rc.get_clock().now().to_msg()
                    goal_pose.pose.position.x = x
                    goal_pose.pose.position.y = y
                    goal_pose.pose.orientation = rc.YawToQuaternion(yaw)
                    
                    rc.goToPose(goal_pose)
            else:
                pass

        else:
            if not rc.face_detected:
                 rc.info("Face interaction seems complete or was cancelled.")
                 rc.resumeGlobalPlan()
            else:
                 pass
        
        time.sleep(0.1)

    rc.info("Navigation sequence finished.")
    rc.destroyNode()
    rclpy.shutdown()

if __name__=="__main__":
    main()