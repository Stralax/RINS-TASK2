 
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build

 ros2 launch dis_tutorial3 sim_turtlebot_nav.launch.py world:=task1

ros2 run dis_tutorial3 detect_people1.py
ros2 run dis_tutorial3 robot_commander.py

ros2 launch dis_tutorial7 sim_turtlebot_nav.launch.py
