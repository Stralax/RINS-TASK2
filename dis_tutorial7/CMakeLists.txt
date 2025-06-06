cmake_minimum_required(VERSION 3.5)
project(dis_tutorial7)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)


find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)

install(DIRECTORY
  launch
  meshes
  urdf
  config
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY scripts/
        DESTINATION share/${PROJECT_NAME}/scripts
        FILES_MATCHING PATTERN "*.txt")

install(PROGRAMS
  #scripts/detect_birds.py
  scripts/arm_mover_actions.py
  # scripts/robot_commander_.py
  scripts/robot_commander.py
  scripts/follow_bridge.py
  scripts/bird_classifier.py
  scripts/detect_people1.py
  # scripts/detect_rings_depth.py
  # scripts/detect_rings.py
  # scripts/detect_rings1.py
  # scripts/skeletonized_path.py
  scripts/skeletonized_path_2.py
  scripts/detect_classify_birds.py
  scripts/nlp_test.py
  # scripts/rings_kice.py 
  scripts/speak.py
  scripts/rings.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
