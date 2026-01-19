cmake_minimum_required(VERSION 3.8)
project(cone_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)

set(msg_files
  "msg/ConeDetection.msg"
  "msg/ConeDetectionArray.msg"
)

set(action_files
  "action/FollowCone.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${action_files}
  DEPENDENCIES std_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
