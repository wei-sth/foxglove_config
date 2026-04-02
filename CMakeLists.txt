cmake_minimum_required(VERSION 3.8)
project(foxglove_config)
set(CMAKE_BUILD_TYPE "Release")  # small_gicp must use Release, or it will be really really slow (100 times more)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
# find_package(sensor_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenMP REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PahoMqttCpp REQUIRED)
find_package(Protobuf REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# create PROJ_INCLUDE_DIRS and PROJ_LIBRARIES
pkg_check_modules(PROJ REQUIRED proj)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SegmentationInstance.msg"
  "msg/SegmentationResult.msg"
  DEPENDENCIES std_msgs
)

set(ros2_lib "rclcpp;rclpy;std_msgs;sensor_msgs;geometry_msgs;nav_msgs;cv_bridge;pcl_msgs;pcl_conversions;visualization_msgs;tf2;tf2_ros;tf2_eigen;tf2_sensor_msgs;tf2_geometry_msgs;nlohmann_json")

# generated obstacle.pb.h and obstacle.pb.cc are located in /build/foxglove_config/, so need to include ${CMAKE_CURRENT_BINARY_DIR}
include_directories(
  include
  ${PROJ_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Protobuf_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# protobuf (generated from proto/obstacle.proto)
set(OBSTACLE_PROTO_FILE proto/obstacle.proto)
protobuf_generate_cpp(OBSTACLE_PROTO_SRCS OBSTACLE_PROTO_HDRS ${OBSTACLE_PROTO_FILE})

add_executable(obstacle_detector src/obstacle_detector_standalone.cpp src/obstacle_detector.cpp)
add_executable(boundary_detector src/boundary_detector_standalone.cpp src/boundary_detector.cpp)
add_executable(obstacle_detector_node src/obstacle_detector_node.cpp src/obstacle_detector.cpp)
add_executable(localmap_node src/localmap_node.cpp src/obstacle_detector.cpp ${OBSTACLE_PROTO_SRCS})
add_executable(test_vgicp src/test_vgicp.cpp)
add_executable(test_vio src/test_vio.cpp)
add_executable(livo src/livo_node.cpp)
add_executable(proto_viewer src/proto_viewer.cpp ${OBSTACLE_PROTO_SRCS})
add_executable(imu_calib_offline src/imu_calib_offline.cpp)
# add_executable(imu_rtk_calib src/imu_rtk_calib.cpp)

ament_target_dependencies(obstacle_detector_node
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
  visualization_msgs
  nlohmann_json
)
ament_target_dependencies(localmap_node ${ros2_lib})
ament_target_dependencies(livo ${ros2_lib})
ament_target_dependencies(imu_calib_offline
  rclcpp
  sensor_msgs
  rosbag2_cpp
  rosbag2_storage
)
# ament_target_dependencies(imu_rtk_calib rclcpp sensor_msgs geometry_msgs rosbag2_cpp rosbag2_storage tf2_eigen tf2_geometry_msgs)
# target_link_libraries(imu_rtk_calib ${PCL_LIBRARIES})

target_link_libraries(obstacle_detector ${PCL_LIBRARIES} ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(boundary_detector ${OpenCV_LIBS})
target_link_libraries(obstacle_detector_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
  PahoMqttCpp::paho-mqttpp3
)
target_link_libraries(localmap_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
  PahoMqttCpp::paho-mqttpp3
  OpenMP::OpenMP_CXX
  ${Protobuf_LIBRARIES}
  ${PROJ_LIBRARIES}
)
target_link_libraries(test_vgicp
  ${PCL_LIBRARIES}
)
target_link_libraries(test_vio ${OpenCV_LIBS})
target_link_libraries(livo
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)
target_link_libraries(proto_viewer
  ${PCL_LIBRARIES}
  ${Protobuf_LIBRARIES}
)
target_link_libraries(imu_calib_offline
  yaml-cpp
)

install(TARGETS
  obstacle_detector
  boundary_detector
  obstacle_detector_node
  localmap_node
  test_vgicp
  test_vio
  livo
  proto_viewer
  imu_calib_offline
  # imu_rtk_calib
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/map_publisher.py
  scripts/yolo_segmentation_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
