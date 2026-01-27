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

set(ros2_lib "rclcpp;rclpy;std_msgs;sensor_msgs;geometry_msgs;nav_msgs;cv_bridge;pcl_msgs;pcl_conversions;visualization_msgs;tf2;tf2_ros;tf2_eigen;tf2_sensor_msgs;tf2_geometry_msgs;nlohmann_json")

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(obstacle_detector src/obstacle_detector_standalone.cpp src/obstacle_detector.cpp)
add_executable(obstacle_detector_node src/obstacle_detector_node.cpp src/obstacle_detector.cpp)
add_executable(test_vgicp src/test_vgicp.cpp)
add_executable(test_vio src/test_vio.cpp)
add_executable(livo src/livo_node.cpp)

ament_target_dependencies(obstacle_detector_node
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
  visualization_msgs
  nlohmann_json
)
ament_target_dependencies(livo ${ros2_lib})

target_link_libraries(obstacle_detector
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
)
target_link_libraries(obstacle_detector_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
  PahoMqttCpp::paho-mqttpp3
)
target_link_libraries(test_vgicp
  ${PCL_LIBRARIES}
)
target_link_libraries(test_vio ${OpenCV_LIBS})
target_link_libraries(livo
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)

install(TARGETS
  obstacle_detector
  obstacle_detector_node
  test_vgicp
  test_vio
  livo
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/map_publisher.py
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