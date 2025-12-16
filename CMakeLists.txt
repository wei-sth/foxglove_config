cmake_minimum_required(VERSION 3.8)
project(foxglove_config)

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
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(obstacle_detector src/obstacle_detector_standalone.cpp src/obstacle_detector.cpp)
ament_target_dependencies(obstacle_detector
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
)
target_link_libraries(obstacle_detector
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

install(TARGETS
  obstacle_detector
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/map_publisher.py
  scripts/test_1216.py
  scripts/pcd.py
  scripts/util_func.py
  DESTINATION lib/${PROJECT_NAME}
)

# 安装launch文件 - 添加这部分
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()