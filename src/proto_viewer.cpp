#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <google/protobuf/util/time_util.h>
#include "obstacle.pb.h"

// ros2 run foxglove_config proto_viewer --input /home/weizh/data/voxel_cloud_1769046779403201818.pb
// use Ctrl+C to close the pcl viewer window, otherwise get segmentation fault

namespace {

struct Args {
  std::string input_path;
};

static bool ParseArgs(int argc, char** argv, Args* out) {
  if (!out) return false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];

    auto need_value = [&](const std::string& opt) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << opt << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (a == "--input" || a == "-i") {
      const char* v = need_value(a);
      if (!v) return false;
      out->input_path = v;
    }  else {
      std::cerr << "Only accept --input\n";
      return false;
    }
  }

  if (out->input_path.empty()) {
    std::cerr << "--input is required\n";
    return false;
  }
  return true;
}


static bool LoadVoxelCloud(const std::string& path, foxglove_config::VoxelCloud* out) {
  if (!out) return false;

  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open: " << path << "\n";
    return false;
  }

  if (!out->ParseFromIstream(&ifs)) {
    std::cerr << "Failed to parse protobuf from: " << path << "\n";
    return false;
  }

  return true;
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr ToPclPointCloud(const foxglove_config::VoxelCloud& cloud) {
  auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl_cloud->reserve(static_cast<size_t>(cloud.voxels_size()));

  for (const auto& v : cloud.voxels()) {
    pcl_cloud->push_back(pcl::PointXYZ(v.x(), v.y(), v.z()));
  }

  pcl_cloud->width = static_cast<uint32_t>(pcl_cloud->size());
  pcl_cloud->height = 1;
  pcl_cloud->is_dense = true;
  return pcl_cloud;
}

}  // namespace

int main(int argc, char** argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    return 1;
  }

  foxglove_config::VoxelCloud cloud;
  if (!LoadVoxelCloud(args.input_path, &cloud)) {
    return 2;
  }

  if (cloud.voxels_size() == 0) {
    std::cerr << "VoxelCloud.voxels is empty\n";
    return 3;
  }

  std::cout << "[voxel_cloud_viewer] loaded: " << args.input_path << "\n"
            << "  timestamp(ns): " << cloud.timestamp() << "\n"
            << "  resolution: " << cloud.resolution() << "\n"
            << "  voxel_count: " << cloud.voxels_size() << "\n";

  auto pcl_cloud = ToPclPointCloud(cloud);

  auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("VoxelCloud Viewer (PCL)");
  viewer->setBackgroundColor(0.05, 0.05, 0.05);

  // Add point cloud + settings
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(pcl_cloud, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(pcl_cloud, color, "voxels");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "voxels");
  viewer->addCoordinateSystem(1.0);

  // Ensure something is visible (otherwise it may look "all black")
  viewer->resetCamera();
  viewer->setCameraPosition(0.0, 0.0, 50.0,   // camera position
                            0.0, 0.0, 0.0,    // look-at
                            1.0, 0.0, 0.0);   // up, 0.0, 1.0, 0.0 means y is upward (image only has x and y, point cloud has x y z)

  viewer->spin();
  viewer->close();

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
