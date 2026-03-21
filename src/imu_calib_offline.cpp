#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/storage_options.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rosbag2_storage/storage_filter.hpp>

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Offline IMU bias & noise estimator for a *static* bag.
//
// Inputs (edit in code, not CLI):
//   - bag_folder: rosbag2 folder containing metadata.yaml and *.mcap
//   - imu_topic: topic name for sensor_msgs/msg/Imu
//
// Outputs:
//   - YAML file containing:
//       gyro_bias (rad/s), gyro_noise_std (rad/s),
//       acc_bias (m/s^2),  acc_noise_std (m/s^2),
//       gravity_direction_imu (unit vector, in IMU frame),
//       gravity_magnitude (m/s^2),
//       recommended initial gravity for filters (g0 in IMU frame)
//
// Notes:
//   - This tool assumes the robot is static: true angular velocity ~ 0, true specific force ~ gravity.
//   - Engine vibration will increase noise_std; bias estimates are still useful as "effective bias under that condition".

static inline double sqr(const double x) { return x * x; }

struct RunningStatsVec3 {
  size_t n = 0;
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Vector3d m2 = Eigen::Vector3d::Zero();  // sum of squares of differences from the current mean (per-axis)

  void add(const Eigen::Vector3d& x) {
    n++;
    const Eigen::Vector3d delta = x - mean;
    mean += delta / static_cast<double>(n);
    const Eigen::Vector3d delta2 = x - mean;
    m2 += delta.cwiseProduct(delta2);
  }

  Eigen::Vector3d variance() const {
    if (n < 2) {
      return Eigen::Vector3d::Zero();
    }
    return m2 / static_cast<double>(n - 1);
  }

  Eigen::Vector3d stddev() const {
    return variance().cwiseMax(0.0).cwiseSqrt();
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  // -------------------- USER CONFIG (edit here) --------------------
  const std::string bag_folder = "/home/weizh/data/rosbag2_2026_02_05-14_04_25_lidar_camera_calibrate";  // e.g. "/home/weizh/data/static_imu_bag"
  const std::string imu_topic = "/rslidar_imu_data";          // e.g. "/rslidar_imu_data"
  const std::string out_yaml = "/home/weizh/data/imu_static_calib.yaml";       // output file path (relative to cwd)

  // Optional: discard first few seconds to avoid startup transient.
  const double discard_first_seconds = 2.0;

  // -------------------- UNIT NOTE (READ THIS) --------------------
  // ROS sensor_msgs/Imu convention:
  //   - angular_velocity: rad/s
  //   - linear_acceleration: m/s^2
  //
  // RoboSense RSAIRY driver publishes linear_acceleration in "g" (see decoder_RSAIRY.hpp):
  //   a_g = raw * (range_g / 32768)
  // i.e. WITHOUT multiplying 9.80665.
  //
  // This offline tool will always output standard units in YAML:
  //   - accel in m/s^2
  // So we convert the incoming message by this scale factor.
  //
  // If you switch to a driver that already publishes m/s^2, set this to 1.0.
  const double acc_scale_to_mps2 = 9.80665;
  // ----------------------------------------------------------------

  rosbag2_cpp::readers::SequentialReader reader;
  rosbag2_cpp::StorageOptions storage_options;
  storage_options.uri = bag_folder;
  storage_options.storage_id = "mcap";  // ROS2 Humble rosbag2_storage_mcap

  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";

  try {
    reader.open(storage_options, converter_options);
  } catch (const std::exception& e) {
    std::cerr << "Failed to open bag folder: " << bag_folder << "\n" << e.what() << std::endl;
    return 1;
  }

  rosbag2_storage::StorageFilter filter;
  filter.topics = {imu_topic};
  reader.set_filter(filter);

  rclcpp::Serialization<sensor_msgs::msg::Imu> ser;
  sensor_msgs::msg::Imu imu_msg;

  RunningStatsVec3 gyro_stats;
  RunningStatsVec3 acc_stats;

  bool has_t0 = false;
  double t0 = 0.0;

  size_t msg_count = 0;

  while (reader.has_next()) {
    auto bag_msg = reader.read_next();
    if (!bag_msg) {
      continue;
    }

    // Deserialize to sensor_msgs/msg/Imu
    rclcpp::SerializedMessage serialized_msg(*(bag_msg->serialized_data));
    ser.deserialize_message(&serialized_msg, &imu_msg);

    const double t = rclcpp::Time(imu_msg.header.stamp).seconds();
    if (!has_t0) {
      has_t0 = true;
      t0 = t;
    }

    if (t - t0 < discard_first_seconds) {
      continue;
    }

    const Eigen::Vector3d w(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z);
    const Eigen::Vector3d a_raw(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z);
    const Eigen::Vector3d a = a_raw * acc_scale_to_mps2;  // convert g -> m/s^2 (or keep as-is if scale=1)

    gyro_stats.add(w);
    acc_stats.add(a);

    msg_count++;
  }

  if (msg_count < 10) {
    std::cerr << "Too few IMU messages read (" << msg_count << "). Check bag_folder/topic." << std::endl;
    return 2;
  }

  const Eigen::Vector3d gyro_bias = gyro_stats.mean;  // rad/s
  const Eigen::Vector3d gyro_noise_std = gyro_stats.stddev();

  const Eigen::Vector3d acc_mean = acc_stats.mean;    // m/s^2
  const Eigen::Vector3d acc_noise_std = acc_stats.stddev();

  const double g_mag = acc_mean.norm();
  Eigen::Vector3d g_dir_imu = Eigen::Vector3d::Zero();
  if (g_mag > 1e-6) {
    g_dir_imu = acc_mean / g_mag;
  }

  // If you want an initial gravity vector (in IMU frame) for filters:
  // For a static IMU, linear_acceleration reading typically includes gravity.
  // Here we use standard gravity magnitude 9.80665 m/s^2.
  const Eigen::Vector3d g0_imu = g_dir_imu * 9.80665;

  YAML::Node root;
  root["bag_folder"] = bag_folder;
  root["imu_topic"] = imu_topic;
  root["discard_first_seconds"] = discard_first_seconds;
  root["used_message_count"] = static_cast<int>(msg_count);

  auto vec3 = [](const Eigen::Vector3d& v) {
    YAML::Node n;
    n.push_back(v.x());
    n.push_back(v.y());
    n.push_back(v.z());
    return n;
  };

  root["gyro_bias_radps"] = vec3(gyro_bias);
  root["gyro_noise_std_radps"] = vec3(gyro_noise_std);

  root["acc_scale_to_mps2"] = acc_scale_to_mps2;
  root["acc_mean_mps2"] = vec3(acc_mean);
  root["acc_noise_std_mps2"] = vec3(acc_noise_std);

  root["gravity_magnitude_mps2"] = g_mag;
  root["gravity_direction_imu_unit"] = vec3(g_dir_imu);
  root["gravity_init_imu_mps2"] = vec3(g0_imu);

  // Also provide "bias-corrected" stats suggestion:
  // For static bag, "acc_bias" is ambiguous because gravity dominates. We still output a simple estimate:
  //   acc_bias_approx ~= acc_mean - g_dir_imu * 9.80665
  // This removes the gravity component and leaves residual offset (captures bias/scale roughly).
  const Eigen::Vector3d acc_bias_approx = acc_mean - g0_imu;
  root["acc_bias_approx_mps2"] = vec3(acc_bias_approx);

  std::ofstream ofs(out_yaml);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open output yaml: " << out_yaml << std::endl;
    return 3;
  }
  ofs << root;
  ofs.close();

  std::cout << "Wrote YAML: " << out_yaml << "\n";
  std::cout << "gyro_bias(rad/s): " << gyro_bias.transpose() << "\n";
  std::cout << "gyro_noise_std(rad/s): " << gyro_noise_std.transpose() << "\n";
  std::cout << "acc_scale_to_mps2: " << acc_scale_to_mps2 << "\n";
  std::cout << "acc_mean(m/s^2): " << acc_mean.transpose() << "\n";
  std::cout << "acc_noise_std(m/s^2): " << acc_noise_std.transpose() << "\n";
  std::cout << "gravity_dir_imu: " << g_dir_imu.transpose() << ", |g|=" << g_mag << "\n";

  rclcpp::shutdown();
  return 0;
}
