#include <pcl/io/pcd_io.h>
#include "/home/weizh/foxglove_ws/src/foxglove_config/include/obstacle_detector.h"

#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <algorithm>
#include <iostream> // For std::cout
#include <numeric> // For std::accumulate
#include <cmath> // For M_PI

// Helper for sorting PointXYZ_Local by Z
bool point_z_cmp_local(PointXYZ_Local a, PointXYZ_Local b) { return a.z < b.z; }

RangeImageObstacleDetector::RangeImageObstacleDetector(int num_rings, int num_sectors, 
                               float max_distance, float min_cluster_z_difference)
    : num_rings(num_rings), num_sectors(num_sectors), max_distance(max_distance),
      min_cluster_z_difference_(min_cluster_z_difference) {
    
    // Initialize Range Image related members (retained for saveNormalsToPCD)
    range_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, 
                           cv::Scalar(std::numeric_limits<float>::max()));
    x_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    y_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    z_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    valid_mask_ = cv::Mat(num_rings, num_sectors, CV_8UC1, cv::Scalar(0));

    // Initialize PatchWork++ inspired parameters
    // These are default values, can be made configurable if needed
    verbose_ = false;
    enable_RNR_ = true;
    enable_RVPF_ = true;
    num_iter_ = 3;
    num_lpr_ = 20;
    num_min_pts_ = 10;
    num_zones_ = 4;
    num_rings_of_interest_ = 4;
    RNR_ver_angle_thr_ = -15.0;
    RNR_intensity_thr_ = 0.2;
    sensor_height_ = 0.4; // This should be calibrated for the specific sensor setup
    th_seeds_ = 0.125;
    th_dist_ = 0.125;
    th_seeds_v_ = 0.25;
    th_dist_v_ = 0.1;
    uprightness_thr_ = 0.707;
    adaptive_seed_selection_margin_ = -1.2;
    num_sectors_each_zone_ = {16, 32, 54, 32};
    num_rings_each_zone_ = {2, 4, 4, 4};
    max_flatness_storage_ = 1000;
    max_elevation_storage_ = 1000;
    elevation_thr_ = {0, 0, 0, 0};
    flatness_thr_ = {0, 0, 0, 0};
    min_ground_ring_index_ = 8; // Rings 0-7 are upwards, 8-15 are downwards. Ground points should be from ring 8 or higher.

    // Initialize PatchWork++ internal state variables
    update_flatness_.resize(num_zones_);
    for(int i=0; i<num_zones_; ++i) update_flatness_[i].reserve(max_flatness_storage_);
    update_elevation_.resize(num_zones_);
    for(int i=0; i<num_zones_; ++i) update_elevation_[i].reserve(max_elevation_storage_);

    // Initialize CZM parameters
    double min_range_z2_ = (7 * 0.5 + max_distance) / 8.0; // Using 0.5 as a conceptual min_range for CZM
    double min_range_z3_ = (3 * 0.5 + max_distance) / 4.0;
    double min_range_z4_ = (0.5 + max_distance) / 2.0;
    min_ranges_          = {0.5, min_range_z2_, min_range_z3_, min_range_z4_}; // Using 0.5 as a conceptual min_range for CZM

    ring_sizes_   = {(min_range_z2_ - min_ranges_.at(0)) / num_rings_each_zone_.at(0),
                     (min_range_z3_ - min_range_z2_) / num_rings_each_zone_.at(1),
                     (min_range_z4_ - min_range_z3_) / num_rings_each_zone_.at(2),
                     (max_distance - min_range_z4_) / num_rings_each_zone_.at(3)};
    sector_sizes_ = {2 * M_PI / num_sectors_each_zone_.at(0),
                     2 * M_PI / num_sectors_each_zone_.at(1),
                     2 * M_PI / num_sectors_each_zone_.at(2),
                     2 * M_PI / num_sectors_each_zone_.at(3)};

    // Initialize Concentric Zone Model
    czm_patches_.resize(num_zones_);
    for (int k = 0; k < num_zones_; k++) {
      czm_patches_[k].resize(num_rings_each_zone_[k]);
      for (int i = 0; i < num_rings_each_zone_[k]; i++) {
        czm_patches_[k][i].resize(num_sectors_each_zone_[k]);
      }
    }

    if (verbose_) std::cout << "RangeImageObstacleDetector::RangeImageObstacleDetector() - INITIALIZATION COMPLETE" << std::endl;
}

// Helper function for PatchWork++
void RangeImageObstacleDetector::addCloud(std::vector<PointXYZ_Local> &cloud, const std::vector<PointXYZ_Local> &add) {
  cloud.insert(cloud.end(), add.begin(), add.end());
}

// Helper function for PatchWork++
void RangeImageObstacleDetector::flush_patches(std::vector<ZonePatches> &czm) {
  for (int k = 0; k < num_zones_; k++) {
    for (int i = 0; i < num_rings_each_zone_[k]; i++) {
      for (int j = 0; j < num_sectors_each_zone_[k]; j++) {
        czm[k][i][j].clear();
      }
    }
  }
  if (verbose_) std::cout << "\033[1;31m" << "RangeImageObstacleDetector::flush_patches() - Flushed patches successfully!" << "\033[0m" << std::endl;
}

// Helper function for PatchWork++
void RangeImageObstacleDetector::estimate_plane(const std::vector<PointXYZ_Local> &ground) {
  if (ground.empty()) {
    normal_ = Eigen::Vector3f(0,0,1); // Default normal if no ground points
    pc_mean_ = Eigen::Vector3f(0,0,0);
    singular_values_ = Eigen::Vector3f(0,0,0);
    d_ = 0;
    return;
  }

  Eigen::MatrixX3f eigen_ground(ground.size(), 3);
  int j = 0;
  for (auto &p : ground) {
    eigen_ground.row(j++) << p.x, p.y, p.z;
  }
  Eigen::MatrixX3f centered = eigen_ground.rowwise() - eigen_ground.colwise().mean();
  Eigen::MatrixX3f cov =
      (centered.adjoint() * centered) / static_cast<double>(eigen_ground.rows() - 1);

  pc_mean_.resize(3);
  pc_mean_ << eigen_ground.colwise().mean()(0), eigen_ground.colwise().mean()(1),
      eigen_ground.colwise().mean()(2);

  Eigen::JacobiSVD<Eigen::MatrixX3f> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
  singular_values_ = svd.singularValues();

  // use the least singular vector as normal
  normal_ = (svd.matrixU().col(2));

  if (normal_(2) < 0) {
    for (int i = 0; i < 3; i++) normal_(i) *= -1;
  }

  // mean ground seeds value
  Eigen::Vector3f seeds_mean = pc_mean_.head<3>();

  // according to normal.T*[x,y,z] = -d
  d_ = -(normal_.transpose() * seeds_mean)(0, 0);
}

// Helper function for PatchWork++
void RangeImageObstacleDetector::extract_initial_seeds(const int zone_idx,
                                        const std::vector<PointXYZ_Local> &p_sorted,
                                        std::vector<PointXYZ_Local> &init_seeds,
                                        double th_seed) {
  init_seeds.clear();

  // LPR is the mean of low point representative
  double sum = 0;
  int cnt    = 0;

  int init_idx = 0;
  if (zone_idx == 0) {
    for (int i = 0; i < p_sorted.size(); i++) {
      if (p_sorted[i].z < adaptive_seed_selection_margin_ * sensor_height_) {
        ++init_idx;
      } else {
        break;
      }
    }
  }

  // Calculate the mean height value.
  for (int i = init_idx; i < p_sorted.size() && cnt < num_lpr_; i++) {
    sum += p_sorted[i].z;
    cnt++;
  }
  double lpr_height = cnt != 0 ? sum / cnt : 0;  // in case divide by 0

  // iterate pointcloud, filter those height is less than lpr.height+th_seed
  for (int i = 0; i < p_sorted.size(); i++) {
    if (p_sorted[i].z < lpr_height + th_seed) {
      init_seeds.push_back(p_sorted[i]);
    }
  }
}

// Helper function for PatchWork++
void RangeImageObstacleDetector::extract_initial_seeds(const int zone_idx,
                                        const std::vector<PointXYZ_Local> &p_sorted,
                                        std::vector<PointXYZ_Local> &init_seeds) {
  init_seeds.clear();

  // LPR is the mean of low point representative
  double sum = 0;
  int cnt    = 0;

  int init_idx = 0;
  if (zone_idx == 0) {
    for (int i = 0; i < p_sorted.size(); i++) {
      if (p_sorted[i].z < adaptive_seed_selection_margin_ * sensor_height_) {
        ++init_idx;
      } else {
        break;
      }
    }
  }

  // Calculate the mean height value.
  for (int i = init_idx; i < p_sorted.size() && cnt < num_lpr_; i++) {
    sum += p_sorted[i].z;
    cnt++;
  }
  double lpr_height = cnt != 0 ? sum / cnt : 0;  // in case divide by 0

  // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
  for (int i = 0; i < p_sorted.size(); i++) {
    if (p_sorted[i].z < lpr_height + th_seeds_) {
      init_seeds.push_back(p_sorted[i]);
    }
  }
}

// Helper function for PatchWork++
void RangeImageObstacleDetector::extract_piecewiseground(const int zone_idx,
                                          const std::vector<PointXYZ_Local> &src,
                                          std::vector<PointXYZ_Local> &dst,
                                          std::vector<PointXYZ_Local> &non_ground_dst) {
  // 0. Initialization
  ground_pc_local_.clear();
  dst.clear();
  non_ground_dst.clear();

  // 1. Region-wise Vertical Plane Fitting (R-VPF)
  // : removes potential vertical plane under the ground plane
  std::vector<PointXYZ_Local> src_wo_verticals = src;

  if (enable_RVPF_) {
    for (int i = 0; i < num_iter_; i++) {
      extract_initial_seeds(zone_idx, src_wo_verticals, ground_pc_local_, th_seeds_v_);
      estimate_plane(ground_pc_local_);

      if (zone_idx == 0 && normal_(2) < uprightness_thr_) {
        std::vector<PointXYZ_Local> src_tmp = src_wo_verticals;
        src_wo_verticals.clear();

        for (auto point : src_tmp) {
          double distance = calc_point_to_plane_d(point, normal_, d_);

          if (std::abs(distance) < th_dist_v_) {
            non_ground_dst.push_back(point);
          } else {
            src_wo_verticals.push_back(point);
          }
        }
      } else
        break;
    }
  }

  // 2. Region-wise Ground Plane Fitting (R-GPF)
  // : fits the ground plane

  extract_initial_seeds(zone_idx, src_wo_verticals, ground_pc_local_);
  estimate_plane(ground_pc_local_);

  for (int i = 0; i < num_iter_; i++) {
    ground_pc_local_.clear();

    for (auto point : src_wo_verticals) {
      double distance = calc_point_to_plane_d(point, normal_, d_);

      if (i < num_iter_ - 1) {
        if (distance < th_dist_) {
          ground_pc_local_.push_back(point);
        }
      } else {
        if (distance < th_dist_) {
          dst.push_back(point);
        } else {
          non_ground_dst.push_back(point);
        }
      }
    }

    if (i < num_iter_ - 1) {
      estimate_plane(ground_pc_local_);
    } else {
      estimate_plane(dst);
    }
  }

  if (dst.size() + non_ground_dst.size() != src.size()) {
    if (verbose_) {
      std::cout << "\033[1;33m"
                << "Points are Missing/Adding !!! Please Check !! "
                << "\033[0m" << std::endl;
      std::cout << "gnd size: " << dst.size() << ", non gnd size: " << non_ground_dst.size()
                << ", src: " << src.size() << std::endl;
    }
  }
}

// Helper function for PatchWork++
double RangeImageObstacleDetector::calc_point_to_plane_d(PointXYZ_Local p, Eigen::VectorXf normal, double d) {
  return normal(0) * p.x + normal(1) * p.y + normal(2) * p.z + d;
}

// Helper function for PatchWork++
void RangeImageObstacleDetector::calc_mean_stdev(const std::vector<double> &vec, double &mean, double &stdev) {
  if (vec.empty()) { // Handle empty vector case
    mean = 0.0;
    stdev = 0.0;
    return;
  }
  if (vec.size() == 1) { // Handle single element vector case
    mean = vec[0];
    stdev = 0.0;
    return;
  }

  mean = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();

  double sum_sq_diff = 0.0;
  for (double val : vec) {
    sum_sq_diff += (val - mean) * (val - mean);
  }
  stdev = std::sqrt(sum_sq_diff / (vec.size() - 1));
}

// Helper function for PatchWork++
double RangeImageObstacleDetector::xy2theta(const double &x, const double &y) {  // 0 ~ 2 * PI
  double angle = atan2(y, x);
  return angle > 0 ? angle : 2 * M_PI + angle;
}

// Helper function for PatchWork++
double RangeImageObstacleDetector::xy2radius(const double &x, const double &y) {
  return std::sqrt(x * x + y * y);
}

// Helper function for PatchWork++
void RangeImageObstacleDetector::pc2czm(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &src, std::vector<ZonePatches> &czm) {
  double max_range = max_distance; // Use max_distance from RangeImageObstacleDetector
  double min_range = min_ranges_[0]; // Use the first min_range from CZM setup

  double min_range_0 = min_ranges_[0], min_range_1 = min_ranges_[1], min_range_2 = min_ranges_[2],
         min_range_3 = min_ranges_[3];
  int num_ring_0 = num_rings_each_zone_[0], num_sector_0 = num_sectors_each_zone_[0];
  int num_ring_1 = num_rings_each_zone_[1], num_sector_1 = num_sectors_each_zone_[1];
  int num_ring_2 = num_rings_each_zone_[2], num_sector_2 = num_sectors_each_zone_[2];
  int num_ring_3 = num_rings_each_zone_[3], num_sector_3 = num_sectors_each_zone_[3];

  for (int i = 0; i < src->points.size(); i++) {
    const auto& pt_normal = src->points[i];
    float x = pt_normal.x, y = pt_normal.y, z = pt_normal.z;
    float intensity = pt_normal.intensity;
    uint16_t ring = static_cast<uint16_t>(pt_normal.normal_x); // Retrieve ring from normal_x

    double r = xy2radius(x, y);
    int ring_idx, sector_idx;
    if ((r <= max_range) && (r > min_range)) {
      double theta = xy2theta(x, y);

      if (r < min_range_1) {  // In First rings
        ring_idx   = std::min(static_cast<int>(((r - min_range_0) / ring_sizes_[0])), num_ring_0 - 1);
        sector_idx = std::min(static_cast<int>((theta / sector_sizes_[0])), num_sector_0 - 1);
        czm[0][ring_idx][sector_idx].emplace_back(PointXYZ_Local(x, y, z, intensity, ring, i));
      } else if (r < min_range_2) {
        ring_idx   = std::min(static_cast<int>(((r - min_range_1) / ring_sizes_[1])), num_ring_1 - 1);
        sector_idx = std::min(static_cast<int>((theta / sector_sizes_[1])), num_sector_1 - 1);
        czm[1][ring_idx][sector_idx].emplace_back(PointXYZ_Local(x, y, z, intensity, ring, i));
      } else if (r < min_range_3) {
        ring_idx   = std::min(static_cast<int>(((r - min_range_2) / ring_sizes_[2])), num_ring_2 - 1);
        sector_idx = std::min(static_cast<int>((theta / sector_sizes_[2])), num_sector_2 - 1);
        czm[2][ring_idx][sector_idx].emplace_back(PointXYZ_Local(x, y, z, intensity, ring, i));
      } else {  // Far!
        ring_idx   = std::min(static_cast<int>(((r - min_range_3) / ring_sizes_[3])), num_ring_3 - 1);
        sector_idx = std::min(static_cast<int>((theta / sector_sizes_[3])), num_sector_3 - 1);
        czm[3][ring_idx][sector_idx].emplace_back(PointXYZ_Local(x, y, z, intensity, ring, i));
      }

    } else {
      cloud_nonground_local_.emplace_back(PointXYZ_Local(x, y, z, intensity, ring, i));
    }
  }
  if (verbose_) std::cout << "\033[1;33m" << "RangeImageObstacleDetector::pc2czm() - Divides pointcloud into the concentric zone model successfully" << "\033[0m" << std::endl;
}

// Helper function for PatchWork++
pcl::PointCloud<pcl::PointXYZINormal>::Ptr RangeImageObstacleDetector::reflected_noise_removal(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_in) {
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZINormal>);
  rnr_noise_points_.clear(); // Clear previous RNR noise points

  int cnt = 0;
  for (size_t i = 0; i < cloud_in->points.size(); i++) {
    const auto& pt = cloud_in->points[i];
    double r = xy2radius(pt.x, pt.y);
    double z = pt.z;
    double ver_angle_in_deg = atan2(z, r) * 180 / M_PI;

    if (ver_angle_in_deg < RNR_ver_angle_thr_ && z < -sensor_height_ - 0.8 &&
        pt.intensity < RNR_intensity_thr_) {
      rnr_noise_points_.emplace_back(PointXYZ_Local(pt.x, pt.y, pt.z, pt.intensity, static_cast<uint16_t>(pt.normal_x), i));
      cnt++;
    } else {
      cloud_out->points.push_back(pt);
    }
  }

  if (verbose_) std::cout << "RangeImageObstacleDetector::reflected_noise_removal() - Number of Noises : " << cnt << std::endl;
  return cloud_out;
}

// Helper function for PatchWork++
void RangeImageObstacleDetector::update_elevation_thr(void) {
  for (int i = 0; i < num_rings_of_interest_; i++) {
    if (update_elevation_[i].empty()) continue;

    double update_mean = 0.0, update_stdev = 0.0;
    calc_mean_stdev(update_elevation_[i], update_mean, update_stdev);
    if (i == 0) {
      elevation_thr_[i] = update_mean + 3 * update_stdev;
      sensor_height_    = -update_mean;
    } else {
      elevation_thr_[i] = update_mean + 2 * update_stdev;
    }

    if (verbose_) std::cout << "elevation threshold [" << i << "]: " << elevation_thr_[i] << std::endl;

    int exceed_num = update_elevation_[i].size() - max_elevation_storage_;
    if (exceed_num > 0)
      update_elevation_[i].erase(update_elevation_[i].begin(),
                                 update_elevation_[i].begin() + exceed_num);
  }
}

// Helper function for PatchWork++
void RangeImageObstacleDetector::update_flatness_thr(void) {
  for (int i = 0; i < num_rings_of_interest_; i++) {
    if (update_flatness_[i].empty()) break;
    if (update_flatness_[i].size() <= 1) break;

    double update_mean = 0.0, update_stdev = 0.0;
    calc_mean_stdev(update_flatness_[i], update_mean, update_stdev);
    flatness_thr_[i] = update_mean + update_stdev;

    if (verbose_) { std::cout << "flatness threshold [" << i << "]: " << flatness_thr_[i] << std::endl; }

    int exceed_num = update_flatness_[i].size() - max_flatness_storage_;
    if (exceed_num > 0)
      update_flatness_[i].erase(update_flatness_[i].begin(),
                                update_flatness_[i].begin() + exceed_num);
  }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr RangeImageObstacleDetector::detectObstacles(
    pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw) {
    
    // Clear previous results
    cloud_ground_local_.clear();
    cloud_nonground_local_.clear();
    ground_pc_local_.clear();
    non_ground_pc_local_.clear();
    regionwise_ground_local_.clear();
    regionwise_nonground_local_.clear();
    rnr_noise_points_.clear(); // Clear RNR noise points

    if (verbose_) std::cout << "\033[1;32m" << "RangeImageObstacleDetector::detectObstacles() - Ground Estimation starts !" << "\033[0m" << std::endl;

    // Step 1: Filter by distance (retained from original)
    // This step is still useful for initial filtering before building range image or CZM
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_filtered_normal_for_range_image = filterByDistance(cloud_raw);
    
    // Step 2: Build Range Image (retained for saveNormalsToPCD)
    // This populates x_image_, y_image_, z_image_, valid_mask_
    buildRangeImage(cloud_filtered_normal_for_range_image);

    // Step 3: Reflected Noise Removal (RNR) - PatchWork++ inspired
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_after_rnr = cloud_filtered_normal_for_range_image;
    if (enable_RNR_) {
        cloud_after_rnr = reflected_noise_removal(cloud_filtered_normal_for_range_image);
    }

    // Step 4: Concentric Zone Model (CZM) Population - PatchWork++ inspired
    flush_patches(czm_patches_); // Clear CZM patches
    pc2czm(cloud_after_rnr, czm_patches_); // Populate CZM with points after RNR

    int concentric_idx = 0;

    // Step 5: Iterate through patches for ground segmentation - PatchWork++ inspired
    for (int zone_idx = 0; zone_idx < num_zones_; ++zone_idx) {
        auto& zone = czm_patches_[zone_idx];

        for (int ring_idx = 0; ring_idx < num_rings_each_zone_[zone_idx]; ++ring_idx) {
            for (int sector_idx = 0; sector_idx < num_sectors_each_zone_[zone_idx]; ++sector_idx) {
                if (zone[ring_idx][sector_idx].size() < num_min_pts_) {
                    addCloud(cloud_nonground_local_, zone[ring_idx][sector_idx]);
                    continue;
                }

                // Region-wise sorting (faster than global sorting method)
                std::sort(zone[ring_idx][sector_idx].begin(), zone[ring_idx][sector_idx].end(), point_z_cmp_local);

                extract_piecewiseground(
                    zone_idx, zone[ring_idx][sector_idx], regionwise_ground_local_, regionwise_nonground_local_);
                
                // Status of each patch
                const double ground_uprightness = normal_(2);
                const double ground_elevation   = pc_mean_(2);
                const double ground_flatness    = singular_values_.minCoeff();
                // const double line_variable      = singular_values_(1) != 0 ? singular_values_(0) / singular_values_(1) : std::numeric_limits<double>::max(); // TGR is disabled

                double heading = 0.0;
                for (int i = 0; i < 3; i++) heading += pc_mean_(i) * normal_(i);

                bool is_upright         = ground_uprightness > uprightness_thr_;
                bool is_near_zone       = concentric_idx < num_rings_of_interest_;
                bool is_heading_outside = heading < 0.0;

                bool is_not_elevated = false;
                bool is_flat         = false;

                if (concentric_idx < num_rings_of_interest_) {
                    is_not_elevated = ground_elevation < elevation_thr_[concentric_idx];
                    is_flat         = ground_flatness < flatness_thr_[concentric_idx];
                }

                // NEW: Ring-based ground plausibility check
                bool is_ground_plausible_by_ring = true;
                if (!regionwise_ground_local_.empty()) {
                    for (const auto& pt : regionwise_ground_local_) {
                        if (pt.ring < min_ground_ring_index_) { // If any ground candidate point comes from an upward-scanning ring (0-7)
                            is_ground_plausible_by_ring = false;
                            if (verbose_) {
                                std::cout << "\033[1;33m" << "Patch rejected by ring-based ground check: "
                                          << "Zone " << zone_idx << ", Ring " << ring_idx << ", Sector " << sector_idx
                                          << ", Point from invalid ring: " << pt.ring << "\033[0m" << std::endl;
                            }
                            break; // This patch is deemed non-ground, no need to check further points
                        }
                    }
                }

                // Store the elevation & flatness variables for A-GLE
                // Only update adaptive thresholds if the patch is considered plausible ground
                if (is_upright && is_not_elevated && is_near_zone && is_ground_plausible_by_ring) {
                    update_elevation_[concentric_idx].push_back(ground_elevation);
                    update_flatness_[concentric_idx].push_back(ground_flatness);
                }

                // Ground estimation based on conditions, incorporating the new ring-based check
                if (!is_upright || !is_ground_plausible_by_ring) { // If not upright or ring-based check fails, it's non-ground
                    addCloud(cloud_nonground_local_, regionwise_ground_local_);
                } else if (!is_near_zone) { // Far zones, default to ground
                    addCloud(cloud_ground_local_, regionwise_ground_local_);
                } else if (!is_heading_outside) { // Normal vector points outside sensor, usually non-ground
                    addCloud(cloud_nonground_local_, regionwise_ground_local_);
                } else if (is_not_elevated || is_flat) { // Sufficiently low or flat, considered ground
                    addCloud(cloud_ground_local_, regionwise_ground_local_);
                } else { // Other cases, temporarily considered obstacle
                    // If TGR was enabled, these would be candidates for revert.
                    // Since TGR is disabled, these are considered non-ground for now.
                    addCloud(cloud_nonground_local_, regionwise_ground_local_);
                }
                // Every regionwise_nonground is considered nonground.
                addCloud(cloud_nonground_local_, regionwise_nonground_local_);
            }
            concentric_idx++;
        }
    }

    // Step 6: Update adaptive thresholds - PatchWork++ inspired
    update_elevation_thr();
    update_flatness_thr();

    if (verbose_) std::cout << "\033[1;32m" << "RangeImageObstacleDetector::detectObstacles() - Ground Estimation is finished !" << "\033[0m" << std::endl;

    // Step 7: Prepare obstacles for clustering (convert PointXYZ_Local to pcl::PointXYZINormal)
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_for_clustering(new pcl::PointCloud<pcl::PointXYZINormal>);
    for (const auto& pt_local : cloud_nonground_local_) {
        pcl::PointXYZINormal pt_normal;
        pt_normal.x = pt_local.x;
        pt_normal.y = pt_local.y;
        pt_normal.z = pt_local.z;
        pt_normal.intensity = pt_local.intensity;
        pt_normal.normal_x = static_cast<float>(pt_local.ring); // Store ring in normal_x
        // normal_y, normal_z, curvature can be left as default or set to 0
        obstacles_for_clustering->points.push_back(pt_normal);
    }
    
    // Step 8: Perform clustering on the detected obstacles (PointXYZINormal) - Retained from original
    pcl::PointCloud<pcl::PointXYZI>::Ptr clustered_obstacles(new pcl::PointCloud<pcl::PointXYZI>);
    
    if (obstacles_for_clustering->points.empty()) {
        return clustered_obstacles;
    }

    float current_intensity = 0.0f; // Initialize intensity for clusters

    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZINormal>);
    tree->setInputCloud(obstacles_for_clustering);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZINormal> ec;
    ec.setClusterTolerance(0.3); // 30cm
    ec.setMinClusterSize(20);
    ec.setMaxClusterSize(10000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(obstacles_for_clustering);
    ec.extract(cluster_indices);

    for (const auto& indices : cluster_indices) {
        std::set<uint16_t> rings_in_cluster;
        float min_z_cluster = std::numeric_limits<float>::max();
        float max_z_cluster = -std::numeric_limits<float>::max();
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr current_cluster_xyzi(new pcl::PointCloud<pcl::PointXYZI>);

        for (int idx : indices.indices) {
            const auto& pt = obstacles_for_clustering->points[idx];
            rings_in_cluster.insert(static_cast<uint16_t>(pt.normal_x)); // Retrieve ring from normal_x
            
            min_z_cluster = std::min(min_z_cluster, pt.z);
            max_z_cluster = std::max(max_z_cluster, pt.z);

            pcl::PointXYZI xyzi_pt;
            xyzi_pt.x = pt.x;
            xyzi_pt.y = pt.y;
            xyzi_pt.z = pt.z;
            xyzi_pt.intensity = current_intensity; // Assign unique intensity
            current_cluster_xyzi->points.push_back(xyzi_pt);
        }
        
        // Filter clusters based on the number of rings spanned AND Z-difference
        if (rings_in_cluster.size() >= 2 && (max_z_cluster - min_z_cluster) > min_cluster_z_difference_) {
            *clustered_obstacles += *current_cluster_xyzi;
            current_intensity += 10.0f; // Increment intensity for the next cluster
        }
    }
    
    return clustered_obstacles;
}

pcl::PointCloud<pcl::PointXYZINormal>::Ptr RangeImageObstacleDetector::filterByDistance(pcl::PointCloud<PointXYZIRT>::Ptr cloud) {
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr filtered_normal(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    
    for (const auto& pt_irt : cloud->points) {
        float distance = std::sqrt(pt_irt.x * pt_irt.x + pt_irt.y * pt_irt.y + pt_irt.z * pt_irt.z);
        
        if (distance <= max_distance && distance > 0.5f) {  // Min distance 0.5m
            pcl::PointXYZINormal pt_normal;
            pt_normal.x = pt_irt.x;
            pt_normal.y = pt_irt.y;
            pt_normal.z = pt_irt.z;
            pt_normal.intensity = pt_irt.intensity;
            pt_normal.normal_x = static_cast<float>(pt_irt.ring); // Store ring in normal_x
            pt_normal.normal_y = pt_irt.time; // Store time in normal_y
            // normal_z, curvature can be left as default or set to 0
            
            filtered_normal->points.push_back(pt_normal);
        }
    }
    
    return filtered_normal;
}

void RangeImageObstacleDetector::buildRangeImage(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud) {
    // since we use min range, must filter out ego cloud first

    range_image_.setTo(std::numeric_limits<float>::max());
    x_image_.setTo(0);
    y_image_.setTo(0);
    z_image_.setTo(0);
    valid_mask_.setTo(0);
    
    for (const auto& pt : cloud->points) {
        uint16_t ring = static_cast<uint16_t>(pt.normal_x); // Retrieve ring from normal_x
        if (ring >= num_rings) continue;
        
        float azimuth = std::atan2(pt.y, pt.x);
        if (azimuth < 0) azimuth += 2 * M_PI;
        
        int col = static_cast<int>(azimuth / (2 * M_PI) * num_sectors);
        col = std::min(col, num_sectors - 1);
        
        int row = ring;
        
        float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        
        if (range < range_image_.at<float>(row, col)) {
            range_image_.at<float>(row, col) = range;
            x_image_.at<float>(row, col) = pt.x;
            y_image_.at<float>(row, col) = pt.y;
            z_image_.at<float>(row, col) = pt.z;
            valid_mask_.at<uint8_t>(row, col) = 1;
        }
    }
}

// Removed segmentGroundByNormal() as its logic is replaced by PatchWork++ inspired methods

Eigen::Vector3f RangeImageObstacleDetector::computeNormal(int row, int col) {
    // find vertical neighbor, handle boundary, todo: consider handle horizontal boundary?
    int row_ver = (row < num_rings - 1) ? (row + 1) : (row - 1);
    int col_right = (col + 1) % num_sectors;
    
    if (!valid_mask_.at<uint8_t>(row, col_right) || !valid_mask_.at<uint8_t>(row_ver, col)) {
        return Eigen::Vector3f(0, 0, 1);
    }
    
    Eigen::Vector3f p(x_image_.at<float>(row, col),
                     y_image_.at<float>(row, col),
                     z_image_.at<float>(row, col));
    
    Eigen::Vector3f p_right(x_image_.at<float>(row, col_right),
                           y_image_.at<float>(row, col_right),
                           z_image_.at<float>(row, col_right));
    
    Eigen::Vector3f p_ver(x_image_.at<float>(row_ver, col),
                          y_image_.at<float>(row_ver, col),
                          z_image_.at<float>(row_ver, col));
    
    Eigen::Vector3f v1 = p_right - p;
    Eigen::Vector3f v2 = p_ver - p;
    
    // discontinuity check, if p is far away (0.5m) from its neighbor, they do not belong to the same object, 0.5 is ok for 10m detector, if further, v2.norm() > 0.5f might need to change
    if (v1.norm() > 0.5f || v2.norm() > 0.5f) {
        return Eigen::Vector3f(0, 0, 1);
    }
    
    Eigen::Vector3f normal = v1.cross(v2);
    
    if (normal.norm() > 1e-6) {
        normal.normalize();
    } else {
        return Eigen::Vector3f(0, 0, 1);
    }
    
    if (normal.z() < 0) {
        normal = -normal;
    }
    
    return normal;
}

// Removed visualizeRangeImage() and visualizeNormals() as they are not explicitly requested to be retained

void RangeImageObstacleDetector::saveNormalsToPCD(const std::string& path) {
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(
        new pcl::PointCloud<pcl::PointXYZINormal>);

    for (int row = 0; row < num_rings; ++row) {
        for (int col = 1; col < num_sectors - 1; ++col) {
            if (!valid_mask_.at<uint8_t>(row, col)) continue;

            float x = x_image_.at<float>(row, col);
            float y = y_image_.at<float>(row, col);
            float z = z_image_.at<float>(row, col);

            Eigen::Vector3f normal = computeNormal(row, col);

            pcl::PointXYZINormal pt_normal;
            pt_normal.x = x;
            pt_normal.y = y;
            pt_normal.z = z;
            pt_normal.normal_x = row; // normal.x()
            pt_normal.normal_y = col; // normal.y()
            pt_normal.normal_z = normal.z();
            pt_normal.intensity = std::abs(normal.z()); // Map abs(normal.z) to intensity for CloudCompare visualization
            cloud_with_normals->points.push_back(pt_normal);
        }
    }

    cloud_with_normals->width = cloud_with_normals->points.size();
    cloud_with_normals->height = 1;
    cloud_with_normals->is_dense = true;
    pcl::io::savePCDFileBinary(path, *cloud_with_normals);
}

std::vector<BoundingBox> getObstacleBoundingBoxes(
    pcl::PointCloud<pcl::PointXYZI>::Ptr obstacles) {
    
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(obstacles);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(0.3);  // 30cm cluster distance
    ec.setMinClusterSize(20);     // Min 20 points
    ec.setMaxClusterSize(10000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(obstacles);
    ec.extract(cluster_indices);
    
    std::vector<BoundingBox> bboxes;
    for (const auto& indices : cluster_indices) {
        pcl::PointXYZ min_pt, max_pt;
        
        min_pt.x = min_pt.y = min_pt.z = std::numeric_limits<float>::max();
        max_pt.x = max_pt.y = max_pt.z = -std::numeric_limits<float>::max();
        
        for (int idx : indices.indices) {
            const auto& pt = obstacles->points[idx];
            min_pt.x = std::min(min_pt.x, pt.x);
            min_pt.y = std::min(min_pt.y, pt.y);
            min_pt.z = std::min(min_pt.z, pt.z);
            max_pt.x = std::max(max_pt.x, pt.x);
            max_pt.y = std::max(max_pt.y, pt.y);
            max_pt.z = std::max(max_pt.z, pt.z);
        }
        
        BoundingBox bbox;
        bbox.min_point = min_pt;
        bbox.max_point = max_pt;
        bbox.center.x = (min_pt.x + max_pt.x) / 2;
        bbox.center.y = (min_pt.y + max_pt.y) / 2;
        bbox.center.z = (min_pt.z + max_pt.z) / 2;
        
        float volume = (max_pt.x - min_pt.x) * 
                      (max_pt.y - min_pt.y) * 
                      (max_pt.z - min_pt.z);
        
        if (volume > 0.001) {  // Minimum volume threshold
            bboxes.push_back(bbox);
        }
    }
    
    return bboxes;
}

void RangeImageObstacleDetector::saveGroundToPCD(const std::string& path) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud_pcd(
        new pcl::PointCloud<pcl::PointXYZI>);

    for (const auto& pt_local : cloud_ground_local_) {
        pcl::PointXYZI pt_pcd;
        pt_pcd.x = pt_local.x;
        pt_pcd.y = pt_local.y;
        pt_pcd.z = pt_local.z;
        pt_pcd.intensity = pt_local.intensity;
        ground_cloud_pcd->points.push_back(pt_pcd);
    }

    ground_cloud_pcd->width = ground_cloud_pcd->points.size();
    ground_cloud_pcd->height = 1;
    ground_cloud_pcd->is_dense = true;
    pcl::io::savePCDFileBinary(path, *ground_cloud_pcd);
}

void RangeImageObstacleDetector::saveNongroundBeforeClusteringToPCD(const std::string& path) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr nonground_cloud_pcd(
        new pcl::PointCloud<pcl::PointXYZI>);

    for (const auto& pt_local : cloud_nonground_local_) {
        pcl::PointXYZI pt_pcd;
        pt_pcd.x = pt_local.x;
        pt_pcd.y = pt_local.y;
        pt_pcd.z = pt_local.z;
        pt_pcd.intensity = pt_local.intensity;
        nonground_cloud_pcd->points.push_back(pt_pcd);
    }

    nonground_cloud_pcd->width = nonground_cloud_pcd->points.size();
    nonground_cloud_pcd->height = 1;
    nonground_cloud_pcd->is_dense = true;
    pcl::io::savePCDFileBinary(path, *nonground_cloud_pcd);
}

void RangeImageObstacleDetector::saveRNRSkippedPointsToPCD(const std::string& path) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr rnr_skipped_cloud_pcd(
        new pcl::PointCloud<pcl::PointXYZI>);

    for (const auto& pt_local : rnr_noise_points_) {
        pcl::PointXYZI pt_pcd;
        pt_pcd.x = pt_local.x;
        pt_pcd.y = pt_local.y;
        pt_pcd.z = pt_local.z;
        pt_pcd.intensity = pt_local.intensity;
        rnr_skipped_cloud_pcd->points.push_back(pt_pcd);
    }

    rnr_skipped_cloud_pcd->width = rnr_skipped_cloud_pcd->points.size();
    rnr_skipped_cloud_pcd->height = 1;
    rnr_skipped_cloud_pcd->is_dense = true;
    pcl::io::savePCDFileBinary(path, *rnr_skipped_cloud_pcd);
}
