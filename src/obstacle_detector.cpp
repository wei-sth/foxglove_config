#include <pcl/io/pcd_io.h>
#include "obstacle_detector.h"

#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h> // Include for pcl::MomentOfInertiaEstimation
#include <pcl/surface/convex_hull.h> // For pcl::ConvexHull
#include <algorithm>
#include <iostream> // For std::cout
#include <fstream>  // For file operations (OBJ saving)
#include <pcl/common/pca.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Eigen::AngleAxisf
#include <queue> // Required for std::queue
#include <cmath> // For std::abs
#include <chrono>
#include <iomanip>

static constexpr bool LOCAL_DEBUG = false;


WildTerrainSegmenter::WildTerrainSegmenter(float max_range) : max_range(max_range) {
}

void WildTerrainSegmenter::segment(const cv::Mat& range_image, const cv::Mat& x_image, const cv::Mat& y_image, const cv::Mat& z_image, const cv::Mat& valid_mask,
                                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr& ground_cloud,
                                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr& obstacle_cloud) {
    ground_cloud->points.reserve(range_image.rows * range_image.cols);
    obstacle_cloud->points.reserve(range_image.rows * range_image.cols);

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    temp_cloud->points.reserve(range_image.rows * range_image.cols);
    std::vector<int> temp_cloud_indices_map(range_image.rows * range_image.cols, -1); // Map (row, col) to temp_cloud index

    // 1. Polar Binning
    std::vector<std::vector<Cell>> polar_grid(num_rings, std::vector<Cell>(num_sectors));
    int current_point_idx = 0;
    for (int r_img = 0; r_img < range_image.rows; ++r_img) {
        for (int c_img = 0; c_img < range_image.cols; ++c_img) {
            if (valid_mask.at<uint8_t>(r_img, c_img) == 0) continue;

            float x = x_image.at<float>(r_img, c_img);
            float y = y_image.at<float>(r_img, c_img);
            float z = z_image.at<float>(r_img, c_img);
            float range_val = range_image.at<float>(r_img, c_img);

            if (range_val > max_range) continue;

            float angle = atan2(y, x) * 180.0 / M_PI;
            if (angle < 0) angle += 360.0;

            int ring_idx = std::min(num_rings - 1, (int)(range_val / (max_range / num_rings)));
            int sector_idx = std::min(num_sectors - 1, (int)(angle / (360.0 / num_sectors)));
            
            pcl::PointXYZINormal pt;
            pt.x = x;
            pt.y = y;
            pt.z = z;
            pt.intensity = range_val; // range as intensity
            pt.normal_x = static_cast<float>(r_img);
            pt.normal_y = static_cast<float>(c_img);
            // normal_z not assigned
            temp_cloud->points.push_back(pt);
            polar_grid[ring_idx][sector_idx].point_indices.push_back(current_point_idx);
            temp_cloud_indices_map[r_img * range_image.cols + c_img] = current_point_idx;
            current_point_idx++;
        }
    }
    temp_cloud->width = temp_cloud->points.size();
    temp_cloud->height = 1;
    temp_cloud->is_dense = true;

    if constexpr (LOCAL_DEBUG) {
        // float target_x = 7.693483;
        // float target_y = -0.854771;
        // float target_z = -0.574363;
        // float target_x = 0.939290;
        // float target_y = 0.925231;
        // float target_z = 0.096044;
        float target_x = -1.393945;
        float target_y = 1.085931;
        float target_z = 3.254558;
        
        float target_range = std::sqrt(target_x * target_x + target_y * target_y + target_z * target_z);
        float target_angle = atan2(target_y, target_x) * 180.0 / M_PI;
        if (target_angle < 0) target_angle += 360.0;
        debug_r = std::min(num_rings - 1, (int)(target_range / (max_range / num_rings)));
        debug_s = std::min(num_sectors - 1, (int)(target_angle / (360.0 / num_sectors)));
        std::cout << "debug point (" << target_x << ", " << target_y << ", " << target_z << ") belongs to Cell [" << debug_r << ", " << debug_s << "]" << std::endl;

        debugSavePolarGrid(polar_grid, temp_cloud, "/home/weizh/data/polar_grid.pcd");
    }

    // 2. Process each cell
    struct PlaneParams {
        Eigen::Vector3f normal;
        float d;
        bool valid = false;
    };
    std::vector<std::vector<PlaneParams>> cell_planes(num_rings, std::vector<PlaneParams>(num_sectors));

    for (int r = 0; r < num_rings; ++r) {
        for (int s = 0; s < num_sectors; ++s) {
            const auto& indices = polar_grid[r][s].point_indices;
            if (indices.size() < 10) continue; // Too few points to fit

            // Extract seed points (Lowest Point Representative)
            std::vector<int> seed_indices = extract_seeds(temp_cloud, indices);

            // Iteratively fit plane
            Eigen::Vector3f normal;
            float d;
            float linearity;
            estimate_plane(temp_cloud, seed_indices, normal, d, linearity);

            if constexpr (LOCAL_DEBUG) {
                if (r == debug_r && s == debug_s) {
                    std::cout << "Cell [" << r << ", " << s << "] Initial Plane: normal=" << normal.transpose() << ", d=" << d << ", seeds=" << seed_indices.size() << std::endl;
                    
                    // Save cell cloud
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cell_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
                    for (int idx : indices) cell_cloud->push_back(temp_cloud->points[idx]);
                    pcl::io::savePCDFileBinary("/home/weizh/data/debug_cell.pcd", *cell_cloud);

                    // Save seeds
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr seed_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
                    for (int idx : seed_indices) seed_cloud->push_back(temp_cloud->points[idx]);
                    pcl::io::savePCDFileBinary("/home/weizh/data/debug_cell_seed.pcd", *seed_cloud);
                }
            }

            for (int iter = 0; iter < num_iter; ++iter) {
                seed_indices.clear();
                for (int idx : indices) {
                    const auto& pt = temp_cloud->points[idx].getVector3fMap();
                    // Calculate vertical distance from point to plane
                    float dist = normal.dot(pt) + d;
                    // Points close enough are considered ground points and used for the next iteration
                    if (dist < dist_threshold) {
                        seed_indices.push_back(idx);
                    }
                }
                if (seed_indices.size() < 5) break;
                estimate_plane(temp_cloud, seed_indices, normal, d, linearity);
            }

            cell_planes[r][s].normal = normal;
            cell_planes[r][s].d = d;
            cell_planes[r][s].valid = true;

            // if (linearity < 0.01f) {
            //     std::cout << "Cell [" << r << ", " << s << "] linearity=" << linearity << std::endl;
            // }

            if constexpr (LOCAL_DEBUG) {
                if (r == debug_r && s == debug_s) {
                    std::cout << "Cell [" << r << ", " << s << "] Final Plane: normal=" << normal.transpose() << ", d=" << d << ", seeds=" << seed_indices.size() << std::endl;
                    // Save inliers (final seed_indices after iterations are the inliers)
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
                    for (int idx : seed_indices) inlier_cloud->push_back(temp_cloud->points[idx]);
                    pcl::io::savePCDFileBinary("/home/weizh/data/debug_cell_final_inlier.pcd", *inlier_cloud);
                }
            }

            // 3. Final classification
            bool contain_obstacle = false;
            for (int idx : indices) {
                const auto& pt = temp_cloud->points[idx].getVector3fMap();
                float dist = normal.dot(pt) + d;

                // use normal_z_threshold might also classify some ground points in this bin as obstacle, need to revise, see target_x = 0.939290
                if (dist > dist_threshold || std::abs(normal.z()) < normal_z_threshold) {
                    obstacle_cloud->points.push_back(temp_cloud->points[idx]);
                    contain_obstacle = true;
                } else if (dist < -dist_threshold) {
                    // Negative values represent "depressions", which could be pits that the mower must avoid
                    obstacle_cloud->points.push_back(temp_cloud->points[idx]);
                    contain_obstacle = true;
                } else {
                    ground_cloud->points.push_back(temp_cloud->points[idx]);
                }
            }

            // if (contain_obstacle && linearity < 0.01f) {
            //     std::cout << "Cell [" << r << ", " << s << "] contains obstacle, linearity=" << linearity << std::endl;
            // }
        }
    }

    // // 4. Global Consistency Check (Reporting only)
    // for (int r = 1; r < num_rings; ++r) {
    //     for (int s = 0; s < num_sectors; ++s) {
    //         if (cell_planes[r][s].valid && cell_planes[r-1][s].valid) {
    //             // Only check if both are ground-like (not walls)
    //             if (std::abs(cell_planes[r][s].normal.z()) > normal_z_threshold &&
    //                 std::abs(cell_planes[r-1][s].normal.z()) > normal_z_threshold) {
                    
    //                 float R = r * (max_range / num_rings);
    //                 float angle_deg = (s + 0.5f) * (360.0f / num_sectors);
    //                 float angle_rad = angle_deg * M_PI / 180.0f;
    //                 float x = R * std::cos(angle_rad);
    //                 float y = R * std::sin(angle_rad);
                    
    //                 float z_prev = -(cell_planes[r-1][s].normal.x() * x + cell_planes[r-1][s].normal.y() * y + cell_planes[r-1][s].d) / cell_planes[r-1][s].normal.z();
    //                 float z_curr = -(cell_planes[r][s].normal.x() * x + cell_planes[r][s].normal.y() * y + cell_planes[r][s].d) / cell_planes[r][s].normal.z();
                    
    //                 if (std::abs(z_prev - z_curr) > 0.15f) {
    //                     std::cout << "[Consistency Check] Radial discontinuity detected between Cell [" << r-1 << "," << s << "] and [" << r << "," << s << "]: delta_z=" << std::abs(z_prev - z_curr) << "m" << std::endl;
    //                 }
    //             }
    //         }
    //     }
    // }

    // // Angular check
    // for (int r = 0; r < num_rings; ++r) {
    //     for (int s = 0; s < num_sectors; ++s) {
    //         int s_next = (s + 1) % num_sectors;
    //         if (cell_planes[r][s].valid && cell_planes[r][s_next].valid) {
    //             if (std::abs(cell_planes[r][s].normal.z()) > normal_z_threshold &&
    //                 std::abs(cell_planes[r][s_next].normal.z()) > normal_z_threshold) {
                    
    //                 float R = (r + 0.5f) * (max_range / num_rings);
    //                 float angle_rad = (s + 1.0f) * (360.0f / num_sectors) * M_PI / 180.0f;
    //                 float x = R * std::cos(angle_rad);
    //                 float y = R * std::sin(angle_rad);
                    
    //                 float z_curr = -(cell_planes[r][s].normal.x() * x + cell_planes[r][s].normal.y() * y + cell_planes[r][s].d) / cell_planes[r][s].normal.z();
    //                 float z_next = -(cell_planes[r][s_next].normal.x() * x + cell_planes[r][s_next].normal.y() * y + cell_planes[r][s_next].d) / cell_planes[r][s_next].normal.z();
                    
    //                 if (std::abs(z_curr - z_next) > 0.15f) {
    //                     std::cout << "[Consistency Check] Angular discontinuity detected between Cell [" << r << "," << s << "] and [" << r << "," << s_next << "]: delta_z=" << std::abs(z_curr - z_next) << "m" << std::endl;
    //                 }
    //             }
    //         }
    //     }
    // }
}

std::vector<int> WildTerrainSegmenter::extract_seeds(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, const std::vector<int>& indices) {
    std::vector<int> sorted_indices = indices;
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int i, int j) {
        return cloud->points[i].z < cloud->points[j].z;
    });
    
    int n = std::min((int)sorted_indices.size(), num_lpr);
    return std::vector<int>(sorted_indices.begin(), sorted_indices.begin() + n);
}

void WildTerrainSegmenter::estimate_plane(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, const std::vector<int>& indices, 
                                        Eigen::Vector3f& normal, float& d, float& linearity) {
    Eigen::Vector3f mean(0, 0, 0);
    for (int idx : indices) mean += cloud->points[idx].getVector3fMap();
    mean /= indices.size();

    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (int idx : indices) {
        Eigen::Vector3f vec = cloud->points[idx].getVector3fMap() - mean;
        cov += vec * vec.transpose();
    }

    // The eigenvector corresponding to the smallest eigenvalue is the plane normal
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    Eigen::Vector3f eigenvalues = solver.eigenvalues(); // e0 <= e1 <= e2
    // if e0 and e1 are very small compared with e2 -- line, if linearity too small, need double check
    linearity = (eigenvalues(2) > 1e-4) ? (eigenvalues(1) / eigenvalues(2)) : 0.0f;
    normal = solver.eigenvectors().col(0);
    
    // Ensure the normal vector points upwards (for the robot coordinate system)
    if (normal.z() < 0) normal = -normal;
    d = -(normal.dot(mean));
}

void WildTerrainSegmenter::debugSavePolarGrid(const std::vector<std::vector<Cell>>& polar_grid, const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud_in, const std::string& path) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr binned_cloud_debug(new pcl::PointCloud<pcl::PointXYZI>);
    for (int r = 0; r < num_rings; ++r) {
        for (int s = 0; s < num_sectors; ++s) {
            const auto& indices = polar_grid[r][s].point_indices;
            if (!indices.empty()) {
                // Assign a distinct intensity based on bin indices for better visualization of boundaries.
                // Using a checkerboard pattern (0 or 255) to clearly delineate bin boundaries.
                float bin_intensity = static_cast<float>(((r + s) % 2) * 255); 
                for (int idx : indices) {
                    pcl::PointXYZI pt_xyzi;
                    pt_xyzi.x = cloud_in->points[idx].x;
                    pt_xyzi.y = cloud_in->points[idx].y;
                    pt_xyzi.z = cloud_in->points[idx].z;
                    pt_xyzi.intensity = bin_intensity;
                    binned_cloud_debug->points.push_back(pt_xyzi);
                }
            }
        }
    }
    pcl::io::savePCDFileBinary(path, *binned_cloud_debug);
}

// ring index = row index
// note xt16, ring 15 is the lowest, ring 0 is the highest, ring 15 and ring 14 angle difference is 2 degree
// due to beam divergence, the further from lidar (assume r is distance), the larger vertical distance between rings, 
// delta_z = r * tan(2 degree) = r * 0.0349
// so if r = 2.0, even for a vertical wall, the vertical distance would be less than 0.07, 
// but if r=20.0, for a slope, the vertical distance between rings might easily exceed 0.2m
// so while doing clustering, need to consider connectivity, also need to consider physical distance:
// 1. same col, row and row+1, distance is far, but they indeed belong to the same object (pole)
// 2. same row, col and col+1, distance is far, but they belong to different objects

RangeImageObstacleDetector::RangeImageObstacleDetector(int num_rings, int num_sectors, 
                               float max_range, float min_cluster_z_difference, VisResultType vis_type)
    : num_rings(num_rings), num_sectors(num_sectors), max_range(max_range),
      min_cluster_z_difference_(min_cluster_z_difference), vis_type_(vis_type),
      obstacle_grid_flat_(num_rings * num_sectors, nullptr), // Initialize flattened grid
      temp_valid_mask_(num_rings, num_sectors, CV_8UC1, cv::Scalar(0)), // Initialize temp valid mask
      visited_mask_(num_rings, num_sectors, CV_8UC1, cv::Scalar(0)), // Initialize visited mask
      wild_terrain_segmenter_(max_range)
{
    range_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, 
                           cv::Scalar(std::numeric_limits<float>::max()));
    x_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    y_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    z_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    valid_mask_ = cv::Mat(num_rings, num_sectors, CV_8UC1, cv::Scalar(0));

    updateSensorTransform();
}

void RangeImageObstacleDetector::updateSensorTransform() {
    apply_sensor_transform_ = true;
    // only keep rotation, the last row and col must be 0,0,0,1, since filter by distance is distance from lidar origin (0,0,0), and range image is built on lidar origin, so the lidar origin should not change
    // mower
    // sensor_transform_.matrix() << 0.882501, -0.0144251, 0.470089, 0,
    //                              -0, 0.99953, 0.0306714, 0,
    //                              -0.470311, -0.0270675, 0.882086, 0,
    //                              0, 0, 0, 1;
    // mower, front-left-up
    sensor_transform_.matrix() << 0.883228, -0.0186038, 0.468574, 0,
                                  0, 0.999213, 0.0396717, 0,
                                  -0.468943, -0.0350392, 0.882533, 0,
                                  0, 0, 0, 1;
    // front-left-up to right-front-up
    sensor_transform_.prerotate((Eigen::Matrix3f() << 0, -1, 0, 1, 0, 0, 0, 0, 1).finished());
    sensor_inv_transform_ = sensor_transform_.inverse();

    // if (std::abs(sensor_roll) < 1e-6 && std::abs(sensor_pitch) < 1e-6 && std::abs(sensor_yaw) < 1e-6) {
    //     apply_sensor_transform_ = false;
    //     sensor_transform_ = Eigen::Affine3f::Identity();
    //     sensor_inv_transform_ = Eigen::Affine3f::Identity();
    // } else {
    //     apply_sensor_transform_ = true;
    //     // ROS standard: Extrinsic XYZ (Roll-Pitch-Yaw)
    //     // R = Rz(yaw) * Ry(pitch) * Rx(roll)
    //     sensor_transform_ = Eigen::AngleAxisf(sensor_yaw, Eigen::Vector3f::UnitZ()) *
    //                        Eigen::AngleAxisf(sensor_pitch, Eigen::Vector3f::UnitY()) *
    //                        Eigen::AngleAxisf(sensor_roll, Eigen::Vector3f::UnitX());
    //     sensor_inv_transform_ = sensor_transform_.inverse();
    // }
    // std::cout << "--- Sensor Transform Matrix ---" << std::endl;
    // std::cout << sensor_transform_.matrix() << std::endl;
}

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> RangeImageObstacleDetector::detectObstacles(
    pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw) {
    // auto t1 = std::chrono::steady_clock::now();
    auto cloud_filtered_normal = filterByRangeEgo(cloud_raw);
    
    // auto t2 = std::chrono::steady_clock::now();
    buildRangeImage(cloud_filtered_normal);

    // auto t3 = std::chrono::steady_clock::now();
    // auto obstacles_with_normal_info = segmentGroundByNormal(); // Original function, commented out
    auto obstacles_with_normal_info = segmentGroundByPatchwork();

    // auto t4 = std::chrono::steady_clock::now();

    // double d1 = std::chrono::duration<double, std::milli>(t2 - t1).count();
    // double d2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
    // double d3 = std::chrono::duration<double, std::milli>(t4 - t3).count();
    // double total = d1 + d2 + d3;
    // if (total > 0) {
    //     std::cout << std::fixed << std::setprecision(2);
    //     std::cout << "total:" << total << "ms,filterByRangeEgo:" << d1 << "ms,buildRangeImage:" << d2 << "ms,segmentGroundByPatchwork:" << d3 << std::endl;
    // }

    // debug
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_lidar(new pcl::PointCloud<pcl::PointXYZINormal>);  // in lidar frame
    pcl::copyPointCloud(*obstacles_with_normal_info, *obstacles_lidar);
    if (apply_sensor_transform_) {
        pcl::transformPointCloud(*obstacles_lidar, *obstacles_lidar, sensor_inv_transform_);
    }
    pcl::io::savePCDFileBinary("/home/weizh/data/obstacles_before_cluster.pcd", *obstacles_lidar);
    pcl::io::savePCDFileBinary("/home/weizh/data/obstacles_before_cluster_transformed.pcd", *obstacles_with_normal_info);

    // Step 4: Perform clustering using Euclidean method
    // return clusterEuclidean(obstacles_with_normal_info);
    auto clusters = clusterConnectivity(obstacles_with_normal_info);

    bboxes_lidar_frame_ = getObstacleBBoxesFromGround(clusters);

    // Step 5: Transform clusters back to original sensor frame
    if (apply_sensor_transform_) {
        for (auto& cluster : clusters) {
            pcl::transformPointCloud(*cluster, *cluster, sensor_inv_transform_);
        }
    }

    return clusters;
}

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> RangeImageObstacleDetector::detectObstacles(
    pcl::PointCloud<RSPointDefault>::Ptr cloud_raw) {
    // auto t1 = std::chrono::steady_clock::now();
    auto cloud_filtered_normal = filterByRangeEgo(cloud_raw);
    buildRangeImage(cloud_filtered_normal);
    // auto obstacles_with_normal_info = segmentGroundByNormal(); // Original function, commented out
    auto obstacles_with_normal_info = segmentGroundByPatchwork();

    if constexpr (LOCAL_DEBUG) {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_lidar(new pcl::PointCloud<pcl::PointXYZINormal>);  // in lidar frame
        pcl::copyPointCloud(*obstacles_with_normal_info, *obstacles_lidar);
        if (apply_sensor_transform_) {
            pcl::transformPointCloud(*obstacles_lidar, *obstacles_lidar, sensor_inv_transform_);
        }
        pcl::io::savePCDFileBinary("/home/weizh/data/obstacles_before_cluster.pcd", *obstacles_lidar);
        pcl::io::savePCDFileBinary("/home/weizh/data/obstacles_before_cluster_transformed.pcd", *obstacles_with_normal_info);
    }

    // return clusterEuclidean(obstacles_with_normal_info);
    auto clusters = clusterConnectivity(obstacles_with_normal_info);

    if (vis_type_ == VisResultType::BBOX_GROUND) {
        bboxes_lidar_frame_ = getObstacleBBoxesFromGround(clusters);
    }
    else if (vis_type_ == VisResultType::BBOX_GROUND_2D || vis_type_ == VisResultType::BBOX_2D_AND_VOXEL) {
        return clusters;
    }

    // transform clusters back to original sensor frame
    if (apply_sensor_transform_) {
        for (auto& cluster : clusters) {
            pcl::transformPointCloud(*cluster, *cluster, sensor_inv_transform_);
        }
    }

    if (vis_type_ == VisResultType::BBOX_LIDAR_XY) {
        bboxes_lidar_frame_ = getObstacleBBoxes(clusters);
    }

    // auto t2 = std::chrono::steady_clock::now();
    // double total_d = std::chrono::duration<double, std::milli>(t2 - t1).count();
    // std::cout << std::fixed << std::setprecision(2);
    // std::cout << "total duration:" << total_d << "ms" << std::endl;

    return clusters;
}

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> RangeImageObstacleDetector::clusterEuclidean(
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_with_normal_info) {
    
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clustered_obstacles_vector;
    
    if (obstacles_with_normal_info->points.empty()) {
        return clustered_obstacles_vector;
    }

    float current_intensity = 0.0f; // Initialize intensity for clusters

    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZINormal>);
    tree->setInputCloud(obstacles_with_normal_info);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZINormal> ec;
    ec.setClusterTolerance(0.3); // 30cm
    ec.setMinClusterSize(20);
    ec.setMaxClusterSize(10000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(obstacles_with_normal_info);
    ec.extract(cluster_indices);

    for (const auto& indices : cluster_indices) {
        std::set<uint16_t> rings_in_cluster;
        float min_z_cluster = std::numeric_limits<float>::max();
        float max_z_cluster = -std::numeric_limits<float>::max();
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr current_cluster_xyzi(new pcl::PointCloud<pcl::PointXYZI>);

        for (int idx : indices.indices) {
            const auto& pt = obstacles_with_normal_info->points[idx];
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
            clustered_obstacles_vector.push_back(current_cluster_xyzi);
            current_intensity += 10.0f; // Increment intensity for the next cluster
        }
    }
    
    return clustered_obstacles_vector;
}

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> RangeImageObstacleDetector::clusterConnectivity(
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_with_normal_info) {
    
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clustered_obstacles_vector;
    
    if (obstacles_with_normal_info->points.empty()) {
        return clustered_obstacles_vector;
    }

    // Clear and reset member variables for current frame
    std::fill(obstacle_grid_flat_.begin(), obstacle_grid_flat_.end(), nullptr); // Reset all pointers to nullptr
    temp_valid_mask_.setTo(0); // Reset all values to 0

    for (const auto& pt : obstacles_with_normal_info->points) {
        uint16_t ring = static_cast<uint16_t>(pt.normal_x); // Retrieve ring (row)
        int col = static_cast<int>(pt.normal_y); // Directly use normal_y as col

        if (ring < num_rings && col >= 0 && col < num_sectors) {
            int flat_idx = ring * num_sectors + col;
            obstacle_grid_flat_[flat_idx] = &pt;
            temp_valid_mask_.at<uint8_t>(ring, col) = 1;
        }
    }

    visited_mask_.setTo(0); // Reset visited mask for current frame
    float current_intensity = 0.0f;

    // Define 8-connectivity neighbors (dr, dc)
    int dr[] = {-1, -1, -1,  0, 0,  1, 1, 1};
    int dc[] = {-1,  0,  1, -1, 1, -1, 0, 1};

    for (int r = 0; r < num_rings; ++r) {
        for (int c = 0; c < num_sectors; ++c) {
            if (temp_valid_mask_.at<uint8_t>(r, c) == 1 && visited_mask_.at<uint8_t>(r, c) == 0) {
                // Start a new cluster
                pcl::PointCloud<pcl::PointXYZI>::Ptr current_cluster_xyzi(new pcl::PointCloud<pcl::PointXYZI>);
                std::set<uint16_t> rings_in_cluster;
                float min_z_cluster = std::numeric_limits<float>::max();
                float max_z_cluster = -std::numeric_limits<float>::max();

                std::queue<std::pair<int, int>> q;
                q.push({r, c});
                visited_mask_.at<uint8_t>(r, c) = 1;

                while (!q.empty()) {
                    std::pair<int, int> current_rc = q.front();
                    q.pop();
                    int cur_r = current_rc.first;
                    int cur_c = current_rc.second;

                    int flat_idx = cur_r * num_sectors + cur_c;
                    const pcl::PointXYZINormal* pt_normal = obstacle_grid_flat_[flat_idx];
                    if (pt_normal) { // Should always be true if temp_valid_mask is 1
                        // Add point to current cluster
                        pcl::PointXYZI pt_xyzi;
                        pt_xyzi.x = pt_normal->x;
                        pt_xyzi.y = pt_normal->y;
                        pt_xyzi.z = pt_normal->z;
                        pt_xyzi.intensity = current_intensity;
                        current_cluster_xyzi->points.push_back(pt_xyzi);

                        rings_in_cluster.insert(cur_r); // Ring is the row index
                        min_z_cluster = std::min(min_z_cluster, pt_xyzi.z);
                        max_z_cluster = std::max(max_z_cluster, pt_xyzi.z);
                    }

                    // Explore neighbors
                    for (int i = 0; i < 8; ++i) {
                        int next_r = cur_r + dr[i];
                        int next_c = (cur_c + dc[i] + num_sectors) % num_sectors; // Handle circular wrap-around for columns

                        if (next_r >= 0 && next_r < num_rings &&
                            temp_valid_mask_.at<uint8_t>(next_r, next_c) == 1 && // Use temp_valid_mask_
                            visited_mask_.at<uint8_t>(next_r, next_c) == 0) {
                            
                            // Get current point's range (from intensity)
                            const pcl::PointXYZINormal* current_pt_normal = obstacle_grid_flat_[flat_idx];
                            float current_range = current_pt_normal ? current_pt_normal->intensity : std::numeric_limits<float>::max();

                            // Get neighbor point's range (from intensity)
                            int next_flat_idx = next_r * num_sectors + next_c;
                            const pcl::PointXYZINormal* next_pt_normal = obstacle_grid_flat_[next_flat_idx];
                            float next_range = next_pt_normal ? next_pt_normal->intensity : std::numeric_limits<float>::max();

                            // Check range difference
                            if (std::abs(current_range - next_range) <= 0.3f) { // Only cluster if range difference is within 0.3m
                                q.push({next_r, next_c});
                                visited_mask_.at<uint8_t>(next_r, next_c) = 1;
                            }
                        }
                    }
                }

                // Filter clusters based on the number of rings spanned AND Z-difference
                if (current_cluster_xyzi->points.size() >= 10 && rings_in_cluster.size() >= 2 && (max_z_cluster - min_z_cluster) > min_cluster_z_difference_) {
                    clustered_obstacles_vector.push_back(current_cluster_xyzi);
                    current_intensity += 10.0f;
                }
            }
        }
    }
    
    return clustered_obstacles_vector;
}

pcl::PointCloud<pcl::PointXYZINormal>::Ptr RangeImageObstacleDetector::filterByRangeEgo(pcl::PointCloud<PointXYZIRT>::Ptr cloud) {
    // Filter by distance, convert to PointXYZINormal, transform xy plane to ground
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr filtered_normal(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    filtered_normal->points.reserve(cloud->points.size()); // Reserve space
    
    for (const auto& pt_irt : cloud->points) {
        Eigen::Vector3f p(pt_irt.x, pt_irt.y, pt_irt.z);
        if (apply_sensor_transform_) {
            p = sensor_transform_ * p;
        }
        float distance = p.norm();
        
        if (distance <= max_range && distance > 0.5f) {  // Min distance 0.5m
            pcl::PointXYZINormal pt_normal;
            pt_normal.x = p.x();
            pt_normal.y = p.y();
            pt_normal.z = p.z();
            pt_normal.intensity = pt_irt.intensity;
            pt_normal.normal_x = static_cast<float>(pt_irt.ring); // Store ring in normal_x
            pt_normal.normal_y = pt_irt.time; // Store time in normal_y
            // normal_z, curvature can be left as default or set to 0
            
            filtered_normal->points.push_back(pt_normal);
        }
    }
    
    return filtered_normal;
}

pcl::PointCloud<pcl::PointXYZINormal>::Ptr RangeImageObstacleDetector::filterByRangeEgo(pcl::PointCloud<RSPointDefault>::Ptr cloud) {
    // Filter by distance, convert to PointXYZINormal, transform xy plane to ground
    // add remove z>2.0, since now coords are already in ground frame, 2026/01/21
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr filtered_normal(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    filtered_normal->points.reserve(cloud->points.size()); // Reserve space
    
    for (const auto& pt : cloud->points) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        if (apply_sensor_transform_) {
            p = sensor_transform_ * p;
        }
        float distance = p.norm();
        
        if (distance <= max_range && distance > 0.5f && p.z()<2.0) {  // Min distance 0.5m
            pcl::PointXYZINormal pt_normal;
            pt_normal.x = p.x();
            pt_normal.y = p.y();
            pt_normal.z = p.z();
            pt_normal.intensity = pt.intensity;
            pt_normal.normal_x = static_cast<float>(pt.ring); // Store ring in normal_x
            pt_normal.normal_y = pt.timestamp; // Store time in normal_y
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

pcl::PointCloud<pcl::PointXYZINormal>::Ptr RangeImageObstacleDetector::segmentGroundByNormal() {
    // use row + 1 (larger ring is lower, from ground to top, in case ring 6 at one obstacle and ring 7 at another obstacle, miss the highest obstacle ring) 
    // by default, unless last ring
    
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_with_normal_info(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    
    for (int row = 0; row < num_rings; ++row) {
        for (int col = 0; col < num_sectors; ++col) {
            
            if (!valid_mask_.at<uint8_t>(row, col)) continue;

            float x = x_image_.at<float>(row, col);
            float y = y_image_.at<float>(row, col);
            float z = z_image_.at<float>(row, col);
            
            bool z_trend_condition = false;
            float slope_threshold = 1.0f; // 45 degree

            if (row == num_rings - 1) { // Special case for the last ring (lowest)
                if (valid_mask_.at<uint8_t>(row - 1, col)) {
                    float z_current = z_image_.at<float>(row, col);
                    float x_prev = x_image_.at<float>(row - 1, col);
                    float y_prev = y_image_.at<float>(row - 1, col);
                    float z_prev = z_image_.at<float>(row - 1, col);

                    float dz_prev = z_prev - z_current; // Previous is higher than current
                    float dx_prev = x_prev - x;
                    float dy_prev = y_prev - y;
                    float d_xy_prev = std::sqrt(dx_prev*dx_prev + dy_prev*dy_prev);
                    z_trend_condition = (dz_prev > (d_xy_prev * slope_threshold));
                }
            }
            else {
                if (valid_mask_.at<uint8_t>(row + 1, col)) {
                    float z_current = z_image_.at<float>(row, col);
                    float x_next = x_image_.at<float>(row + 1, col);
                    float y_next = y_image_.at<float>(row + 1, col);
                    float z_next = z_image_.at<float>(row + 1, col);

                    float dz_next = z_current - z_next; // Current is higher than next
                    float dx_next = x - x_next;
                    float dy_next = y - y_next;
                    float d_xy_next = std::sqrt(dx_next*dx_next + dy_next*dy_next);
                    z_trend_condition = (dz_next > (d_xy_next * slope_threshold));
                }
            }
                
            Eigen::Vector3f normal = computeNormal(row, col);
            
            float normal_z_threshold = 0.7f;
            if (std::abs(normal.z()) < normal_z_threshold && z_trend_condition) {
                pcl::PointXYZINormal obstacle_pt;
                obstacle_pt.x = x;
                obstacle_pt.y = y;
                obstacle_pt.z = z;
                obstacle_pt.normal_x = static_cast<float>(row);
                obstacle_pt.normal_y = static_cast<float>(col);
                obstacle_pt.normal_z = normal.z();
                obstacle_pt.intensity = range_image_.at<float>(row, col);
                obstacles_with_normal_info->points.push_back(obstacle_pt);
            }
        }
    }
    
    return obstacles_with_normal_info;
}

pcl::PointCloud<pcl::PointXYZINormal>::Ptr RangeImageObstacleDetector::segmentGroundByPatchwork() {
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    wild_terrain_segmenter_.segment(range_image_, x_image_, y_image_, z_image_, valid_mask_, ground_cloud, obstacle_cloud);
    return obstacle_cloud;
}

Eigen::Vector3f RangeImageObstacleDetector::computeNormal(int row, int col) {
    // find vertical and horizontal neighbor, handle boundary, default use row+1 because row+1 is lower (ground)
    int row_ver = (row < num_rings - 1) ? (row + 1) : (row - 1); // lower
    int col_hor = (col + 1) % num_sectors;  // right
    
    if (!valid_mask_.at<uint8_t>(row, col_hor) || !valid_mask_.at<uint8_t>(row_ver, col)) {
        return Eigen::Vector3f(0, 0, 1);
    }
    
    Eigen::Vector3f p(x_image_.at<float>(row, col),
                     y_image_.at<float>(row, col),
                     z_image_.at<float>(row, col));
    
    Eigen::Vector3f p_hor(x_image_.at<float>(row, col_hor),
                           y_image_.at<float>(row, col_hor),
                           z_image_.at<float>(row, col_hor));
    
    Eigen::Vector3f p_ver(x_image_.at<float>(row_ver, col),
                          y_image_.at<float>(row_ver, col),
                          z_image_.at<float>(row_ver, col));
    
    Eigen::Vector3f v1 = p_hor - p;
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

void RangeImageObstacleDetector::saveNormalsToPCD(const std::string& path) {
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(
        new pcl::PointCloud<pcl::PointXYZINormal>);

    for (int row = 0; row < num_rings; ++row) {
        for (int col = 0; col < num_sectors; ++col) {
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

std::vector<RotatedBoundingBox> RangeImageObstacleDetector::getObstacleBBoxesPCA(
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters) {
    
    std::vector<RotatedBoundingBox> rotated_bboxes;
    for (const auto& current_cluster : clusters) {
        if (current_cluster->points.empty()) continue;

        float min_z_cluster = std::numeric_limits<float>::max();
        float max_z_cluster = -std::numeric_limits<float>::max();
        for (const auto& pt : current_cluster->points) {
            min_z_cluster = std::min(min_z_cluster, pt.z);
            max_z_cluster = std::max(max_z_cluster, pt.z);
        }

        // Compute PCA for the cluster
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(current_cluster);
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
        Eigen::Vector4f centroid = pca.getMean();

        // Project points to the 2D plane defined by the first two principal components
        pcl::PointCloud<pcl::PointXYZI>::Ptr projected_points(new pcl::PointCloud<pcl::PointXYZI>);
        pca.project(*current_cluster, *projected_points);

        // Find min/max in the projected 2D space
        pcl::PointXYZI min_pt_proj, max_pt_proj;
        pcl::getMinMax3D(*projected_points, min_pt_proj, max_pt_proj);

        // Calculate width, height, and angle
        // Calculate dimensions along the principal axes
        float dim[3];
        dim[0] = max_pt_proj.x - min_pt_proj.x; // Extent along 1st PC
        dim[1] = max_pt_proj.y - min_pt_proj.y; // Extent along 2nd PC
        dim[2] = max_pt_proj.z - min_pt_proj.z; // Extent along 3rd PC

        // Determine which principal component is most vertical (aligned with global Z)
        int vertical_pc_idx = 0;
        float max_z_component = std::abs(eigen_vectors(2, 0)); // Z-component of 1st PC
        if (std::abs(eigen_vectors(2, 1)) > max_z_component) {
            max_z_component = std::abs(eigen_vectors(2, 1));
            vertical_pc_idx = 1;
        }
        if (std::abs(eigen_vectors(2, 2)) > max_z_component) {
            max_z_component = std::abs(eigen_vectors(2, 2));
            vertical_pc_idx = 2;
        }

        // The other two principal components are considered horizontal
        std::vector<int> horizontal_pc_indices;
        for (int i = 0; i < 3; ++i) {
            if (i != vertical_pc_idx) {
                horizontal_pc_indices.push_back(i);
            }
        }

        float horizontal_dim1 = dim[horizontal_pc_indices[0]];
        float horizontal_dim2 = dim[horizontal_pc_indices[1]];

        // Assign width and height to the two horizontal dimensions
        // It's conventional to assign the larger one to width, but for a general box, order doesn't strictly matter
        float final_width = horizontal_dim1;
        float final_height = horizontal_dim2;

        // Calculate the angle of the "primary" horizontal principal component with the global X-axis
        // We pick the first identified horizontal PC to define the angle.
        Eigen::Vector3f primary_horizontal_pc = eigen_vectors.col(horizontal_pc_indices[0]);
        float angle = std::atan2(primary_horizontal_pc(1), primary_horizontal_pc(0)); // Angle of this PC's XY projection
        // todo: note need to revise, about angle and orientation ... 
        RotatedBoundingBox rbbox;
        rbbox.center.x = centroid[0];
        rbbox.center.y = centroid[1];
        rbbox.center.z = (min_z_cluster + max_z_cluster) / 2.0f; // Corrected Z-center: midpoint of actual Z range
        rbbox.size_x = final_width;
        rbbox.size_y = final_height;
        rbbox.size_z = max_z_cluster - min_z_cluster;

        // Filter by a minimum volume or size if needed
        // Use the actual 3D dimensions for filtering
        if (dim[0] > 0.01 && dim[1] > 0.01 && dim[2] > 0.01) {
            rotated_bboxes.push_back(rbbox);
        }
    }
    
    return rotated_bboxes;
}

bool RangeImageObstacleDetector::computeClusterGeom(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster, MinAreaRect& mar, float& min_z_cluster, float& max_z_cluster) {
    // compute cluster geometry in ref frame
    if (cluster->points.empty()) return false;

    // 1. Extract 2D points (XY plane) and find Z-extents
    pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_2d(new pcl::PointCloud<pcl::PointXYZI>);
    min_z_cluster = std::numeric_limits<float>::max();
    max_z_cluster = -std::numeric_limits<float>::max();

    for (const auto& pt : cluster->points) {
        pcl::PointXYZI pt_2d;
        pt_2d.x = pt.x;
        pt_2d.y = pt.y;
        pt_2d.z = 0.0f; // Project to XY plane
        cluster_2d->points.push_back(pt_2d);

        min_z_cluster = std::min(min_z_cluster, pt.z);
        max_z_cluster = std::max(max_z_cluster, pt.z);
    }

    if (cluster_2d->points.size() < 3) { // Need at least 3 points for a convex hull
        return false;
    }

    // 2. Compute 2D Convex Hull
    pcl::PointCloud<pcl::PointXYZI>::Ptr hull_points(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::ConvexHull<pcl::PointXYZI> ch;
    ch.setInputCloud(cluster_2d);
    ch.reconstruct(*hull_points);

    if (hull_points->points.empty()) {
        return false;
    }

    // 3. Apply Rotating Calipers to find Minimum Area Rectangle
    mar = findMinAreaRect(hull_points);

    return true;
}

std::vector<RotatedBoundingBox> RangeImageObstacleDetector::getObstacleBBoxesFromGround(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters) {
    // input clusters are in ground frame, output is in lidar frame
    
    std::vector<RotatedBoundingBox> rotated_bboxes;
    for (const auto& current_cluster : clusters) {
        MinAreaRect mar;
        float min_z_cluster;
        float max_z_cluster;

        if(!computeClusterGeom(current_cluster, mar, min_z_cluster, max_z_cluster)) continue;

        // the only diff from getObstacleBBoxes is center and orientation, mar center is in ground frame, transform to lidar frame
        RotatedBoundingBox rbbox;
        Eigen::Vector3f center_ground(mar.center.x(), mar.center.y(), (min_z_cluster + max_z_cluster) / 2.0f);
        Eigen::Vector3f center_lidar = sensor_inv_transform_ * center_ground;
        rbbox.center.x = center_lidar.x();
        rbbox.center.y = center_lidar.y();
        rbbox.center.z = center_lidar.z();
        rbbox.size_x = mar.size_x;
        rbbox.size_y = mar.size_y;
        rbbox.size_z = max_z_cluster - min_z_cluster;
        Eigen::AngleAxisf rotation_in_ground(mar.angle, Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f rotation_mat = sensor_inv_transform_.rotation() * rotation_in_ground.toRotationMatrix();
        rbbox.orientation = Eigen::Quaternionf(rotation_mat);
        rbbox.is_ground_aligned = true;

        // Filter by a minimum volume or size if needed
        if (rbbox.size_x > 0.01 && rbbox.size_y > 0.01 && rbbox.size_z > 0.01) {
            rotated_bboxes.push_back(rbbox);
        }
    }
    return rotated_bboxes;
}

std::vector<MqttLidarData> RangeImageObstacleDetector::getMqttLidarData(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters) {
    std::vector<MqttLidarData> mqtt_data_vec;
    for (const auto& cluster : clusters) {
        if (cluster->points.empty()) continue;

        float min_x = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_z = -std::numeric_limits<float>::max();
        float min_dist = std::numeric_limits<float>::max();

        for (const auto& pt : cluster->points) {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
            min_z = std::min(min_z, pt.z);
            max_z = std::max(max_z, pt.z);
            
            float d = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
            if (d < min_dist) min_dist = d;
        }

        MqttLidarData data;
        data.width = max_x - min_x;
        data.length = max_y - min_y;
        data.height = max_z - min_z;
        data.center_coordinates.x = (min_x + max_x) / 2.0f;
        data.center_coordinates.y = (min_y + max_y) / 2.0f;
        data.distance = min_dist;
        
        float angle_rad = std::atan2(data.center_coordinates.y, data.center_coordinates.x);
        float angle_deg = angle_rad * 180.0f / M_PI;
        data.angle = angle_deg;

        mqtt_data_vec.push_back(data);
    }
    return mqtt_data_vec;
}

std::vector<RotatedBoundingBox> RangeImageObstacleDetector::getObstacleBBoxes(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters) {
    // input clusters are in lidar frame, output is in lidar frame
    
    std::vector<RotatedBoundingBox> rotated_bboxes;
    for (const auto& current_cluster : clusters) {
        MinAreaRect mar;
        float min_z_cluster;
        float max_z_cluster;

        if(!computeClusterGeom(current_cluster, mar, min_z_cluster, max_z_cluster)) continue;

        // 4. Populate RotatedBoundingBox
        RotatedBoundingBox rbbox;
        rbbox.center.x = mar.center.x();
        rbbox.center.y = mar.center.y();
        rbbox.center.z = (min_z_cluster + max_z_cluster) / 2.0f; // Midpoint of actual Z range
        rbbox.size_x = mar.size_x;
        rbbox.size_y = mar.size_y;
        rbbox.size_z = max_z_cluster - min_z_cluster;
        // Create orientation quaternion from angle (rotation around lidar Z-axis only)
        Eigen::AngleAxisf rotation_z(mar.angle, Eigen::Vector3f::UnitZ());
        rbbox.orientation = Eigen::Quaternionf(rotation_z);
        rbbox.is_ground_aligned = false;

        // Filter by a minimum volume or size if needed
        if (rbbox.size_x > 0.01 && rbbox.size_y > 0.01 && rbbox.size_z > 0.01) {
            rotated_bboxes.push_back(rbbox);
        }
    }
    return rotated_bboxes;
}

// Helper function to find the minimum area enclosing rectangle using rotating calipers
MinAreaRect RangeImageObstacleDetector::findMinAreaRect(const pcl::PointCloud<pcl::PointXYZI>::Ptr& hull_points_2d) {
    MinAreaRect result;
    result.size_x = std::numeric_limits<float>::max();
    result.size_y = std::numeric_limits<float>::max();
    result.angle = 0.0f;
    result.center = Eigen::Vector2f(0.0f, 0.0f);

    if (hull_points_2d->points.size() < 2) {
        return result;
    }

    float min_area = std::numeric_limits<float>::max();

    // Iterate through each edge of the convex hull
    for (size_t i = 0; i < hull_points_2d->points.size(); ++i) {
        const pcl::PointXYZI& p1 = hull_points_2d->points[i];
        const pcl::PointXYZI& p2 = hull_points_2d->points[(i + 1) % hull_points_2d->points.size()];

        // Vector representing the edge
        Eigen::Vector2f edge_vec(p2.x - p1.x, p2.y - p1.y);
        float edge_length = edge_vec.norm();

        if (edge_length < 1e-6) continue; // Skip degenerate edges

        // Normalize the edge vector to get the orientation
        edge_vec.normalize();

        // Perpendicular vector
        Eigen::Vector2f perp_vec(-edge_vec.y(), edge_vec.x());

        // Project all hull points onto the edge vector and its perpendicular
        float min_proj_edge = std::numeric_limits<float>::max();
        float max_proj_edge = -std::numeric_limits<float>::max();
        float min_proj_perp = std::numeric_limits<float>::max();
        float max_proj_perp = -std::numeric_limits<float>::max();

        for (const auto& pt : hull_points_2d->points) {
            Eigen::Vector2f current_pt(pt.x, pt.y);
            float proj_edge = current_pt.dot(edge_vec);
            float proj_perp = current_pt.dot(perp_vec);

            min_proj_edge = std::min(min_proj_edge, proj_edge);
            max_proj_edge = std::max(max_proj_edge, proj_edge);
            min_proj_perp = std::min(min_proj_perp, proj_perp);
            max_proj_perp = std::max(max_proj_perp, proj_perp);
        }

        // width is length along edge_vec direction, height is length along perp_vec direction, angle is the direction of edge_vec
        float current_width = max_proj_edge - min_proj_edge;
        float current_height = max_proj_perp - min_proj_perp;
        float current_area = current_width * current_height;

        if (current_area < min_area) {
            min_area = current_area;
            result.size_x = current_width;
            result.size_y = current_height;
            result.angle = std::atan2(edge_vec.y(), edge_vec.x());

            // Calculate center of the bounding box
            Eigen::Vector2f center_on_edge = (min_proj_edge + max_proj_edge) / 2.0f * edge_vec;
            Eigen::Vector2f center_on_perp = (min_proj_perp + max_proj_perp) / 2.0f * perp_vec;
            result.center = center_on_edge + center_on_perp;
        }
    }
    return result;
}

void RangeImageObstacleDetector::saveRotatedBoundingBoxesToObj(
    const std::vector<RotatedBoundingBox>& rotated_bboxes,
    const std::string& file_path) {
    
    std::ofstream obj_file(file_path);
    if (!obj_file.is_open()) {
        std::cerr << "Error: Could not open OBJ file for writing: " << file_path << std::endl;
        return;
    }

    // Generate MTL file path
    std::string mtl_file_path = file_path;
    size_t dot_pos = mtl_file_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        mtl_file_path = mtl_file_path.substr(0, dot_pos);
    }
    mtl_file_path += ".mtl";

    // Write MTL file
    std::ofstream mtl_file(mtl_file_path);
    if (!mtl_file.is_open()) {
        std::cerr << "Error: Could not open MTL file for writing: " << mtl_file_path << std::endl;
        obj_file.close();
        return;
    }

    // Define a transparent green material
    mtl_file << "newmtl transparent_green" << std::endl;
    mtl_file << "Kd 0.0 1.0 0.0" << std::endl; // Diffuse color (green)
    mtl_file << "d 0.3" << std::endl;         // alpha value (transparency)
    mtl_file.close();

    // Link MTL file in OBJ
    obj_file << "mtllib " << mtl_file_path.substr(mtl_file_path.find_last_of('/') + 1) << std::endl;

    int vertex_offset = 0; // To keep track of vertex indices for multiple boxes

    for (size_t i = 0; i < rotated_bboxes.size(); ++i) {
        const auto& rbbox = rotated_bboxes[i];

        // Calculate half dimensions
        float half_size_x = rbbox.size_x / 2.0f;
        float half_size_y = rbbox.size_y / 2.0f;
        float half_size_z = rbbox.size_z / 2.0f;

        // Base points in the local coordinate system of the bbox (before rotation and translation)
        std::vector<Eigen::Vector3f> local_corners = {
            {-half_size_x, -half_size_y, -half_size_z}, // 0
            { half_size_x, -half_size_y, -half_size_z}, // 1
            { half_size_x,  half_size_y, -half_size_z}, // 2
            {-half_size_x,  half_size_y, -half_size_z}, // 3
            {-half_size_x, -half_size_y,  half_size_z}, // 4
            { half_size_x, -half_size_y,  half_size_z}, // 5
            { half_size_x,  half_size_y,  half_size_z}, // 6
            {-half_size_x,  half_size_y,  half_size_z}  // 7
        };

        Eigen::Matrix3f rotation_matrix = rbbox.orientation.toRotationMatrix();

        // Translate and rotate each local corner to global coordinates
        Eigen::Vector3f center_eigen(rbbox.center.x, rbbox.center.y, rbbox.center.z);

        obj_file << "o BoundingBox_" << i << std::endl; // Object name
        obj_file << "usemtl transparent_green" << std::endl; // Assign material

        // Write vertices
        for (const auto& lc : local_corners) {
            Eigen::Vector3f rotated_corner = rotation_matrix * lc;
            Eigen::Vector3f global_corner = center_eigen + rotated_corner;
            obj_file << "v " << global_corner.x() << " " << global_corner.y() << " " << global_corner.z() << std::endl;
        }

        // Define faces (1-based indexing relative to the current object's vertices)
        // Bottom face
        obj_file << "f " << (vertex_offset + 1) << " " << (vertex_offset + 2) << " " << (vertex_offset + 3) << " " << (vertex_offset + 4) << std::endl;
        // Top face
        obj_file << "f " << (vertex_offset + 5) << " " << (vertex_offset + 6) << " " << (vertex_offset + 7) << " " << (vertex_offset + 8) << std::endl;
        // Front face
        obj_file << "f " << (vertex_offset + 1) << " " << (vertex_offset + 2) << " " << (vertex_offset + 6) << " " << (vertex_offset + 5) << std::endl;
        // Back face
        obj_file << "f " << (vertex_offset + 4) << " " << (vertex_offset + 3) << " " << (vertex_offset + 7) << " " << (vertex_offset + 8) << std::endl;
        // Left face
        obj_file << "f " << (vertex_offset + 1) << " " << (vertex_offset + 4) << " " << (vertex_offset + 8) << " " << (vertex_offset + 5) << std::endl;
        // Right face
        obj_file << "f " << (vertex_offset + 2) << " " << (vertex_offset + 3) << " " << (vertex_offset + 7) << " " << (vertex_offset + 6) << std::endl;

        vertex_offset += 8; // Increment offset for the next bounding box
    }

    obj_file.close();
    std::cout << "Saved " << rotated_bboxes.size() << " bounding boxes to " << file_path << std::endl;
}

void RangeImageObstacleDetector::saveObstacleBBoxes2DToJson(
    const std::vector<ObstacleBBox2D>& bboxes_2d,
    const std::string& file_path) {
    nlohmann::json j = bboxes_2d;
    std::ofstream json_file(file_path);
    if (json_file.is_open()) {
        json_file << j.dump(4);
        json_file.close();
        std::cout << "Saved " << bboxes_2d.size() << " 2D bounding boxes to " << file_path << std::endl;
    } else {
        std::cerr << "Error: Could not open JSON file for writing: " << file_path << std::endl;
    }
}
