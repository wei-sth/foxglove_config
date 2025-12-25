#include <pcl/io/pcd_io.h>
#include "/home/weizh/foxglove_ws/src/foxglove_config/include/obstacle_detector.h"

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

// ring index = row index
// note xt16, ring 15 is the lowest, ring 0 is the highest, ring 15 and ring 14 angle difference is 2 degree
// due to beam divergence, the further from lidar (assume r is distance), the larger vertical distance between rings, 
// delta_z = r * tan(2 degree) = r * 0.0349
// so if r = 2.0, even for a vertical wall, the vertical distance would be less than 0.07, 
// but if r=20.0, for a slope, the vertical distance between rings might easily exceed 0.2m
RangeImageObstacleDetector::RangeImageObstacleDetector(int num_rings, int num_sectors, 
                               float max_distance, float min_cluster_z_difference)
    : num_rings(num_rings), num_sectors(num_sectors), max_distance(max_distance),
      min_cluster_z_difference_(min_cluster_z_difference),
      obstacle_grid_flat_(num_rings * num_sectors, nullptr), // Initialize flattened grid
      temp_valid_mask_(num_rings, num_sectors, CV_8UC1, cv::Scalar(0)), // Initialize temp valid mask
      visited_mask_(num_rings, num_sectors, CV_8UC1, cv::Scalar(0)) // Initialize visited mask
{
    
    range_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, 
                           cv::Scalar(std::numeric_limits<float>::max()));
    x_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    y_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    z_image_ = cv::Mat(num_rings, num_sectors, CV_32FC1, cv::Scalar(0));
    
    valid_mask_ = cv::Mat(num_rings, num_sectors, CV_8UC1, cv::Scalar(0));
}

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> RangeImageObstacleDetector::detectObstacles(
    pcl::PointCloud<PointXYZIRT>::Ptr cloud_raw) {
    
    // Step 1: Filter by distance and convert to PointXYZINormal
    auto cloud_filtered_normal = filterByDistance(cloud_raw);
    
    // Step 2: Build Range Image using PointXYZINormal
    buildRangeImage(cloud_filtered_normal);
    
    // Step 3: Segment ground by normal and get obstacles with ring info in normal_x
    auto obstacles_with_normal_info = segmentGroundByNormal();
    
    // debug
    // pcl::io::savePCDFileBinary("/home/weizh/data/obstacles_before_cluster.pcd", *obstacles_with_normal_info);

    // Step 4: Perform clustering using Euclidean method
    // return clusterEuclidean(obstacles_with_normal_info);
    return clusterConnectivity(obstacles_with_normal_info);
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
                            
                            q.push({next_r, next_c});
                            visited_mask_.at<uint8_t>(next_r, next_c) = 1;
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

// pcl::PointCloud<pcl::PointXYZINormal>::Ptr RangeImageObstacleDetector::segmentGroundByNormal() {
//     // use row + 1 (larger ring is lower, from ground to top, in case ring 6 at one obstacle and ring 7 at another obstacle, miss the highest obstacle ring) 
//     // by default, unless last ring
    
//     pcl::PointCloud<pcl::PointXYZINormal>::Ptr obstacles_with_normal_info(
//         new pcl::PointCloud<pcl::PointXYZINormal>);
    
//     for (int row = 0; row < num_rings; ++row) {
//         for (int col = 0; col < num_sectors; ++col) {
            
//             if (!valid_mask_.at<uint8_t>(row, col)) continue;

//             float x = x_image_.at<float>(row, col);
//             float y = y_image_.at<float>(row, col);
//             float z = z_image_.at<float>(row, col);
            
//             bool z_trend_condition = false;
//             bool neighbors_valid = false;
//             float slope_threshold = 1.0f; // 45 degree

//             if (row == num_rings - 1) { // Special case for the last ring (lowest)
//                 if (valid_mask_.at<uint8_t>(row - 1, col)) {
//                     neighbors_valid = true;
//                     float z_current = z_image_.at<float>(row, col);
//                     float x_prev = x_image_.at<float>(row - 1, col);
//                     float y_prev = y_image_.at<float>(row - 1, col);
//                     float z_prev = z_image_.at<float>(row - 1, col);

//                     float dz_prev = z_prev - z_current; // Previous is higher than current
//                     float dx_prev = x_prev - x;
//                     float dy_prev = y_prev - y;
//                     float d_xy_prev = std::sqrt(dx_prev*dx_prev + dy_prev*dy_prev);
//                     z_trend_condition = (dz_prev > (d_xy_prev * slope_threshold));
//                 }
//             }
//             else {
//                 if (valid_mask_.at<uint8_t>(row + 1, col)) {
//                     neighbors_valid = true;
//                     float z_current = z_image_.at<float>(row, col);
//                     float x_next = x_image_.at<float>(row + 1, col);
//                     float y_next = y_image_.at<float>(row + 1, col);
//                     float z_next = z_image_.at<float>(row + 1, col);

//                     float dz_next = z_current - z_next; // Current is higher than next
//                     float dx_next = x - x_next;
//                     float dy_next = y - y_next;
//                     float d_xy_next = std::sqrt(dx_next*dx_next + dy_next*dy_next);
//                     z_trend_condition = (dz_next > (d_xy_next * slope_threshold));
//                 }
//             }

//             if (!neighbors_valid) {
//                 continue; // Skip if required surrounding points are not valid
//             }
                
//             Eigen::Vector3f normal = computeNormal(row, col);
            
//             float normal_z_threshold = 0.7f;
//             if (std::abs(normal.z()) < normal_z_threshold && z_trend_condition) {
//                 pcl::PointXYZINormal obstacle_pt;
//                 obstacle_pt.x = x;
//                 obstacle_pt.y = y;
//                 obstacle_pt.z = z;
//                 obstacle_pt.normal_x = static_cast<float>(row); // Store ring in normal_x
//                 // Other normal_y, normal_z, intensity, curvature can be set as needed
//                 obstacles_with_normal_info->points.push_back(obstacle_pt);
//             }
//         }
//     }
    
//     return obstacles_with_normal_info;
// }

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
                // Other normal_y, normal_z, intensity, curvature can be set as needed
                obstacles_with_normal_info->points.push_back(obstacle_pt);
            }
        }
    }
    
    return obstacles_with_normal_info;
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

std::vector<RotatedBoundingBox> RangeImageObstacleDetector::getObstacleBoundingBoxesNew(
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

        RotatedBoundingBox rbbox;
        rbbox.center.x = centroid[0];
        rbbox.center.y = centroid[1];
        rbbox.center.z = (min_z_cluster + max_z_cluster) / 2.0f; // Corrected Z-center: midpoint of actual Z range
        rbbox.width = final_width;
        rbbox.height = final_height;
        rbbox.angle = angle;
        rbbox.min_z_point.z = min_z_cluster;
        rbbox.max_z_point.z = max_z_cluster;

        // Filter by a minimum volume or size if needed
        // Use the actual 3D dimensions for filtering
        if (dim[0] > 0.01 && dim[1] > 0.01 && dim[2] > 0.01) {
            rotated_bboxes.push_back(rbbox);
        }
    }
    
    return rotated_bboxes;
}

std::vector<RotatedBoundingBox> RangeImageObstacleDetector::getObstacleBoundingBoxesNewV2(
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters) {
    
    std::vector<RotatedBoundingBox> rotated_bboxes;
    for (const auto& current_cluster : clusters) {
        if (current_cluster->points.empty()) continue;

        // 1. Extract 2D points (XY plane) and find Z-extents
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_2d(new pcl::PointCloud<pcl::PointXYZI>);
        float min_z_cluster = std::numeric_limits<float>::max();
        float max_z_cluster = -std::numeric_limits<float>::max();

        for (const auto& pt : current_cluster->points) {
            pcl::PointXYZI pt_2d;
            pt_2d.x = pt.x;
            pt_2d.y = pt.y;
            pt_2d.z = 0.0f; // Project to XY plane
            cluster_2d->points.push_back(pt_2d);

            min_z_cluster = std::min(min_z_cluster, pt.z);
            max_z_cluster = std::max(max_z_cluster, pt.z);
        }

        if (cluster_2d->points.size() < 3) { // Need at least 3 points for a convex hull
            continue;
        }

        // 2. Compute 2D Convex Hull
        pcl::PointCloud<pcl::PointXYZI>::Ptr hull_points(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::ConvexHull<pcl::PointXYZI> ch;
        ch.setInputCloud(cluster_2d);
        ch.reconstruct(*hull_points);

        if (hull_points->points.empty()) {
            continue;
        }

        // 3. Apply Rotating Calipers to find Minimum Area Rectangle
        MinAreaRect mar = findMinAreaRect(hull_points);

        // 4. Populate RotatedBoundingBox
        RotatedBoundingBox rbbox;
        rbbox.center.x = mar.center.x();
        rbbox.center.y = mar.center.y();
        rbbox.center.z = (min_z_cluster + max_z_cluster) / 2.0f; // Midpoint of actual Z range
        rbbox.width = mar.width;
        rbbox.height = mar.height;
        rbbox.angle = mar.angle;
        rbbox.min_z_point.z = min_z_cluster;
        rbbox.max_z_point.z = max_z_cluster;

        // Filter by a minimum volume or size if needed
        float depth = max_z_cluster - min_z_cluster;
        if (rbbox.width > 0.01 && rbbox.height > 0.01 && depth > 0.01) {
            rotated_bboxes.push_back(rbbox);
        }
    }
    return rotated_bboxes;
}

// Helper function to find the minimum area enclosing rectangle using rotating calipers
MinAreaRect RangeImageObstacleDetector::findMinAreaRect(const pcl::PointCloud<pcl::PointXYZI>::Ptr& hull_points_2d) {
    MinAreaRect result;
    result.width = std::numeric_limits<float>::max();
    result.height = std::numeric_limits<float>::max();
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

        float current_width = max_proj_edge - min_proj_edge;
        float current_height = max_proj_perp - min_proj_perp;
        float current_area = current_width * current_height;

        if (current_area < min_area) {
            min_area = current_area;
            result.width = current_width;
            result.height = current_height;
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
        float half_width = rbbox.width / 2.0f;
        float half_height = rbbox.height / 2.0f;
        float half_depth = (rbbox.max_z_point.z - rbbox.min_z_point.z) / 2.0f;

        // Base points in the local coordinate system of the bbox (before rotation and translation)
        std::vector<Eigen::Vector3f> local_corners = {
            {-half_width, -half_height, -half_depth}, // 0
            { half_width, -half_height, -half_depth}, // 1
            { half_width,  half_height, -half_depth}, // 2
            {-half_width,  half_height, -half_depth}, // 3
            {-half_width, -half_height,  half_depth}, // 4
            { half_width, -half_height,  half_depth}, // 5
            { half_width,  half_height,  half_depth}, // 6
            {-half_width,  half_height,  half_depth}  // 7
        };

        // Create rotation matrix around Z-axis
        Eigen::AngleAxisf rotation_z(rbbox.angle, Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f rotation_matrix = rotation_z.toRotationMatrix();

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
