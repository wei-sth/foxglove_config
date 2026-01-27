#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

struct VIOMetrics {
    double gradient_pixel_ratio;
    double avg_isotropy;
    double photometric_contrast;
    double spatial_uniformity;
    double fast_feature_density;
};

VIOMetrics computeMetrics(const cv::Mat& img) {
    VIOMetrics metrics;
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    // 1. Gradient Pixel Ratio
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);

    double grad_thresh = 20.0;
    int high_grad_count = cv::countNonZero(grad_mag > grad_thresh);
    metrics.gradient_pixel_ratio = static_cast<double>(high_grad_count) / (gray.rows * gray.cols);

    // 2. Structure Tensor & Isotropy
    cv::Mat ixx, iyy, ixy;
    cv::multiply(grad_x, grad_x, ixx);
    cv::multiply(grad_y, grad_y, iyy);
    cv::multiply(grad_x, grad_y, ixy);

    // Box filter to simulate local integration
    int block_size = 5;
    cv::boxFilter(ixx, ixx, CV_32F, cv::Size(block_size, block_size));
    cv::boxFilter(iyy, iyy, CV_32F, cv::Size(block_size, block_size));
    cv::boxFilter(ixy, ixy, CV_32F, cv::Size(block_size, block_size));

    double total_isotropy = 0;
    int valid_pixels = 0;
    for (int r = 0; r < gray.rows; ++r) {
        for (int c = 0; c < gray.cols; ++c) {
            float a = ixx.at<float>(r, c);
            float b = ixy.at<float>(r, c);
            float d = iyy.at<float>(r, c);

            float trace = a + d;
            float det = a * d - b * b;
            float sqrt_val = std::sqrt(std::max(0.0f, trace * trace / 4.0f - det));
            float l1 = trace / 2.0f + sqrt_val;
            float l2 = trace / 2.0f - sqrt_val;

            if (l1 > 1e-4) {
                total_isotropy += (l2 / l1);
                valid_pixels++;
            }
        }
    }
    metrics.avg_isotropy = (valid_pixels > 0) ? (total_isotropy / valid_pixels) : 0.0;

    // 3. Photometric Contrast (Std Dev)
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    metrics.photometric_contrast = stddev[0];

    // 4. Spatial Distribution Uniformity
    int grid_rows = 8;
    int grid_cols = 8;
    std::vector<int> grid_counts(grid_rows * grid_cols, 0);
    int cell_h = gray.rows / grid_rows;
    int cell_w = gray.cols / grid_cols;

    for (int r = 0; r < gray.rows; ++r) {
        for (int c = 0; c < gray.cols; ++c) {
            if (grad_mag.at<float>(r, c) > grad_thresh) {
                int gr = std::min(r / cell_h, grid_rows - 1);
                int gc = std::min(c / cell_w, grid_cols - 1);
                grid_counts[gr * grid_cols + gc]++;
            }
        }
    }

    double grid_mean = static_cast<double>(high_grad_count) / (grid_rows * grid_cols);
    double grid_var = 0;
    for (int count : grid_counts) {
        grid_var += std::pow(count - grid_mean, 2);
    }
    grid_var /= (grid_rows * grid_cols);
    // Uniformity = 1 / (1 + CV), where CV is coefficient of variation
    double cv = (grid_mean > 0) ? (std::sqrt(grid_var) / grid_mean) : 10.0;
    metrics.spatial_uniformity = 1.0 / (1.0 + cv);

    // 5. FAST Feature Density
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(gray, keypoints, 20);
    metrics.fast_feature_density = static_cast<double>(keypoints.size()) / (gray.rows * gray.cols) * 1000.0; // per 1000px

    return metrics;
}

int main(int argc, char** argv) {
    std::vector<std::string> img_fn_vec = {"20260127165512.jpg","20260127200629.jpg","20260127200640.jpg",
        "20260127200700.jpg","20260127200727.jpg","20260127200617.jpg",
        "20260127200635.jpg","20260127200646.jpg","20260127200719.jpg"};
    std::string base_path = "/home/weizh/data/";

    for (const auto& name : img_fn_vec) {
        std::string img_path = base_path + name;
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {
            std::cerr << "Could not open or find the image: " << img_path << std::endl;
            continue;
        }
        
        std::cout << "\n process image: " << name << std::endl;

        VIOMetrics m = computeMetrics(img);
        std::cout << "--- SVO/SLAM Suitability Analysis ---" << std::endl;
        std::cout << "Image Size: " << img.cols << "x" << img.rows << std::endl;
        std::cout << "1. Gradient Pixel Ratio (>20): " << m.gradient_pixel_ratio * 100.0 << "%" << std::endl;
        std::cout << "   (Target: >10% for robust direct alignment)" << std::endl;
        
        std::cout << "2. Avg Isotropy (Structure Tensor): " << m.avg_isotropy << std::endl;
        std::cout << "   (0: edge-like, 1: corner-like. Higher is better for localization)" << std::endl;

        std::cout << "3. Photometric Contrast (StdDev): " << m.photometric_contrast << std::endl;
        std::cout << "   (Target: >20 for good signal-to-noise ratio)" << std::endl;

        std::cout << "4. Spatial Uniformity (Grid-based): " << m.spatial_uniformity << std::endl;
        std::cout << "   (Higher means features are well-distributed)" << std::endl;

        std::cout << "5. FAST Feature Density: " << m.fast_feature_density << " pts/1000px" << std::endl;
        std::cout << "   (Target: >0.5 for reliable keyframe tracking)" << std::endl;
        
        bool suitable = (m.gradient_pixel_ratio > 0.1) && (m.photometric_contrast > 15.0) && (m.fast_feature_density > 0.2);
        if (suitable) {
            std::cout << "Result: SUITABLE for Semi-Direct SLAM (SVO)." << std::endl;
        } else {
            std::cout << "Result: POOR suitability. May suffer from tracking loss." << std::endl;
        }
    }
    return 0;
}
