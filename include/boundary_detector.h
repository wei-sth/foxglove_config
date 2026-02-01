#ifndef BOUNDARY_DETECTOR_H
#define BOUNDARY_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>

struct HSVContext {
    cv::Mat hsv_img;
    int h_min = 0, s_min = 0, v_min = 0;
    int h_max = 179, s_max = 255, v_max = 255;
};

bool detectBoundary(const std::string& image_path, const std::string& output_path);
bool detectBoundary_v0(const std::string& image_path, const std::string& output_path);
bool detectBoundary_v1(const std::string& image_path, const std::string& output_path);

void on_trackbar(int, void* userdata);
void tuneHSVThreshold(const std::string& image_path, double scale = 0.0);

#endif // BOUNDARY_DETECTOR_H
