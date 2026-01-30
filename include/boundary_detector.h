#ifndef BOUNDARY_DETECTOR_H
#define BOUNDARY_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>

/**
 * @brief Detects boundaries between grass and concrete in an image using traditional OpenCV strategies.
 * @param image_path Path to the input image.
 * @param output_path Path to save the output image with detected boundaries.
 * @return True if boundary detection is successful, false otherwise.
 */
bool detectBoundary(const std::string& image_path, const std::string& output_path);

#endif // BOUNDARY_DETECTOR_H
