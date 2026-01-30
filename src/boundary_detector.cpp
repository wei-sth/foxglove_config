#include "boundary_detector.h"
#include <iostream>

bool detectBoundary(const std::string& image_path, const std::string& output_path) {
    // 1. Image loading and preprocessing
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return false;
    }

    // Rotate the image to correct its orientation
    cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // 2. Color space conversion and thresholding
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // Adjust HSV ranges for grass (green/brown) and concrete (gray)
    // These values may need fine-tuning based on the actual image
    // Assuming grass is brownish-green, concrete is gray
    // Brown/Green range (H: 20-80, S: 30-255, V: 30-255)
    // Gray range (H: 0-180, S: 0-30, V: 50-200)

    // Attempt to segment grass
    cv::Mat mask_grass;
    cv::Scalar lower_grass(20, 30, 30);
    cv::Scalar upper_grass(80, 255, 255);
    cv::inRange(hsv, lower_grass, upper_grass, mask_grass);

    // Attempt to segment concrete
    cv::Mat mask_concrete;
    cv::Scalar lower_concrete(0, 0, 50);
    cv::Scalar upper_concrete(180, 30, 200);
    cv::inRange(hsv, lower_concrete, upper_concrete, mask_concrete);

    // 3. Morphological operations (optional, if direct Canny is not effective)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask_grass, mask_grass, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask_grass, mask_grass, cv::MORPH_CLOSE, kernel);

    cv::morphologyEx(mask_concrete, mask_concrete, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask_concrete, mask_concrete, cv::MORPH_CLOSE, kernel);

    // Combine masks to get the region of interest
    cv::Mat combined_mask;
    cv::bitwise_or(mask_grass, mask_concrete, combined_mask);
    
    // 4. Edge detection
    // Use Canny on the blurred grayscale image
    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);

    // Alternatively, use Canny on the combined_mask
    // cv::Canny(combined_mask, edges, 50, 150);

    // 5. Contour finding and drawing
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw contours on the original image
    cv::Mat output_img = img.clone();
    cv::drawContours(output_img, contours, -1, cv::Scalar(0, 255, 0), 2); // Green contours

    // 6. Save the result
    if (!cv::imwrite(output_path, output_img)) {
        std::cerr << "Error: Could not save output image to " << output_path << std::endl;
        return false;
    }
    std::cout << "Boundary detection complete. Output saved to " << output_path << std::endl;
    return true;
}
