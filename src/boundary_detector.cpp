#include "boundary_detector.h"
#include <iostream>
#include <filesystem> // For path manipulation

bool detectBoundary(const std::string& image_path, const std::string& output_path) {
    // Extract root and extension from image_path
    std::filesystem::path img_path_obj(image_path);
    std::string root = img_path_obj.stem().string();
    std::string ext = img_path_obj.extension().string();
    std::string parent_path = img_path_obj.parent_path().string();

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return false;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite(parent_path + "/" + root + "_gray" + ext, gray);

    cv::Mat hsv;  // Hue, Saturation, Value
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);  // saving hsv directly to image is meaningless for most image viewer. Two pixels seem the same does not mean we cannot split them by true h,s,v value.
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels); // hsv_channels[0] = H, hsv_channels[1] = S, hsv_channels[2] = V
    cv::imwrite(parent_path + "/" + root + "_h" + ext, hsv_channels[0]);
    cv::imwrite(parent_path + "/" + root + "_s" + ext, hsv_channels[1]);
    cv::imwrite(parent_path + "/" + root + "_v" + ext, hsv_channels[2]);

    // Calculate gradients for texture analysis
    cv::Mat gray_for_gradients;
    cv::cvtColor(img, gray_for_gradients, cv::COLOR_BGR2GRAY);

    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    // Gradient X
    cv::Sobel(gray_for_gradients, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);

    // Gradient Y
    cv::Sobel(gray_for_gradients, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // Convert gradients to floating-point type for cv::magnitude
    cv::Mat grad_x_float, grad_y_float;
    grad_x.convertTo(grad_x_float, CV_32F);
    grad_y.convertTo(grad_y_float, CV_32F);

    // Total Gradient (approximate magnitude)
    cv::Mat grad_magnitude;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_magnitude);
    cv::imwrite(parent_path + "/" + root + "_grad_magnitude" + ext, grad_magnitude); // Save gradient magnitude image

    // Total Gradient (more accurate magnitude using sqrt)
    cv::Mat grad_magnitude_float;
    cv::magnitude(grad_x_float, grad_y_float, grad_magnitude_float);

    // Convert to 8-bit for saving/display
    cv::Mat grad_magnitude_8u;
    // Scale the float values to 0-255 range for 8-bit image
    cv::normalize(grad_magnitude_float, grad_magnitude_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    // grad_magnitude_float.convertTo(grad_magnitude_8u, CV_8U);
    cv::imwrite(parent_path + "/" + root + "_grad_magnitude_sqrt" + ext, grad_magnitude_8u); // Save gradient magnitude image
    
    // -------------------------------------------------------------------------------
    // Apply Gabor filters for texture analysis, does not find anything, not very time costly
    // Parameters for Gabor filters
    // int kernel_size = 31; // Must be odd
    // double sigma = 5.0;
    // double theta_angles[] = {0, CV_PI / 4, CV_PI / 2, 3 * CV_PI / 4}; // 0, 45, 90, 135 degrees
    // double lambda = 10.0; // Wavelength
    // double gamma = 0.5; // Spatial aspect ratio
    // double psi = 0; // Phase offset

    // for (int i = 0; i < sizeof(theta_angles) / sizeof(theta_angles[0]); ++i) {
    //     double theta = theta_angles[i];
    //     cv::Mat gabor_kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambda, gamma, psi, CV_32F);
        
    //     cv::Mat filtered_image;
    //     cv::filter2D(gray, filtered_image, CV_32F, gabor_kernel);

    //     cv::Mat filtered_image_8u;
    //     cv::normalize(filtered_image, filtered_image_8u, 0, 255, cv::NORM_MINMAX, CV_8U);

    //     std::string gabor_output_path = parent_path + "/" + root + "_gabor_theta_" + std::to_string(static_cast<int>(theta * 180 / CV_PI)) + ext;
    //     cv::imwrite(gabor_output_path, filtered_image_8u);
    //     std::cout << "Saved Gabor filtered image to " << gabor_output_path << std::endl;
    // }

    // -------------------------------------------------------------------------------

    // PCA on gradients in regions, does not find anything, time costly
    // int window_size = 25; // Example window size
    // cv::Mat L1_map = cv::Mat::zeros(img.size(), CV_32F);
    // cv::Mat L2_map = cv::Mat::zeros(img.size(), CV_32F);
    // cv::Mat L1_div_L2_map = cv::Mat::zeros(img.size(), CV_32F);

    // for (int r = window_size / 2; r < img.rows - window_size / 2; ++r) {
    //     for (int c = window_size / 2; c < img.cols - window_size / 2; ++c) {
    //         cv::Rect window_roi(c - window_size / 2, r - window_size / 2, window_size, window_size);

    //         cv::Mat window_grad_x = grad_x_float(window_roi);
    //         cv::Mat window_grad_y = grad_y_float(window_roi);

    //         // Prepare data for PCA
    //         cv::Mat data(window_size * window_size, 2, CV_32F);
    //         for (int i = 0; i < window_size; ++i) {
    //             for (int j = 0; j < window_size; ++j) {
    //                 data.at<float>(i * window_size + j, 0) = window_grad_x.at<float>(i, j);
    //                 data.at<float>(i * window_size + j, 1) = window_grad_y.at<float>(i, j);
    //             }
    //         }

    //         cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

    //         // L1 and L2 are the eigenvalues
    //         float L1 = pca.eigenvalues.at<float>(0);
    //         float L2 = pca.eigenvalues.at<float>(1);

    //         L1_map.at<float>(r, c) = L1;
    //         L2_map.at<float>(r, c) = L2;
    //         if (L2 != 0) {
    //             L1_div_L2_map.at<float>(r, c) = L1 / L2;
    //         } else {
    //             L1_div_L2_map.at<float>(r, c) = 0; // Handle division by zero
    //         }
    //     }
    // }

    // // Visualize and save L1, L2, L1/L2 maps
    // cv::Mat L1_8u, L2_8u, L1_div_L2_8u;

    // cv::normalize(L1_map, L1_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::imwrite(parent_path + "/" + root + "_L1" + ext, L1_8u);

    // cv::normalize(L2_map, L2_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::imwrite(parent_path + "/" + root + "_L2" + ext, L2_8u);

    // cv::normalize(L1_div_L2_map, L1_div_L2_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::imwrite(parent_path + "/" + root + "_L1_div_L2" + ext, L1_div_L2_8u);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::imwrite(parent_path + "/" + root + "_blurred" + ext, blurred); // Save blurred image

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
    cv::imwrite(parent_path + "/" + root + "_mask_grass" + ext, mask_grass); // Save grass mask, white is grass

    // Attempt to segment concrete
    cv::Mat mask_concrete;
    cv::Scalar lower_concrete(0, 0, 50);
    cv::Scalar upper_concrete(180, 30, 200);
    cv::inRange(hsv, lower_concrete, upper_concrete, mask_concrete);
    cv::imwrite(parent_path + "/" + root + "_mask_concrete" + ext, mask_concrete); // Save concrete mask, white is concrete

    // 3. Morphological operations (optional, if direct Canny is not effective)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask_grass, mask_grass, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask_grass, mask_grass, cv::MORPH_CLOSE, kernel);

    cv::morphologyEx(mask_concrete, mask_concrete, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask_concrete, mask_concrete, cv::MORPH_CLOSE, kernel);

    // Combine masks to get the region of interest
    cv::Mat combined_mask;
    cv::bitwise_or(mask_grass, mask_concrete, combined_mask);
    cv::imwrite(parent_path + "/" + root + "_combined_mask" + ext, combined_mask); // Save combined mask
    
    // 4. Edge detection
    // Use Canny on the blurred grayscale image
    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);
    cv::imwrite(parent_path + "/" + root + "_edges" + ext, edges); // Save edges image

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

bool detectBoundary_v0(const std::string& image_path, const std::string& output_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return false;
    }

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

bool detectBoundary_v1(const std::string& image_path, const std::string& output_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return false;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // 2. Color space conversion and adaptive thresholding for grass and concrete
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask_grass, mask_concrete;

    // Grass mask (broader range for green/yellow/brownish tones)
    // H: 20-90 (yellow-green to green), S: 30-255, V: 30-255
    // This range should cover most grass colors in different seasons.
    cv::Scalar lower_grass(20, 30, 30);
    cv::Scalar upper_grass(90, 255, 255);
    cv::inRange(hsv, lower_grass, upper_grass, mask_grass);

    // Concrete mask (low saturation for gray tones)
    // H: 0-180, S: 0-60, V: 80-255
    cv::Scalar lower_concrete(0, 0, 80);
    cv::Scalar upper_concrete(180, 60, 255);
    cv::inRange(hsv, lower_concrete, upper_concrete, mask_concrete);

    // 3. Morphological operations to clean up the masks and merge small grass regions
    cv::Mat kernel_small = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat kernel_large = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)); // Larger kernel for merging grass

    // Clean concrete mask
    cv::morphologyEx(mask_concrete, mask_concrete, cv::MORPH_OPEN, kernel_small);
    cv::morphologyEx(mask_concrete, mask_concrete, cv::MORPH_CLOSE, kernel_small);

    // Clean and merge grass mask
    cv::morphologyEx(mask_grass, mask_grass, cv::MORPH_OPEN, kernel_small); // Remove small noise
    cv::morphologyEx(mask_grass, mask_grass, cv::MORPH_CLOSE, kernel_large); // Merge smaller grass regions

    // Create a combined mask for better edge detection between regions
    cv::Mat combined_regions = cv::Mat::zeros(img.size(), CV_8UC1);
    combined_regions.setTo(100, mask_grass); // Grass region
    combined_regions.setTo(200, mask_concrete); // Concrete region

    // 4. Edge detection using Canny on the combined regions or blurred grayscale
    // Using Canny on the combined_regions can highlight the transition between grass and concrete.
    cv::Mat edges;
    cv::Canny(combined_regions, edges, 50, 150); // Adjust thresholds as needed

    // 5. Contour finding and filtering
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat output_img = img.clone();
    std::vector<std::vector<cv::Point>> final_boundaries;

    for (const auto& contour : contours) {
        double arc_length = cv::arcLength(contour, true);
        if (arc_length < 150) continue; // Filter out short contours, adjust threshold

        // Approximate contour to a polygon to smooth it, but keep it "regular"
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, arc_length * 0.005, true); // Smaller epsilon for less aggressive approximation

        if (approx.size() < 4) continue; // Filter out contours that are too simple (e.g., triangles)

        // Region validation: Check if the contour separates grass and concrete regions
        // Sample points along the contour and check their neighborhood in the original masks.
        bool is_boundary = false;
        int grass_count = 0;
        int concrete_count = 0;
        int sample_step = std::max(1, (int)approx.size() / 10); // Sample 10 points along the contour

        for (size_t k = 0; k < approx.size(); k += sample_step) {
            cv::Point p = approx[k];
            if (p.x < 0 || p.x >= img.cols || p.y < 0 || p.y >= img.rows) continue;

            // Check a small neighborhood around the point
            int neighborhood_size = 5;
            cv::Rect roi(std::max(0, p.x - neighborhood_size), std::max(0, p.y - neighborhood_size),
                         std::min(img.cols - (p.x - neighborhood_size), 2 * neighborhood_size + 1),
                         std::min(img.rows - (p.y - neighborhood_size), 2 * neighborhood_size + 1));

            cv::Mat grass_roi = mask_grass(roi);
            cv::Mat concrete_roi = mask_concrete(roi);

            if (cv::countNonZero(grass_roi) > 0) grass_count++;
            if (cv::countNonZero(concrete_roi) > 0) concrete_count++;
        }

        // If a significant portion of the contour is near both grass and concrete, it's a boundary
        if (grass_count > 0 && concrete_count > 0 && (grass_count + concrete_count) > (approx.size() / sample_step) / 2) {
            is_boundary = true;
        }

        if (is_boundary) {
            final_boundaries.push_back(contour);
        }
    }
    
    // Draw the final boundaries in green (on top of the overlay)
    cv::drawContours(output_img, final_boundaries, -1, cv::Scalar(0, 255, 0), 2);

    // 6. Save the result
    if (!cv::imwrite(output_path, output_img)) {
        std::cerr << "Error: Could not save output image to " << output_path << std::endl;
        return false;
    }
    std::cout << "Boundary detection complete. Output saved to " << output_path << std::endl;
    return true;
}
