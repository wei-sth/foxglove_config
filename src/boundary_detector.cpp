#include "boundary_detector.h"
#include <iostream>
#include <filesystem> // For path manipulation

// try to use use Canny on mask, or use canny on gray, gray_blurred, h_channel etc.
// use cv::bitwise_or to combine mask
// use cv::GaussianBlur

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

    // HSV space works well for this scene, whereas Lab space shows no clear patterns. Comment out in case other scenes might need Lab.
    // cv::Mat lab;
    // cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);  // saving lab directly to image is meaningless for most image viewer. Two pixels seem the same does not mean we cannot split them by true l,a,b value.
    // std::vector<cv::Mat> lab_channels;
    // cv::split(lab, lab_channels); // lab_channels[0] = L, lab_channels[1] = a, lab_channels[2] = b
    // cv::imwrite(parent_path + "/" + root + "_l" + ext, lab_channels[0]);
    // cv::imwrite(parent_path + "/" + root + "_a" + ext, lab_channels[1]);
    // cv::imwrite(parent_path + "/" + root + "_b" + ext, lab_channels[2]);

    // Calculate gradients for texture analysis, FILTER_SCHARR is more sensitive
    // cv::Mat grad_x, grad_y;
    // cv::Mat abs_grad_x, abs_grad_y;
    // // cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    // cv::Sobel(gray, grad_x, CV_16S, 1, 0, cv::FILTER_SCHARR);
    // cv::convertScaleAbs(grad_x, abs_grad_x);
    // // cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    // cv::Sobel(gray, grad_y, CV_16S, 0, 1, cv::FILTER_SCHARR);
    // cv::convertScaleAbs(grad_y, abs_grad_y);
    // // Total Gradient (approximate magnitude)
    // cv::Mat grad_magnitude;
    // cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_magnitude);
    // cv::imwrite(parent_path + "/" + root + "_grad_magnitude" + ext, grad_magnitude);
    // cv::Mat texture_energy;
    // // 卷积核大一些，比如 15x15 或 21x21, seems not good  均值模糊会保留线条，但中值滤波是线条的克星。
    // cv::blur(grad_magnitude, texture_energy, cv::Size(33, 33));
    // cv::imwrite(parent_path + "/" + root + "_grad_magnitude_blur" + ext, texture_energy);
    // cv::Mat median_grad;
    // cv::medianBlur(grad_magnitude, median_grad, 11); // 使用较大的核，如 9 或 11
    // cv::imwrite(parent_path + "/" + root + "_grad_magnitude_blur_m" + ext, median_grad);
    // 水泥地的六边形直线纹理太干净、太直了。在频域上，这种直线属于低频成分，普通的模糊很难滤除。

    // mask_grass should contain most part of grass, snow and soil might be excluded(which should be included), edges on concrete might be included (which should be excluded) 
    cv::Mat mask_grass;
    cv::Scalar lower_grass(0, 87, 0);
    cv::Scalar upper_grass(49, 255, 93);
    cv::inRange(hsv, lower_grass, upper_grass, mask_grass);
    cv::imwrite(parent_path + "/" + root + "_mask_grass" + ext, mask_grass); // white is grass

    // -------------
    // --- 3. 补洞处理 (闭运算 MORPH_CLOSE) ---
    // 目的：把草地里的雪洞、枯草空隙连成一片
    // 核的大小取决于雪洞的大小，建议用较大的核 (例如 25x25)
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));
    cv::Mat mask_closed;
    cv::morphologyEx(mask_grass, mask_closed, cv::MORPH_CLOSE, close_kernel);

    // --- 4. 去线处理 (开运算 MORPH_OPEN = 先腐蚀后膨胀) ---
    // 目的：抹去细长的水泥缝隙
    // 关键点：核的大小必须【大于】水泥缝隙的宽度。
    // 如果缝隙在图中是 5-10 像素宽，我们用 15x15 的核，缝隙就会因为“腐蚀”而彻底消失
    cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::Mat mask_opened;
    cv::morphologyEx(mask_closed, mask_opened, cv::MORPH_OPEN, open_kernel);

    // --- 5. 面积过滤 (保留最大连通域) ---
    // 目的：万一还有残余的杂质，只保留面积最大的那块“草地”
    std::vector<std::vector<cv::Point>> contours_grass;
    cv::findContours(mask_opened, contours_grass, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat grass_final = cv::Mat::zeros(img.size(), CV_8UC1);
    if (!contours_grass.empty()) {
        // 找到面积最大的轮廓
        auto it = std::max_element(contours_grass.begin(), contours_grass.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        
        // 填充该轮廓，彻底抹平内部残留的小孔
        cv::drawContours(grass_final, std::vector<std::vector<cv::Point>>{*it}, -1, 255, -1);
    }

    // Canny: Edge detection; outputs an unordered pixel mask (cv::Mat) for visualization.
    // findContours: Contour extraction; outputs ordered point sequences (vector<Point>) for geometric analysis.
    // cv::Mat boundary;
    // cv::Canny(grass_final, boundary, 100, 200);  // used on mask (binary), threshold param not very important
    // cv::Mat save_img = img.clone();
    // save_img.setTo(cv::Scalar(0, 0, 255), boundary);
    // cv::imwrite(output_path, save_image);
    std::vector<std::vector<cv::Point>> final_contours;
    cv::findContours(grass_final, final_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // visualization option 1: only draw contour
    // cv::Mat save_img = img.clone();
    // cv::drawContours(save_img, final_contours, -1, cv::Scalar(0, 255, 0), 3);  // -1: draw all contours, Scalar(0, 255, 0): green (BGR), 3: line width
    // visualization option 2: draw contour and overlay
    cv::Mat overlay = img.clone();
    cv::Scalar fillColor = cv::Scalar(0, 255, 0);
    cv::drawContours(overlay, final_contours, -1, fillColor, cv::FILLED);
    double alpha = 0.3;
    cv::Mat save_img;
    cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0, save_img);  // save_img = alpha * overlay + (1 - alpha) * img + 0
    cv::drawContours(save_img, final_contours, -1, cv::Scalar(0, 255, 0), 3);
    cv::imwrite(output_path, save_img);
    std::cout << "Boundary detection complete. Output saved to " << output_path << std::endl;
    return true;
    
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
    

    // Attempt to segment concrete
    // cv::Mat mask_concrete;
    // cv::Scalar lower_concrete(0, 0, 136);
    // cv::Scalar upper_concrete(25, 71, 224);
    // cv::inRange(hsv, lower_concrete, upper_concrete, mask_concrete);
    // cv::imwrite(parent_path + "/" + root + "_mask_concrete" + ext, mask_concrete); // Save concrete mask, white is concrete
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

void on_trackbar(int, void* userdata) {
    HSVContext* ctx = static_cast<HSVContext*>(userdata);
    cv::Scalar lower_bound(ctx->h_min, ctx->s_min, ctx->v_min);
    cv::Scalar upper_bound(ctx->h_max, ctx->s_max, ctx->v_max);

    cv::Mat mask;
    cv::inRange(ctx->hsv_img, lower_bound, upper_bound, mask);
    cv::Mat result;
    cv::bitwise_and(ctx->hsv_img, ctx->hsv_img, result, mask);
    cv::imshow("Segmented Result", mask);
    cv::imshow("Color Extracted", result);
}

void tuneHSVThreshold(const std::string& image_path, double scale) {
    cv::Mat img = cv::imread(image_path);

    cv::Mat processed_img;
    double final_scale = scale;

    if (scale <= 0.0) {
        // auto scale
        double max_w = 1280.0;
        double max_h = 720.0;
        if (img.cols > max_w || img.rows > max_h) {
            final_scale = std::min(max_w / img.cols, max_h / img.rows);
        } else {
            final_scale = 1.0;
        }
    }

    if (std::abs(final_scale - 1.0) > 0.001) {
        cv::resize(img, processed_img, cv::Size(), final_scale, final_scale, cv::INTER_AREA);
        std::cout << "Image resized by factor: " << final_scale << std::endl;
    } else {
        processed_img = img;
    }

    HSVContext ctx;
    cv::cvtColor(processed_img, ctx.hsv_img, cv::COLOR_BGR2HSV);
    cv::namedWindow("Segmented Result", cv::WINDOW_NORMAL);
    cv::namedWindow("Trackbars", cv::WINDOW_NORMAL);
    cv::createTrackbar("H_Min", "Trackbars", &ctx.h_min, 179, on_trackbar, &ctx);
    cv::createTrackbar("H_Max", "Trackbars", &ctx.h_max, 179, on_trackbar, &ctx);
    cv::createTrackbar("S_Min", "Trackbars", &ctx.s_min, 255, on_trackbar, &ctx);
    cv::createTrackbar("S_Max", "Trackbars", &ctx.s_max, 255, on_trackbar, &ctx);
    cv::createTrackbar("V_Min", "Trackbars", &ctx.v_min, 255, on_trackbar, &ctx);
    cv::createTrackbar("V_Max", "Trackbars", &ctx.v_max, 255, on_trackbar, &ctx);

    // initialize
    on_trackbar(0, &ctx);

    std::cout << "press any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyWindow("Segmented Result");
    cv::destroyWindow("Trackbars");
    std::printf("Final Thresholds: \nLower: [%d, %d, %d]\nUpper: [%d, %d, %d]\n", 
                ctx.h_min, ctx.s_min, ctx.v_min, ctx.h_max, ctx.s_max, ctx.v_max);
}

// Final Thresholds:
// Lower: [0, 87, 0]
// Upper: [49, 255, 93]