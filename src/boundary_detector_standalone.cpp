#include "boundary_detector.h"

int main(int argc, char** argv) {
    // 20260127200617 | 20260127200629
    std::string input_image_name = "/home/weizh/data/20260127200617.jpg";
    std::string output_image_name = "/home/weizh/data/20260127200617_boundary.jpg";

    detectBoundary(input_image_name, output_image_name);
    // tuneHSVThreshold(input_image_name);
    return 0;
}