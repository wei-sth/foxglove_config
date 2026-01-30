#include "boundary_detector.h"

int main(int argc, char** argv) {
    std::string input_image_name = "/home/weizh/data/20260127200629.jpg";
    std::string output_image_name = "/home/weizh/data/20260127200629_boundary.jpg";
    
    if (detectBoundary(input_image_name, output_image_name)) {
        return 0;
    } else {
        return 1;
    }
}