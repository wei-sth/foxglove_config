#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/pcl/pcl_registration_impl.hpp>

int test_single() {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZRGB>);

    //28 | 699 | 700
    std::string cloud_in_path = "/home/weizh/fast_livo_ws/src/FAST-LIVO2/Log/PCD/699.pcd";
    std::string cloud_target_path = "/home/weizh/fast_livo_ws/src/FAST-LIVO2/Log/PCD/300.pcd";
    std::string output_path = "/home/weizh/data/aligned_cloud_vgicp.pcd"; // Output path for aligned cloud

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(cloud_in_path, *cloud_in) == -1) {
        PCL_ERROR("Couldn't read file %s \n", cloud_in_path.c_str());
        return (-1);
    }
    std::cout << "Loaded " << cloud_in->size() << " data points from " << cloud_in_path << std::endl;


    // Load the target point cloud
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(cloud_target_path, *cloud_target) == -1) {
        PCL_ERROR("Couldn't read file %s \n", cloud_target_path.c_str());
        return -1;
    }
    std::cout << "Loaded " << cloud_target->size() << " data points from " << cloud_target_path << std::endl;


    // Create a VGICP object
    small_gicp::RegistrationPCL<pcl::PointXYZRGB, pcl::PointXYZRGB> vgicp;
    vgicp.setRegistrationType("VGICP"); // Specify VGICP algorithm
    vgicp.setInputSource(cloud_in);
    vgicp.setInputTarget(cloud_target);

    // Set parameters (optional, default values are often good)
    vgicp.setVoxelResolution(1.0); // Voxel grid resolution
    vgicp.setNumThreads(4);   // Number of threads for parallel computation
    vgicp.setMaxCorrespondenceDistance(1.0); // Max correspondence distance
    vgicp.setNumNeighborsForCovariance(40);  // 20 seems too small for original xt16 (56350 points per scan), 40 is OK
    vgicp.setMaximumIterations(100); // Max number of ICP iterations

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_aligned(new pcl::PointCloud<pcl::PointXYZRGB>);
    vgicp.align(*cloud_aligned);

    if (vgicp.hasConverged()) {
        std::cout << "\nVGICP has converged." << std::endl;
        std::cout << "Transformation matrix:" << std::endl;
        std::cout << vgicp.getFinalTransformation() << std::endl;

        // Save the aligned point cloud to a PCD file
        pcl::io::savePCDFileBinary(output_path, *cloud_aligned);
        std::cout << "Saved aligned cloud to " << output_path << std::endl;
    } else {
        std::cerr << "\nVGICP did not converge." << std::endl;
        pcl::io::savePCDFileBinary(output_path, *cloud_aligned);
    }

    return 0;
}

int main(int argc, char** argv) {
    return test_single();
}
