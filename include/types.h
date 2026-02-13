#pragma once
#ifndef TYPES_H
#define TYPES_H
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// robosense airy, time is relative time, unit:ns
struct RSPointDefault {
    PCL_ADD_POINT4D;
    float intensity;
    uint16_t ring;
    uint32_t timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(RSPointDefault,
    (float, x, x)(float, y, y)(float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (uint32_t, timestamp, timestamp))

// A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

#endif // TYPES_H