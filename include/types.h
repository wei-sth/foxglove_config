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

#endif // TYPES_H