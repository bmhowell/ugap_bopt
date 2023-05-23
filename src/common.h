#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Program Constants
#define TFINAL   30.0
#define NODE     21
#define DT       1e-3

// input/output BOpt data structure
typedef struct bopt {
    float temp;    // temperature
    float rp;      // particle radius
    float vp;      // volume fraction
    float uvi;     // initial UV intensity
    float uvt;     // uv exposure time
    float obj;     // objective
} bopt;

// sim settings
typedef struct sim{
    int method;         // simulation method
    int save_voxel;     // save voxel data
    int save_density;   // save density data
} sim;

#endif