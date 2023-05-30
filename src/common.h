#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Program Constants
#define TFINAL   30.0
#define NODE     11        // 11,      15,        21,   25,     31,     41,   51
#define DT       1.25e-3   // 1.25e-3, 7.25e-4,   3.5e-4, 2.5e-4, 1.5e-4, 9.3e-5, 5.9e-5 (FOR SHANNA DM0)


// input/output BOpt data structure
typedef struct bopt{
    float temp;    // temperature
    float rp;      // particle radius
    float vp;      // volume fraction
    float uvi;     // initial UV intensity
    float uvt;     // uv exposure time
    float obj;     // objective
} bopt;

// input constraints
typedef struct constraints{
    float min_temp;  // temperature
    float min_rp;    // particle radius
    float min_vp;    // volume fraction
    float min_uvi;   // initial UV intensity
    float min_uvt;   // uv exposure time

    float max_temp;  // temperature
    float max_rp;    // particle radius
    float max_vp;    // volume fraction
    float max_uvi;   // initial UV intensity
    float max_uvt;   // uv exposure time
    
} constraints;

// sim settings
typedef struct sim{
    int method;         // simulation method
    int save_voxel;     // save voxel data
    int save_density;   // save density data
    int bootstrap;      // bootstrap
} sim;

#endif