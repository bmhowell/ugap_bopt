#ifndef __COMMON_H__
#define __COMMON_H__
#pragma once

// Program Constants
// #define TFINAL   30.0
// #define NODE     11        // {11,     15,     21,   25,     31,     41,   51}
// #define DT       1.5e-3    // {1.5e-3, 7.5e-4,   3.5e-4, 2.5e-4, 1.5e-4, 9.25e-5, 5.9e-5} (FOR SHANNA DM0)


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
typedef struct constraints {
    float min_temp;
    float min_rp;
    float min_vp;
    float min_uvi;
    float min_uvt;

    float max_temp;
    float max_rp;
    float max_vp;
    float max_uvi;
    float max_uvt;

    // Default constructor
    constraints() {
        min_temp = 273.15;
        max_temp = 350.0;
        min_rp = 0.00084 / 200;
        max_rp = 0.00084 / 10;
        min_vp = 0.5;
        max_vp = 0.8;
        min_uvi = 2.0;
        max_uvi = 100.0;
        min_uvt = 1.0;
        max_uvt = 30.0;
    }

    // Overloaded constructor
    constraints(float minTemp, float maxTemp, float minRp, float maxRp,
                float minVp, float maxVp, float minUvi, float maxUvi,
                float minUvt, float maxUvt)
        : min_temp(minTemp), max_temp(maxTemp), min_rp(minRp), max_rp(maxRp),
          min_vp(minVp), max_vp(maxVp), min_uvi(minUvi), max_uvi(maxUvi),
          min_uvt(minUvt), max_uvt(maxUvt) {
    }

} constraints;


// sim settings
typedef struct sim{
    int method;         // simulation method
    int save_voxel;     // save voxel data
    int save_density;   // save density data
    int bootstrap;      // bootstrap
    int time_stepping;  // representing with dt/node pair

    double DT[7]   = {1.5e-3, 7.5e-4,   3.5e-4, 2.5e-4, 1.5e-4, 9.25e-5, 5.9e-5};
    int    NODE[7] = {11,     15,     21,   25,     31,     41,   51};
    
    float  tfinal  = 30.;   // final time
    double dt;              // time step
    int    node;            // number of nodes

    // default constructor
    sim() {
        method = 2;        // forward euler | 1: backward euler | 2: trap
        save_voxel = 0;    // save voxel data
        save_density = 0;  // save density data
        bootstrap = 0;     // bootstrap
        time_stepping = 1; // representing with dt/node pair
        dt = DT[time_stepping];
        node = NODE[time_stepping];
    }

    // overload constructor
    sim(int method, int save_voxel, int save_density, int bootstrap, int time_stepping)
        : method(method), save_voxel(save_voxel), 
          save_density(save_density), bootstrap(bootstrap), 
          time_stepping(time_stepping) {

        dt = DT[time_stepping];
        node = NODE[time_stepping];
    }
} sim;


#endif