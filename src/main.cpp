#include <iostream>
#include <random>
#include "GaussianProcess.h"
#include "Voxel.h"
#include "common.h"


// declare functions
int   find_arg_idx(int argc, char** argv, const char* option); 
float gen_data(float tfinal, double dt, int node, int idsim, bopt* bopti, sim& simi); 
void  init_input(bopt* bopti, int num_sims);

int main(int argc, char** argv) {

    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set number of nodes" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        return 0;
    }

    // simulation settings
    sim simi;
    simi.method       = 2;      // forward euler | 1: backward euler | 2: trap
    simi.save_voxel   = 0;      // save voxel data
    simi.save_density = 0;      // save density data

    // data
    int num_sims = 100;
    bopt* bopti  = new bopt[num_sims];
    init_input(bopti, num_sims);

    // run simulations
    float obj; 
    for (int id = 0; id < num_sims; ++id) {
        obj = gen_data(TFINAL, DT, NODE, id, bopti, simi);
        bopti[id].obj = obj;
    }

    delete[] bopti;
    
    std::cout << "Hello World!" << std::endl;
    return 0;
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

// Generate data
float gen_data(float tfinal, double dt, int node, int idsim, bopt* bopti, sim& simi) {
    
    // unpack input data
    double temp = (*bopti).temp;
    float rp    = (*bopti).rp;
    float vp    = (*bopti).vp;
    float uvi   = (*bopti).uvi;
    float uvt   = (*bopti).uvt;

    std::cout << "================ begin simulation ================" << std::endl;
    std::cout << "id sim: " << idsim << std::endl;
    std::cout << "temp: "   << temp  << std::endl;
    std::cout << "rp: "     << rp    << std::endl;
    std::cout << "vp: "     << vp    << std::endl;
    std::cout << "uvi: "    << uvi   << std::endl;
    std::cout << "uvt: "    << uvt   << std::endl;
    std::cout                        << std::endl;


    // unpack simulation settings
    int   nm    = simi.method;          // numerical method
    int   sv_v  = simi.save_voxel;      // save voxels
    int   sv_d  = simi.save_density;    // save density
    
    // objective
    float obj  = (*bopti).obj;
    
    // run simulation
    Voxel VoxelSystem1(tfinal, dt, node, idsim, temp, uvi, uvt);
    std::cout << "before" << std::endl;
    VoxelSystem1.ComputeParticles(rp, vp);
    if (sv_d == 1){
        VoxelSystem1.Density2File();
    }

    VoxelSystem1.Simulate(nm, sv_v);

    return obj;
}

// initialize input variables
void init_input(bopt* bopti, int num_sims) {
    // initialize input variables
    std::random_device rd;  // Obtain a random seed from the hardware
    std::mt19937 gen(rd()); // Seed the random number generator

    std::uniform_real_distribution<double> distribution(0.0, 1.0); // Define the range [0.0, 1.0)

    // Generate a random number between 0 and 1
    double randomValue = distribution(gen);

    std::cout << "Random value between 0 and 1: " << randomValue << std::endl;
    for (int i = 0; i < num_sims; ++i) {

        // generate 
        bopti[i].temp = (350 - 273.15) * distribution(gen) + 273.15;
        bopti[i].rp   = ((0.00084 / 10) - (0.00084 / 200)) * distribution(gen) + (0.00084 / 200);
        bopti[i].vp   = (.8 - 0.5) * distribution(gen) + 0.5;
        bopti[i].uvi  = (100 - 2)  * distribution(gen) + 2;
        bopti[i].uvt  = (30  - 1)  * distribution(gen) + 1;
        bopti[i].obj  = 1000.0;
    }
}
