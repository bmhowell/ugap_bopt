#include <iostream>
#include <random>
#include <fstream>

#include "GaussianProcess.h"
#include "Voxel.h"
#include "common.h"


/*
- 

*/


// declare functions
int   find_arg_idx(int argc, char** argv, const char* option); 
void  gen_data(float tfinal, double dt, int node, int idsim, bopt& bopti, sim& simi, std::string file_path); 
void  bootstrap(std::vector<bopt>* bopt, int num_sims);
void  write_to_file(bopt& b, int id); 
void  store_tot_data(std::vector<bopt>* bopti, int num_sims); 
int   read_data(std::vector<bopt>* bopti); 
void  to_eigen(std::vector<bopt>* data, Eigen::MatrixXd* X, Eigen::MatrixXd* Y);

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
    simi.bootstrap    = 0;      // bootstrap data

    std::string file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/materials_opt/output";   // MACBOOK PRO
    // file_path = "/home/brian/Documents/berkeley/materials_opt/output/";         // LINUX CENTRAL COMPUTING

    // https://stackoverflow.com/questions/8036474/when-vectors-are-allocated-do-they-use-memory-on-the-heap-or-the-stack
    std::vector<bopt> *bopti = new std::vector<bopt>; // stores all info (header + elements) on heap

    // STEP 1: sample data
    int ndata0;
    if (simi.bootstrap != 1){
        ndata0 = read_data(bopti);
    }else{
        ndata0 = 10; 
        bootstrap(bopti, ndata0);
    }

    // convert data to Eigen matrixs
    Eigen::MatrixXd* p_x_train = new Eigen::MatrixXd(ndata0, 5);  // ∈ ℝ^(ndata x 5)
    Eigen::MatrixXd* p_y_train = new Eigen::MatrixXd(ndata0, 5);  // ∈ ℝ^(ndata x 1)
    to_eigen(bopti, p_x_train, p_y_train);
    
    // set up gaussian process
    GaussianProcess gp_ugap(1.0f, 1.0f, "RBF");

    // int num_sims = 10000; 
    // int ndata    = ndata0; 
    // for (int id = 0; id < num_sims; ++id) {

    //     // STEP 2: fit model
    //     bopt b; 
    //     gen_data(TFINAL, DT, NODE, id, b, simi, file_path);
    //     write_to_file(b, id); 

    //     // store data point
    //     bopti->push_back(b);
    //     ndata++; 
    // }

    // // store data
    // store_tot_data(bopti, num_sims);
    delete bopti;
    
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
void gen_data(float tfinal, double dt, int node, int idsim, bopt& bopti, sim& simi, std::string file_path) {
    
    // unpack input data
    double temp = bopti.temp;
    float rp    = bopti.rp;
    float vp    = bopti.vp;
    float uvi   = bopti.uvi;
    float uvt   = bopti.uvt;

    // objective
    float obj  = bopti.obj;

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
    
    
    // run simulation
    Voxel VoxelSystem1(tfinal, dt, node, idsim, temp, uvi, uvt, file_path);
    VoxelSystem1.ComputeParticles(rp, vp);
    if (sv_d == 1){
        VoxelSystem1.Density2File();
    }

    VoxelSystem1.Simulate(nm, sv_v);

}

// initialize input variables
void bootstrap(std::vector<bopt> *bopti, int num_sims) {

    // initialize input variables
    std::random_device rd;                                          // Obtain a random seed from the hardware
    std::mt19937 gen(rd());                                         // Seed the random number generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)

    // constraints
    float btemp[2] = {273.15, 350.0};
    float brp[2]   = {0.00084 / 200, 0.00084 / 10};
    float bvp[2]   = {0.5, 0.8};
    float buvi[2]  = {2.0, 100.0};
    float buvt[2]  = {1.0, 30.0};

    // generate random values
    for (int id = 0; id < num_sims; ++id) {
        bopt b; 
        b.temp = (btemp[1] - btemp[0]) * distribution(gen) + btemp[0];
        b.rp   = (brp[1] - brp[0])     * distribution(gen) + brp[0];
        b.vp   = (bvp[1] - bvp[0])     * distribution(gen) + bvp[0];
        b.uvi  = (buvi[1] - buvi[0])   * distribution(gen) + buvi[0];
        b.uvt  = (buvt[1]  - buvt[0])  * distribution(gen) + buvt[0];

        b.obj  = 1000.0;

        bopti->push_back(b); 
    }
}

void write_to_file(bopt& b, int id){
    std::ofstream myfile;
    myfile.open("output/sim_" + std::to_string(id) + ".dat");
    myfile << "temp,rp,vp,uvi,uvt,obj" << std::endl;
    myfile << b.temp << "," << b.rp << "," << b.vp << "," << b.uvi << "," << b.uvt << "," << b.obj << std::endl;
    myfile.close();
}

void store_tot_data(std::vector<bopt> *bopti, int num_sims){
    std::ofstream myfile;
    myfile.open("output/tot_bopt.dat");
    myfile << "temp,rp,vp,uvi,uvt,obj" << std::endl;
    for (int id = 0; id < num_sims; ++id) {
        myfile << (*bopti)[id].temp << "," 
               << (*bopti)[id].rp   << "," 
               << (*bopti)[id].vp   << ","
               << (*bopti)[id].uvi  << "," 
               << (*bopti)[id].uvt  << "," 
               << (*bopti)[id].obj  << std::endl;
    }
    myfile.close();
}

int  read_data(std::vector<bopt> *bopti){
    std::ifstream file("output/tot_bopt.dat");
    std::string line;
    std::getline(file, line); // skip first line
    int id = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;

        // create new bopt struct for data point
        bopt b; 

        // Parse the comma-separated values in the line
        std::getline(iss, token, ',');
        b.temp = std::stof(token);

        std::getline(iss, token, ',');
        b.rp = std::stof(token);

        std::getline(iss, token, ',');
        b.vp = std::stof(token);

        std::getline(iss, token, ',');
        b.uvi = std::stof(token);

        std::getline(iss, token, ',');
        b.uvt = std::stof(token);

        std::getline(iss, token, ',');
        b.obj = std::stof(token);

        bopti->push_back(b);

        id++;
    }

    // return number of data points
    return id; 
}

void  to_eigen(std::vector<bopt>* data, Eigen::MatrixXd* X, Eigen::MatrixXd* Y){
    for (int i = 0; i < (*data).size(); ++i) {
        (*X)(i, 0) = (*data)[i].temp;
        (*X)(i, 1) = (*data)[i].rp;
        (*X)(i, 2) = (*data)[i].vp;
        (*X)(i, 3) = (*data)[i].uvi;
        (*X)(i, 4) = (*data)[i].uvt;
        (*Y)(i, 0) = (*data)[i].obj;
    }
}