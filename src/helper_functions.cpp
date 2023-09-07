#include <iostream>
#include <random>
#include <fstream>

#include "GaussianProcess.h"
#include "Voxel.h"
#include "common.h"
#include "helper_functions.h"  // Include the header file


// Generate data
double gen_data(float tfinal, double dt, int node, int idsim, bopt& bopti, sim& simi, std::string file_path) {

    // // objective function value
    // float obj  = bopti.obj;

    std::cout << "================ begin simulation ================" << std::endl;
    std::cout << "id sim: " << idsim       << std::endl;
    std::cout << "temp: "   << bopti.temp  << std::endl;
    std::cout << "rp: "     << bopti.rp    << std::endl;
    std::cout << "vp: "     << bopti.vp    << std::endl;
    std::cout << "uvi: "    << bopti.uvi   << std::endl;
    std::cout << "uvt: "    << bopti.uvt   << std::endl;
    std::cout                              << std::endl;
    
    // run simulation
    auto start = std::chrono::high_resolution_clock::now();
    Voxel VoxelSystem1( tfinal, 
                        dt, 
                        node, 
                        idsim, 
                        bopti.temp,  
                        bopti.uvi, 
                        bopti.uvt, 
                        file_path);

    VoxelSystem1.ComputeParticles(bopti.rp, bopti.vp);
    if (simi.save_density == 1){
        VoxelSystem1.Density2File();
    }

    VoxelSystem1.Simulate(simi.method, simi.save_voxel);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count() / 1e6;

    std::cout << " --- Simulation time: " << duration / 60 << "min ---" << std::endl;
    std::cout << "testing obj: " << VoxelSystem1.obj << std::endl;
    return VoxelSystem1.obj; 
}

// initialize input variables
void bootstrap(sim &sim_settings, constraints &c, std::vector<bopt> *bopti, int num_sims, std::string file_path) {

    // initialize input variables
    std::random_device rd;                                          // Obtain a random seed from the hardware
    std::mt19937 gen(rd());                                         // Seed the random number generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)

    // generate random values
    for (int id = 0; id < num_sims; ++id) {
        bopt b; 
        b.temp = (c.max_temp - c.min_temp) * distribution(gen) +  c.min_temp;
        b.rp   = (c.max_rp   - c.min_rp)   * distribution(gen) +  c.min_rp;
        b.vp   = (c.max_vp   - c.min_vp)   * distribution(gen) +  c.min_vp;
        b.uvi  = (c.max_uvi  - c.min_uvi)  * distribution(gen) +  c.min_uvi;
        b.uvt  = (c.max_uvt  - c.min_uvt)  * distribution(gen) +  c.min_uvt;

        // peform simulation with randomly generatored values
        b.obj = gen_data(sim_settings.tfinal, sim_settings.dt, sim_settings.node, id, b, sim_settings, file_path);
        std::cout << "b.obj: " << b.obj << std::endl;
        std::cout << std::endl; 
        // write individual data to file (prevent accidental loss of data if stopped early)
        write_to_file(b, sim_settings, id, file_path); 

        bopti->push_back(b); 
    }
}

void write_to_file(bopt& b, sim& sim_set, int id, std::string file_path){
    std::ofstream myfile;
    myfile.open(file_path + "/sim_" + std::to_string(id) + ".dat");
    myfile << "temp,rp,vp,uvi,uvt,obj,tn" << std::endl;
    myfile << b.temp << "," << b.rp << "," << b.vp << "," << b.uvi << "," << b.uvt << "," << b.obj << "," << sim_set.time_stepping << std::endl;
    myfile.close();
}

void store_tot_data(std::vector<bopt> *bopti, sim& sim_set, int num_sims, std::string file_path){
    std::cout << "--- storing data ---\n" << std::endl;
    std::ofstream myfile;
    myfile.open(file_path + "/tot_bopt.dat");
    myfile << "temp,rp,vp,uvi,uvt,obj,tn" << std::endl;
    for (int id = 0; id < num_sims; ++id) {
        myfile << (*bopti)[id].temp << "," 
               << (*bopti)[id].rp   << "," 
               << (*bopti)[id].vp   << ","
               << (*bopti)[id].uvi  << "," 
               << (*bopti)[id].uvt  << "," 
               << (*bopti)[id].obj  << ","
               << sim_set.time_stepping << std::endl;
    }
    myfile.close();
}

int  read_data(std::vector<bopt> *bopti, std::string file_path){
    std::ifstream file(file_path + "/tot_bopt.dat"); 

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

void build_dataset(std::vector<bopt>* bopti,
                   Eigen::MatrixXd*   X_TRAIN, Eigen::VectorXd* Y_TRAIN,
                   Eigen::MatrixXd*   X_VAL,   Eigen::VectorXd* Y_VAL){

    // split data into training and validation sets
    int num_data = (*bopti).size();
    int num_train = 0.9 * num_data;
    int num_val   = 0.1 * num_data;

    // resize X_TRAIN, Y_TRAIN, X_VAL, and Y_VAL
    X_TRAIN->resize(num_train, 5);
    Y_TRAIN->resize(num_train);
    X_VAL  ->resize(num_val, 5);
    Y_VAL  ->resize(num_val);
    
    // shuffle dataset
    std::random_device rd;
    // std::mt19937 g(rd());
    std::mt19937 g(47);
    std::shuffle((*bopti).begin(), (*bopti).end(), g);

    // initialize training and validation sets
    *X_TRAIN = Eigen::MatrixXd(num_train, 5);
    *Y_TRAIN = Eigen::VectorXd(num_train);
    *X_VAL   = Eigen::MatrixXd(num_val, 5);
    *Y_VAL   = Eigen::VectorXd(num_val);

    // populate training and validation sets
    for (int i = 0; i < num_train; ++i) {
        (*X_TRAIN)(i, 0) = (*bopti)[i].temp;
        (*X_TRAIN)(i, 1) = (*bopti)[i].rp;
        (*X_TRAIN)(i, 2) = (*bopti)[i].vp;
        (*X_TRAIN)(i, 3) = (*bopti)[i].uvi;
        (*X_TRAIN)(i, 4) = (*bopti)[i].uvt;
        (*Y_TRAIN)(i)    = (*bopti)[i].obj;
    }

    for (int i = 0; i < num_val; ++i) {
        (*X_VAL)(i, 0) = (*bopti)[i + num_train].temp;
        (*X_VAL)(i, 1) = (*bopti)[i + num_train].rp;
        (*X_VAL)(i, 2) = (*bopti)[i + num_train].vp;
        (*X_VAL)(i, 3) = (*bopti)[i + num_train].uvi;
        (*X_VAL)(i, 4) = (*bopti)[i + num_train].uvt;
        (*Y_VAL)(i)    = (*bopti)[i + num_train].obj;
    }

}


void gen_test_points(constraints&     c, 
                     Eigen::MatrixXd& X){

    // initialize input variables
    std::random_device rd;                                          // Obtain a random seed from the hardware
    std::mt19937 gen(rd());                                         // Seed the random number generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)

    for (int ind = 0; ind<X.rows(); ++ind){
        X(ind, 0) = (c.max_temp - c.min_temp) * distribution(gen) + c.min_temp;
        X(ind, 1) = (c.max_rp - c.min_rp)     * distribution(gen) + c.min_rp;
        X(ind, 2) = (c.max_vp - c.min_vp)     * distribution(gen) + c.min_vp;
        X(ind, 3) = (c.max_uvi - c.min_uvi)   * distribution(gen) + c.min_uvi;
        X(ind, 4) = (c.max_uvt - c.min_uvt)   * distribution(gen) + c.min_uvt;
    }

}