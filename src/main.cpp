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
int     find_arg_idx(int argc, char** argv, const char* option); 
double  gen_data(float tfinal, double dt, int node, int idsim, bopt &bopti, sim& simi, std::string file_path); 
void    bootstrap(sim &sim_settings, constraints &c, std::vector<bopt>* bopt, int num_sims, std::string file_path);
void    write_to_file(bopt &b, int id, std::string file_path); 
void    store_tot_data(std::vector<bopt>* bopti, int num_sims, std::string file_path); 
int     read_data(std::vector<bopt>* bopti); 
void    to_eigen(std::vector<bopt>* data, Eigen::MatrixXd* X, Eigen::VectorXd* Y);
void    gen_test_points(constraints &c, Eigen::MatrixXd* X); 

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

    // optimization constraints
    constraints c; 
    // simulation settings
    sim sim_settings;
    sim_settings.bootstrap = 1; 

    // set file path
    std::string file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/materials_opt/output";   // MACBOOK PRO
    // file_path = "/home/brian/Documents/berkeley/materials_opt/output/";         // LINUX CENTRAL COMPUTING

    // https://stackoverflow.com/questions/8036474/when-vectors-are-allocated-do-they-use-memory-on-the-heap-or-the-stack
    std::vector<bopt> *bopti = new std::vector<bopt>; // stores all info (header + elements) on heap

    // STEP 1: sample data
    int ndata0;
    if (sim_settings.bootstrap == 0){
        ndata0 = read_data(bopti);
    }else{
        ndata0 = 10; 
        bootstrap(sim_settings, c, bopti, ndata0, file_path);
        
        // store data
        store_tot_data(bopti, ndata0, file_path);
    }


    // // convert data to Eigen matrices
    // Eigen::MatrixXd* x_train = new Eigen::MatrixXd(ndata0, 5);  // ∈ ℝ^(ndata x 5)
    // Eigen::VectorXd* y_train = new Eigen::VectorXd(ndata0);     // ∈ ℝ^(ndata x 1)
    // to_eigen(bopti, x_train, y_train);

    // std::cout << "x_train: \n" << *x_train << std::endl;
    
    // // set up gaussian process
    // GaussianProcess gp_ugap = GaussianProcess(1.0f, 1.0f, "RBF", file_path); 

    // // uniformly random x_test data for GP
    // Eigen::MatrixXd* x_test = new Eigen::MatrixXd(25, 5);  // 5 decision variables | 25 test points
    // Eigen::VectorXd* y_test = new Eigen::VectorXd(25);     // objective function values

    // gen_test_points(c, x_test); 
    // gp_ugap.predict(*x_test, *x_train, *y_test, *y_train, 'y'); 

    // int num_sims = 10000; 
    // int ndata    = ndata0; 

    // // // OPTIMISATION LOOP
    // // for (int id = 0; id < num_sims; ++id) {

    // //     // generate random points
        

    // //     // STEP 2: fit model
        


    // //     bopt b; 
    // //     gen_data(TFINAL, DT, NODE, id, b, simi, file_path);
    // //     write_to_file(b, id); 

    // //     store data point
    // //     bopti->push_back(b);
    // //     ndata++; 
    // // }

    // // store data
    // store_tot_data(bopti, ndata0, file_path);
    
    // delete y_train;
    // delete x_train;
    // delete x_test;
    // delete y_test;
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
        b.obj = gen_data(TFINAL, DT, NODE, id, b, sim_settings, file_path);
        std::cout << "b.obj: " << b.obj << std::endl;
        std::cout << std::endl; 
        // write individual data to file (prevent accidental loss of data if stopped early)
        write_to_file(b, id, file_path); 

        bopti->push_back(b); 
    }
}

void write_to_file(bopt& b, int id, std::string file_path){
    std::ofstream myfile;
    myfile.open(file_path + "/sim_" + std::to_string(id) + ".dat");
    myfile << "temp,rp,vp,uvi,uvt,obj" << std::endl;
    myfile << b.temp << "," << b.rp << "," << b.vp << "," << b.uvi << "," << b.uvt << "," << b.obj << std::endl;
    myfile.close();
}

void store_tot_data(std::vector<bopt> *bopti, int num_sims, std::string file_path){
    std::ofstream myfile;
    myfile.open(file_path + "/tot_bopt.dat");
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
    std::ifstream file("../output/tot_bopt.dat");
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

void to_eigen(std::vector<bopt>* data, Eigen::MatrixXd* X, Eigen::VectorXd* Y){
    for (int i = 0; i < (*data).size(); ++i) {
        (*X)(i, 0) = (*data)[i].temp;
        (*X)(i, 1) = (*data)[i].rp;
        (*X)(i, 2) = (*data)[i].vp;
        (*X)(i, 3) = (*data)[i].uvi;
        (*X)(i, 4) = (*data)[i].uvt;
        (*Y)(i)    = (*data)[i].obj;
    }
}

void gen_test_points(constraints& c, Eigen::MatrixXd* X){
    // initialize input variables
    std::random_device rd;                                          // Obtain a random seed from the hardware
    std::mt19937 gen(rd());                                         // Seed the random number generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)

    for (int ind = 0; ind<X->rows(); ++ind){
        (*X)(ind, 0) = (c.max_temp - c.min_temp) * distribution(gen) + c.min_temp;
        (*X)(ind, 1) = (c.max_rp - c.min_rp)     * distribution(gen) + c.min_rp;
        (*X)(ind, 2) = (c.max_vp - c.min_vp)     * distribution(gen) + c.min_vp;
        (*X)(ind, 3) = (c.max_uvi - c.min_uvi)   * distribution(gen) + c.min_uvi;
        (*X)(ind, 4) = (c.max_uvt - c.min_uvt)   * distribution(gen) + c.min_uvt;
    }

}
