#include <iostream>
#include <random>
#include <fstream>

#include "GaussianProcess.h"
#include "Voxel.h"
#include "common.h"
#include "helper_functions.h"

/*
- 

*/


int main(int argc, char** argv) {

    auto start = std::chrono::high_resolution_clock::now();

    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set number of nodes" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        return 0;
    }

    // optimization constraints (default) and simulation settings (default)
    constraints c; 
    sim         s;
    s.bootstrap = 0;
    s.time_stepping = 0;
    s.updateTimeSteppingValues();

    // set file path
    // std::string file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/materials_opt/output";   // MACBOOK PRO
    std::string file_path = "/home/brian/Documents/berkeley/opt_ugap/output_" + std::to_string(s.time_stepping);         // LINUX CENTRAL COMPUTING

    // https://stackoverflow.com/questions/8036474/when-vectors-are-allocated-do-they-use-memory-on-the-heap-or-the-stack
    std::vector<bopt> *bopti = new std::vector<bopt>; // stores all info (header + elements) on heap

    // STEP 1: sample data
    int ndata0;
    if (s.bootstrap == 0){
        ndata0 = read_data(bopti, file_path);
    }else{
        ndata0 = 1000; 
        bootstrap(s, c, bopti, ndata0, file_path);
        
        // store data
        store_tot_data(bopti, s, ndata0, file_path);
    }


    // convert data to Eigen matrices
    Eigen::MatrixXd* x_train = new Eigen::MatrixXd(ndata0, 5);  // ∈ ℝ^(ndata x 5)
    Eigen::VectorXd* y_train = new Eigen::VectorXd(ndata0);     // ∈ ℝ^(ndata x 1)
    
    to_eigen(bopti, x_train, y_train);
    
    // set up gaussian process
    GaussianProcess model = GaussianProcess("RBF", file_path); 
    
    // // pre-learned parameters
    std::vector<double> model_param; // = {0.835863, 0.0962956, 0.000346019};  // obj -> -133.356
    
    // if available, define model parameters: length, signal variance, noise variance
    int pre_learned = false; 

    if (pre_learned){
        model.train(*x_train, *y_train, model_param);
    }else{
        model.train(*x_train, *y_train);
    }
    
    // model.train(*x_train, *y_train, model_param);

    // // generate test vector by uniformly random x_test data for GP
    // int num_test = 25; 
    // Eigen::MatrixXd  x_test  = Eigen::MatrixXd(num_test, 5);             // 5 decision variables | 25 test points
    // gen_test_points(c, x_test);  
    // model.predict(x_test, 'y');
    
    // std::cout << "x_test: \n"   << x_test << std::endl; 
    // std::cout << "\ny_test: \n" << model.get_y_test() << std::endl;



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
    
    delete y_train;
    delete x_train;
    // delete x_test; 

    delete bopti;

    // Get the current time after the code segment finishes
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration of the code segment in hours
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end - start).count();

    std::cout << "Time taken by code segment: " << duration/60 << " hours" << std::endl;

    
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
