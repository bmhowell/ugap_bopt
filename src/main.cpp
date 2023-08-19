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
    sim sim_settings;
    // sim_settings.bootstrap = 1;

    // set file path
    std::string file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/materials_opt/output";   // MACBOOK PRO
    // file_path = "/home/brian/Documents/berkeley/materials_opt/output/";         // LINUX CENTRAL COMPUTING

    // https://stackoverflow.com/questions/8036474/when-vectors-are-allocated-do-they-use-memory-on-the-heap-or-the-stack
    std::vector<bopt> *bopti = new std::vector<bopt>; // stores all info (header + elements) on heap

    // STEP 1: sample data
    int ndata0;
    if (sim_settings.bootstrap == 0){
        ndata0 = read_data(bopti, file_path);
    }else{
        ndata0 = 2; 
        bootstrap(sim_settings, c, bopti, ndata0, file_path);
        
        // store data
        store_tot_data(bopti, sim_settings, ndata0, file_path);
    }


    // convert data to Eigen matrices
    Eigen::MatrixXd* x_train = new Eigen::MatrixXd(ndata0, 5);  // ∈ ℝ^(ndata x 5)
    Eigen::VectorXd* y_train = new Eigen::VectorXd(ndata0);     // ∈ ℝ^(ndata x 1)
    
    to_eigen(bopti, x_train, y_train);

    // std::cout << "x_train: \n" << *x_train << std::endl;
    // std::cout << "y_train: \n" << *y_train << std::endl;
    
    // set up gaussian process
    GaussianProcess model = GaussianProcess(0.75f, 0.1f, "RBF", file_path); 
    
    // set up training data for model
    model.train(x_train, y_train);

    // test vector
    int num_test = 25; 
    
    // uniformly random x_test data for GP
    Eigen::MatrixXd* x_test = new Eigen::MatrixXd(num_test, 5);         // 5 decision variables | 25 test points
    Eigen::MatrixXd  sub_mat = (*x_train).block(0, 0, num_test, 5);     // ∈ ℝ^(ndata x 5
    gen_test_points(c, x_test); 
    
    // predict
    // model.predict(sub_mat, *x_train, *y_test, *y_train, 'y'); 
    model.predict(x_test, 'y'); 
    

    std::cout << "x_test: \n" << *x_test << std::endl; 
    std::cout << "\ny_test: \n" << model.y_test << std::endl;



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
    delete x_test;
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
