// Copyright 2023 Brian Howell
// MIT License
// Project: BayesOpt

#include "GaussianProcess.h"
#include "BayesianOpt.h"
#include "Voxel.h"
#include "helper_functions.h"
#include "common.h"

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    //////////////  PARSE ARGS  //////////////
    // opt constrazints (default) and sim settings (default)
    constraints c;
    sim         s;
    s.bootstrap     = true;
    s.time_stepping = 0;
    s.update_time_stepping_values();

    // CHOOSE OBJECTIVE FUNCTION
    // 1: PI, 2: PIdot, 3: Mdot, 4: M, 5: total
    int obj_fn = 5;
    double default_weights[4] = {0.1, 0.2, 0.2, 0.5};
    double pareto_weights[4]  = {3.56574286e-09, 2.42560512e-03, 2.80839829e-01, 7.14916061e-01};

    // MACBOOK PRO
    std::string file_path;
    file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_opt/output_"
               + std::to_string(s.time_stepping);

    // LINUX CENTRAL COMPUTING
    // file_path = "/home/brian/Documents/brian/ugap_opt/output_"
    //             + std::to_string(s.time_stepping);

    std::cout << "\n--- INITIALIZING OPT. FRAMEWORK ---" << std::endl;
    std::cout << "saving to: " << file_path << std::endl;
    std::cout << "time_stepping: " << s.time_stepping << std::endl;

    // initialize function approximator with kernel choice {"RBF", "RQK", "LOC_PER"}
    std::string kernel = "LOC_PER";
    std::cout << "MODEL: " << kernel << std::endl;

    // data storage
    std::vector<bopt> *bopti = new std::vector<bopt>;

    // STEP 1: retrieve data set
    int ndata0;
    bool multi_thread = true;
    if (s.bootstrap) {
        ndata0 = omp_get_num_procs();
        std::cout << "Number of threads: " << ndata0 << std::endl;
        bootstrap(s, c, *bopti, ndata0, file_path, multi_thread, obj_fn, pareto_weights);
    } else {
        ndata0 = read_data(*bopti, file_path);
    }
    std::cout << "Number of data points: " << ndata0 << std::endl;
    
    // STEP 2: initialize function approximator and optimizer
    const int n_dim = 5;            // number of optimization variables
    const bool validate  = false;   // validation toggle

    // itialize GP model
    GaussianProcess model = GaussianProcess(kernel, file_path);
    
    // load model, n opt vars, constraints, settings, and file path into optimizer
    BayesianOpt optimizer(model, n_dim, c, s, file_path);
    optimizer.load_data(*bopti, validate);  // (bopti, _validate)

    // STEP 3: train the model
    const bool pre_learned = false;
    optimizer.condition_model(pre_learned);
    if (validate) {optimizer.evaluate_model();};

    // STEP 4: optimize and evaluate new candidate simulations
    optimizer.optimize(obj_fn, pareto_weights);

    // Get the current time after the code segment finishes
    auto end = std::chrono::high_resolution_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration = t.count() / 1e6;
    std::ofstream time_file;
    time_file.open(file_path + "/time_info.txt");
    time_file << "---- Optimization time: " << duration / 60 << " min ----" << std::endl;
    std::cout << "\n---Time taken by code segment: "
              << duration  / 60
              << " min---" << std::endl;

    delete bopti;
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
