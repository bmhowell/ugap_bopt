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
    // opt constraints (default) and sim settings (default)
    constraints c;
    sim         s;
    s.bootstrap = false;
    s.time_stepping = 0;
    s.update_time_stepping_values();

    // MACBOOK PRO
    std::string file_path;
    file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_opt/output_"
                + std::to_string(s.time_stepping);
    // LINUX CENTRAL COMPUTING
    // file_path = "/home/brian/Documents/berkeley/ugap_opt/output_"
    //           + std::to_string(s.time_stepping);

    std::cout << "\n--- INITIALIZING OPT. FRAMEWORK ---" << std::endl;

    // data storage
    std::vector<bopt> *bopti = new std::vector<bopt>;

    // STEP 1: retrieve data set
    int ndata0;
    bool multi_thread = true;
    if (s.bootstrap) {
        ndata0 = 1000;
        bootstrap(s, c, bopti, ndata0, file_path, multi_thread);

        // store data
        store_tot_data(bopti, s, ndata0, file_path);
    } else {
        ndata0 = read_data(bopti, file_path);
    }
    std::cout << "Number of data points: " << ndata0 << std::endl;
    // STEP 2: initialize function approximator and optimizer
    int n_dim = 5; 
    GaussianProcess model = GaussianProcess("RBF", file_path);
    BayesianOpt optimizer(model, n_dim, c, s, file_path);
    optimizer.load_data(*bopti, true);  // (bopti, _validate)

    // STEP 3: train the model
    optimizer.condition_model(true);   // (pre-learned)
    optimizer.evaluate_model();
    // optimizer.qUCB(false);             // false -> lcb || true -> ucb

    // STEP 4: optimize and evaluate new candidate simulations
    optimizer.optimize();

    // Get the current time after the code segment finishes
    auto end = std::chrono::high_resolution_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration = t.count() / 1e6;
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
