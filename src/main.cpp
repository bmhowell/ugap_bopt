#include "GaussianProcess.h"
#include "BayesianOpt.h"
#include "Voxel.h"
#include "helper_functions.h"
#include "common.h"

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    ////////////////////////////  PARSE ARGS  ////////////////////////////
    // optimization constraints (default) and simulation settings (default)
    constraints c; 
    sim         s;
    s.bootstrap = false;
    s.time_stepping = 0;
    s.update_time_stepping_values();
    ////////////////////////////  //////////  ////////////////////////////
    // set file path
    std::string file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_opt/output_" + std::to_string(s.time_stepping);   // MACBOOK PRO
    // std::string file_path = "/home/brian/Documents/berkeley/ugap_opt/output_" + std::to_string(s.time_stepping);         // LINUX CENTRAL COMPUTING
    std::cout << "\n--- THE ADVENTURE BEGINS ---" << std::endl;

    // STEP 1: retrieve data set
    int ndata0;
    std::vector<bopt> *bopti = new std::vector<bopt>; // stores all info (header + elements) on heap
    bool multi_thread = true; 
    if (s.bootstrap){
        ndata0 = 1000; 
        bootstrap(s, c, bopti, ndata0, file_path, multi_thread);
        
        // store data
        store_tot_data(bopti, s, ndata0, file_path);
    }else{
        ndata0 = read_data(bopti, file_path);
    }

    // initialize function approximator and optimizer
    GaussianProcess model = GaussianProcess("RBF", file_path);
    BayesianOpt optimizer(model, ndata0, c, s, file_path);
    optimizer.load_data(*bopti, true);  // (bopti, _validate)

    // train the model (pre-learned)
    optimizer.condition_model(true);
    optimizer.evaluate_model();
    // optimizer.qUCB(false); 
    // optimizer.evaluate_samples();

    


    // Get the current time after the code segment finishes
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count() / 1e6;
    std::cout << "\n---Time taken by code segment: " << duration  / 60 << " min---" << std::endl;
    
    std::cout << "\nHello World!" << std::endl;
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
