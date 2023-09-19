#include "GaussianProcess.h"
#include "BayesianOpt.h"
#include "Voxel.h"
#include "helper_functions.h"
#include "common.h"


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

    // if available, define model parameters: length, sigma variance, noise variance
    bool pre_learned = true; 
    bool validate    = true; 

    // optimization constraints (default) and simulation settings (default)
    constraints c; 
    sim         s;
    s.bootstrap = false;
    s.time_stepping = 0;
    s.updateTimeSteppingValues();

    // set file path
    std::string file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_opt/output_" + std::to_string(s.time_stepping);   // MACBOOK PRO
    // std::string file_path = "/home/brian/Documents/berkeley/ugap_opt/output_" + std::to_string(s.time_stepping);         // LINUX CENTRAL COMPUTING
    std::cout << "--- THE ADVENTURE BEGINS ---" << std::endl;
    std::cout << "num cores: "  << omp_get_num_procs() << std::endl;
    std::cout << " file_path: " << file_path           << std::endl;

    // https://stackoverflow.com/questions/8036474/when-vectors-are-allocated-do-they-use-memory-on-the-heap-or-the-stack
    std::vector<bopt> *bopti = new std::vector<bopt>; // stores all info (header + elements) on heap

    // STEP 1: retrieve data set
    int ndata0;
    bool multi_thread = true; 
    if (s.bootstrap){
        ndata0 = 1000; 
        bootstrap(s, c, bopti, ndata0, file_path, multi_thread);
        
        // store data
        store_tot_data(bopti, s, ndata0, file_path);
    }else{
        ndata0 = read_data(bopti, file_path);
    }


    // convert data to Eigen matrices
    Eigen::MatrixXd* x_train     = new Eigen::MatrixXd;
    Eigen::VectorXd* y_train     = new Eigen::VectorXd;
    Eigen::VectorXd* y_train_std = new Eigen::VectorXd;

    Eigen::MatrixXd* x_test      = new Eigen::MatrixXd; 
    Eigen::VectorXd* y_test      = new Eigen::VectorXd;
    Eigen::VectorXd* y_test_std  = new Eigen::VectorXd;

    // split and move data from bopti to corresponding matrices
    build_dataset(bopti, x_train, y_train, x_test, y_test);
    
    // set up gaussian process
    GaussianProcess model = GaussianProcess("RBF", file_path); 
    
    // // pre-learned parameters
    std::vector<double> model_param;
    train_prior(model,
                *x_train, 
                *y_train, 
                model_param, 
                s.time_stepping, 
                pre_learned);
    
    // validate or predict
    evaluate_model(model, *x_test, *y_test, validate);
    
    // step 2: evaluate uniform points across domain
    int num_test = 25;
    Eigen::MatrixXd *x_sample      = new Eigen::MatrixXd(num_test, 5);
    Eigen::VectorXd *y_sample_mean = new Eigen::VectorXd(num_test);
    Eigen::VectorXd *y_sample_std  = new Eigen::VectorXd(num_test); 
    Eigen::VectorXd *conf_bound    = new Eigen::VectorXd(num_test);

    sample_posterior(model, 
                     *x_sample, 
                     *y_sample_mean, 
                     *y_sample_std, 
                     c); 
    acq_ucb(model, 
            *x_sample, 
            *y_sample_mean, 
            *y_sample_std,
            *conf_bound,  
            false);


    
    // evaluate voxel simulations using all  
    std::cout << "--- running new evaluations ---" << std::endl;
    int num_evals = omp_get_num_procs();
    std::vector<bopt> voxels;
    #pragma omp parallel for
    for (int id = 0; id < num_evals; ++id){
        bopt b; 
        b.temp = x_sample->coeff(id, 0);
        b.rp   = x_sample->coeff(id, 1);
        b.vp   = x_sample->coeff(id, 2); 
        b.uvi  = x_sample->coeff(id, 3);
        b.uvt  = x_sample->coeff(id, 4);

        // perform simulation with top candidates
        Voxel voxel_sim(s.tfinal, s.dt, s.node, id, b.temp, b.uvi, b.uvt, file_path, true);

        voxel_sim.ComputeParticles(b.rp, b.vp); 
        voxel_sim.Simulate(s.method, s.save_voxel); 
        b.obj  = voxel_sim.obj; 

        // write individual data to file (prevent accidental loss of data if stopped early)
        write_to_file(b, s, id, file_path);
        
        #pragma omp critical
        {
            int thread_id = omp_get_thread_num();
            voxels.push_back(b); 
            std::cout << "Thread " << thread_id << ": i = " << id << std::endl;
        }
    }

    std::cout << "--- finished new evaluations ---" << std::endl;
    for (int i = 0; i < num_evals; ++i){
        std::cout << "voxel " << i << ": " << voxels[i].obj << std::endl;
    }

    // Eigen::MatrixXd *x_eval = new Eigen::MatrixXd(num_evals, 5);
    // Eigen::VectorXd *y_eval = new Eigen::VectorXd(num_evals, 1);

    
    

    
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

    delete x_train;
    delete y_train; 
    delete y_train_std;
    
    delete x_test; 
    delete y_test; 
    delete y_test_std;

    delete x_sample; 
    delete y_sample_mean; 
    delete y_sample_std; 
    
    // delete x_eval;
    // delete y_eval;
    delete conf_bound;

    delete bopti;

    // Get the current time after the code segment finishes
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration of the code segment in minutes
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count() / 1e6;
    std::cout << "\n---Time taken by code segment: " << duration  / 60 << " min---" << std::endl;
    
    std::cout << "\nHello World!" << std::endl;

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
