#include "GaussianProcess.h"
#include "Voxel.h"
#include "helper_functions.h"


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
    s.bootstrap = 0;
    s.time_stepping = 2;
    s.updateTimeSteppingValues();

    // set file path
    std::string file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_opt/output_" + std::to_string(s.time_stepping);   // MACBOOK PRO
    std::cout << " file_path: " << file_path << std::endl;
    // std::string file_path = "/home/brian/Documents/berkeley/ugap_opt/output_" + std::to_string(s.time_stepping);         // LINUX CENTRAL COMPUTING

    // https://stackoverflow.com/questions/8036474/when-vectors-are-allocated-do-they-use-memory-on-the-heap-or-the-stack
    std::vector<bopt> *bopti = new std::vector<bopt>; // stores all info (header + elements) on heap

    // STEP 1: retrieve data set
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
    Eigen::MatrixXd* x_train     = new Eigen::MatrixXd;
    Eigen::VectorXd* y_train     = new Eigen::VectorXd;
    Eigen::VectorXd* y_train_std = new Eigen::VectorXd;

    Eigen::MatrixXd* x_test      = new Eigen::MatrixXd; 
    Eigen::VectorXd* y_test      = new Eigen::VectorXd;
    Eigen::VectorXd* y_test_std  = new Eigen::VectorXd;

    Eigen::MatrixXd* x_total     = new Eigen::MatrixXd;
    Eigen::VectorXd* y_total     = new Eigen::VectorXd;

    // split and move data from bopti to corresponding matrices
    build_dataset(bopti, x_train, y_train, x_test, y_test);
    
    // set up gaussian process
    GaussianProcess model = GaussianProcess("RBF", file_path); 
    
    // // pre-learned parameters
    std::vector<double> model_param;
    switch (s.time_stepping){
        case 0: 
            model_param = {0.99439,0.356547,0.000751229};   // obj_0 -> 673.344
            break;
        case 1: 
            model_param = {0.994256,0.623914,0.000965578};  // obj_1 -> 422.003 
            break;
        case 2:
            model_param = {0.940565,0.708302,0.000328992};  // obj_2 -> 397.977
            break;
        case 3: 
            model_param = {0.956662, 0.78564, 0.00095118};  // obj_3 -> 487.76 
            break; 
    }

    if (pre_learned){
        model.train(*x_train, *y_train, model_param);
    }else{
        model.train(*x_train, *y_train);
    }
    
    // validate or predict
    if (validate){
        model.validate(*x_test, *y_test);
    }else{ 
        model.predict(*x_test, false);
    }
    
    // step 2: evaluate uniform points across domain
    int num_test = 25;
    *x_test  = Eigen::MatrixXd(num_test, 5);
    gen_test_points(c, *x_test);
    model.predict(*x_test, true);
    *y_test_std = model.get_y_test_std();
    Eigen::VectorXd y_mean = model.get_y_test();
    std::cout << "\ny_mean: \n" << y_mean.transpose().head(5) << std::endl;
    std::cout << " # elements in y_mean: " << y_mean.size() << std::endl;
    std::cout << "\ny_test_std: \n" << y_test_std->transpose().head(5) << std::endl;

    
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
    delete x_test; 
    delete y_train; 
    delete y_test; 
    delete y_test_std;
    delete y_train_std;
    delete x_total;
    delete y_total;
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
