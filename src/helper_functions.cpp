#include "common.h"
#include "GaussianProcess.h"
#include "Voxel.h"
#include "helper_functions.h"  // Include the header file


// Generate data
double gen_data(float tfinal, 
                double dt, 
                int node, 
                int idsim, 
                bopt& bopti, 
                sim& simi, 
                std::string file_path, 
                bool multi_thread) {

    // // objective function value
    // float obj  = bopti.obj;

    if (!multi_thread){
        std::cout << "================ begin simulation ================" << std::endl;
        std::cout << "id sim: " << idsim       << std::endl;
        std::cout << "temp: "   << bopti.temp  << std::endl;
        std::cout << "rp: "     << bopti.rp    << std::endl;
        std::cout << "vp: "     << bopti.vp    << std::endl;
        std::cout << "uvi: "    << bopti.uvi   << std::endl;
        std::cout << "uvt: "    << bopti.uvt   << std::endl;
        std::cout                              << std::endl;
    }
    
    // run simulation
    auto start = std::chrono::high_resolution_clock::now();
    Voxel VoxelSystem1( tfinal, 
                        dt, 
                        node, 
                        idsim, 
                        bopti.temp,  
                        bopti.uvi, 
                        bopti.uvt, 
                        file_path, 
                        multi_thread);

    VoxelSystem1.ComputeParticles(bopti.rp, bopti.vp);
    if (simi.save_density == 1){
        VoxelSystem1.Density2File();
    }

    VoxelSystem1.Simulate(simi.method, simi.save_voxel);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count() / 1e6;

    if (!multi_thread){
        std::cout << " --- Simulation time: " << duration / 60 << "min ---" << std::endl;
        std::cout << "testing obj: " << VoxelSystem1.obj << std::endl;
    }else{
        std::cout << "---sim " << idsim << " complete ----" << std::endl;
    }
    return VoxelSystem1.obj; 
}

// initialize input variables
void bootstrap(sim &sim_settings, 
               constraints &c, 
               std::vector<bopt> *bopti, 
               int num_sims, 
               std::string file_path, 
               bool multi_thread) {

    // generate random values
    if (multi_thread){
        // initialize input variables
        std::vector<std::mt19937> gens(num_sims);  // Create an array of generators
        std::vector<std::uniform_real_distribution<double>> distributions(num_sims);  // Create an array of distributions

        // Seed each generator
        for (int id = 0; id < num_sims; ++id) {
            std::random_device rd;
            gens[id].seed(rd());
            distributions[id] = std::uniform_real_distribution<double>(0.0, 1.0);
        }

        #pragma omp parallel for
        for (int id = 0; id < num_sims; ++id) {
            bopt b; 
            b.temp = (c.max_temp - c.min_temp) * distributions[id](gens[id]) +  c.min_temp;
            b.rp   = (c.max_rp   - c.min_rp)   * distributions[id](gens[id]) +  c.min_rp;
            b.vp   = (c.max_vp   - c.min_vp)   * distributions[id](gens[id]) +  c.min_vp;
            b.uvi  = (c.max_uvi  - c.min_uvi)  * distributions[id](gens[id]) +  c.min_uvi;
            b.uvt  = (c.max_uvt  - c.min_uvt)  * distributions[id](gens[id]) +  c.min_uvt;

            // peform simulation with randomly generatored values
            b.obj = gen_data(sim_settings.tfinal, sim_settings.dt, sim_settings.node, id, b, sim_settings, file_path, true);
            std::cout << "b.obj: " << b.obj << std::endl;
            std::cout << "---  ------- ---\n" << std::endl;

            // write individual data to file (prevent accidental loss of data if stopped early)
            write_to_file(b, sim_settings, id, file_path); 
            #pragma omp critical
            {
                int thread_id = omp_get_thread_num();
                bopti->push_back(b);
                std::cout << "Thread " << thread_id << ": i = " << id << std::endl;
            }
        }
    }else{
        // initialize input variables
        std::random_device rd;                                          // Obtain a random seed from the hardware
        std::mt19937 gen(rd());                                         // Seed the random number generator
        std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)
        for (int id = 0; id < num_sims; ++id) {
            bopt b; 
            b.temp = (c.max_temp - c.min_temp) * distribution(gen) +  c.min_temp;
            b.rp   = (c.max_rp   - c.min_rp)   * distribution(gen) +  c.min_rp;
            b.vp   = (c.max_vp   - c.min_vp)   * distribution(gen) +  c.min_vp;
            b.uvi  = (c.max_uvi  - c.min_uvi)  * distribution(gen) +  c.min_uvi;
            b.uvt  = (c.max_uvt  - c.min_uvt)  * distribution(gen) +  c.min_uvt;

            // peform simulation with randomly generatored values
            b.obj = gen_data(sim_settings.tfinal, sim_settings.dt, sim_settings.node, id, b, sim_settings, file_path, false);
            std::cout << "b.obj: " << b.obj << std::endl;
            std::cout << "---  ------- ---\n" << std::endl;

            // write individual data to file (prevent accidental loss of data if stopped early)
            write_to_file(b, sim_settings, id, file_path); 

            bopti->push_back(b); 
        }
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

void build_dataset(std::vector<bopt> &_bopti,
                   Eigen::MatrixXd   &_x_train, Eigen::VectorXd &_y_train,
                   Eigen::MatrixXd   &_x_val,   Eigen::VectorXd &_y_val){

    // split data into training and validation sets
    int num_data = _bopti.size();
    int num_train = 0.9 * num_data;
    int num_val   = 0.1 * num_data;
    std::cout << "\n================ build dataset ================" << std::endl;
    std::cout << "num_data: " << num_data << std::endl;
    std::cout << "num_train: " << num_train << std::endl;
    std::cout << "num_val: " << num_val << std::endl;
    std::cout << "===============================================\n" << std::endl;

    // resize _x_train, _y_train, _x_val, and Y_VAL
    _x_train.resize(num_train, 5);
    _y_train.resize(num_train);
    _x_val.resize(num_val, 5);
    _y_val.resize(num_val);
    
    // shuffle dataset
    std::random_device rd;
    // std::mt19937 g(rd());
    std::mt19937 g(47);
    std::shuffle(_bopti.begin(), _bopti.end(), g);

    // initialize training and validation sets
    _x_train = Eigen::MatrixXd(num_train, 5);
    _y_train = Eigen::VectorXd(num_train);
    _x_val   = Eigen::MatrixXd(num_val, 5);
    _y_val   = Eigen::VectorXd(num_val);

    // populate training and validation sets
    for (int i = 0; i < num_train; ++i) {
        _x_train(i, 0) = _bopti[i].temp;
        _x_train(i, 1) = _bopti[i].rp;
        _x_train(i, 2) = _bopti[i].vp;
        _x_train(i, 3) = _bopti[i].uvi;
        _x_train(i, 4) = _bopti[i].uvt;
        _y_train(i)    = _bopti[i].obj;
    }

    for (int i = 0; i < num_val; ++i) {
        _x_val(i, 0) = _bopti[i + num_train].temp;
        _x_val(i, 1) = _bopti[i + num_train].rp;
        _x_val(i, 2) = _bopti[i + num_train].vp;
        _x_val(i, 3) = _bopti[i + num_train].uvi;
        _x_val(i, 4) = _bopti[i + num_train].uvt;
        _y_val(i)    = _bopti[i + num_train].obj;
    }

}


void build_dataset(std::vector<bopt> &_bopti,
                   Eigen::MatrixXd   &_x_train, 
                   Eigen::VectorXd   &_y_train){

    int num_data = _bopti.size();

    _x_train.resize(num_data, 5);
    _y_train.resize(num_data);

    // populate training and validation sets
    for (int i = 0; i < num_data; ++i) {
        _x_train(i, 0) = _bopti[i].temp;
        _x_train(i, 1) = _bopti[i].rp;
        _x_train(i, 2) = _bopti[i].vp;
        _x_train(i, 3) = _bopti[i].uvi;
        _x_train(i, 4) = _bopti[i].uvt;
        _y_train(i)    = _bopti[i].obj;
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


void train_prior(GaussianProcess &MODEL, 
                 Eigen::MatrixXd &X_TRAIN, 
                 Eigen::VectorXd &Y_TRAIN,
                 std::vector<double> &M_PARAMS, 
                 int TIME_STEPPING, 
                 bool pre_learned){
    if (pre_learned){
        switch (TIME_STEPPING){
            case 0: 
                M_PARAMS = {0.99439,0.356547,0.000751229};   // obj_0 -> 673.344
                break;
            case 1: 
                M_PARAMS = {0.994256,0.623914,0.000965578};  // obj_1 -> 422.003 
                break;
            case 2:
                M_PARAMS = {0.940565,0.708302,0.000328992};  // obj_2 -> 397.977
                break;
            case 3: 
                M_PARAMS = {0.956662, 0.78564, 0.00095118};  // obj_3 -> 487.76 
                break; 
        }
        MODEL.train(X_TRAIN, Y_TRAIN, M_PARAMS);
    }
    else{
        MODEL.train(X_TRAIN, Y_TRAIN);
    }
}


void evaluate_model(GaussianProcess &MODEL, 
                    Eigen::MatrixXd &X_TEST,
                    Eigen::VectorXd &Y_TEST, 
                    bool VALIDATE){

    if (VALIDATE){
        std::cout << "--- model validation ---\n" << std::endl;
        MODEL.validate(X_TEST, Y_TEST);
        std::cout << "--- ---------------- ---\n" << std::endl;
    }
    else{
        std::cout << "--- model prediction ---\n" << std::endl;
        MODEL.predict(X_TEST, false);
        std::cout << "--- ---------------- ---\n" << std::endl;
    }
}

void sample_posterior(GaussianProcess &MODEL, 
                      Eigen::MatrixXd &X_SAMPLE, 
                      Eigen::VectorXd &Y_SAMPLE_MEAN, 
                      Eigen::VectorXd &Y_SAMPLE_STD,
                      constraints &C){

    // generate test points
    gen_test_points(C, X_SAMPLE);
    MODEL.predict(X_SAMPLE, true);
    
    Y_SAMPLE_MEAN = MODEL.get_y_test();
    Y_SAMPLE_STD  = MODEL.get_y_test_std();

}

void acq_ucb(GaussianProcess &MODEL, 
             Eigen::MatrixXd &X_SAMPLE, 
             Eigen::VectorXd &Y_SAMPLE_MEAN, 
             Eigen::VectorXd &Y_SAMPLE_STD, 
             Eigen::VectorXd &CONF_BOUND,
             bool MAXIMIZE){
    if (MAXIMIZE){
        // UCB algorithm (LATER CHANGE 1.96 to decaying parameter)
        CONF_BOUND = Y_SAMPLE_MEAN + 1.96 * Y_SAMPLE_STD;

        Eigen::VectorXi sorted_inds = Eigen::VectorXi::LinSpaced(CONF_BOUND.size(), 0, CONF_BOUND.size() - 1);
        std::sort(sorted_inds.data(), sorted_inds.data() + sorted_inds.size(),
                  [&CONF_BOUND](int a, int b) { return CONF_BOUND(a) > CONF_BOUND(b); });

        CONF_BOUND = CONF_BOUND(sorted_inds);
        X_SAMPLE   = X_SAMPLE(sorted_inds, Eigen::all);
    }
    else{
        // LCB algorithm (LATER CHANGE 1.96 to decaying parameter)
        CONF_BOUND = Y_SAMPLE_MEAN - 1.96 * Y_SAMPLE_STD;

        Eigen::VectorXi sorted_inds = Eigen::VectorXi::LinSpaced(CONF_BOUND.size(), 0, CONF_BOUND.size() - 1);
        std::sort(sorted_inds.data(), sorted_inds.data() + sorted_inds.size(),
                  [&CONF_BOUND](int a, int b) { return CONF_BOUND(a) < CONF_BOUND(b); });

        CONF_BOUND = CONF_BOUND(sorted_inds);
        X_SAMPLE   = X_SAMPLE(sorted_inds, Eigen::all);
    }

    
}