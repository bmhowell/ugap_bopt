#include "Voxel.h"
#include "BayesianOpt.h"
#include "GaussianProcess.h"
#include "common.h"
#include <cmath>

/* default constructor */
BayesianOpt::BayesianOpt() {
    // default constructor
    constraints _c;
    sim _s; 
    _s.time_stepping = 0;
    _s.update_time_stepping_values();

    _file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_opt/output_" + std::to_string(_s.time_stepping);   // MACBOOK PRO
    // _file_path = "/home/brian/Documents/berkeley/ugap_opt/output_" + std::to_string(_s.time_stepping);         // LINUX CENTRAL COMPUTING

    GaussianProcess _model = GaussianProcess("RBF", _file_path); 
}

/* overload constructor */
BayesianOpt::BayesianOpt(GaussianProcess &_model,
                         int             &_n_dim,
                         constraints     &_c, 
                         sim             &_s, 
                         std::string     &_file_path) {
    // overload constructor
    _n_dim     = _n_dim;
    _c         = _c; 
    _s         = _s; 
    _model     = _model;
    _file_path = _file_path;
}

/* destructor */
BayesianOpt::~BayesianOpt() {
    std::cout << "\n--- BayesianOpt Destroyed ---\n" << std::endl;
    // destructor

    // delete data matrices
    delete x_train; 
    delete y_train; 
    delete y_train_std; 

    delete x_val; 
    delete y_val; 

    delete x_test; 
    delete y_test; 
    delete y_test_std; 

    // sampling
    delete x_sample;
    delete y_sample_mean;
    delete y_sample_std;
    delete conf_bound;
}

// PRIVATE MEMBER FUNCTIONS
void BayesianOpt::build_dataset(std::vector<bopt> &BOPTI, 
                                Eigen::MatrixXd   &X_TRAIN,
                                Eigen::VectorXd   &Y_TRAIN) {
    // build dataset for training only
    int num_data = BOPTI.size();

    X_TRAIN.resize(num_data, 5);
    Y_TRAIN.resize(num_data);

    // populate training and validation sets
    for (int i = 0; i < num_data; ++i) {
        X_TRAIN(i, 0) = BOPTI[i].temp;
        X_TRAIN(i, 1) = BOPTI[i].rp;
        X_TRAIN(i, 2) = BOPTI[i].vp;
        X_TRAIN(i, 3) = BOPTI[i].uvi;
        X_TRAIN(i, 4) = BOPTI[i].uvt;
        Y_TRAIN(i)    = BOPTI[i].obj;
    }
}

void BayesianOpt::build_dataset(std::vector<bopt> &BOPTI,
                                Eigen::MatrixXd   &X_TRAIN, 
                                Eigen::VectorXd   &Y_TRAIN,
                                Eigen::MatrixXd   &X_VAL,   
                                Eigen::VectorXd   &Y_VAL, 
                                Eigen::MatrixXd   &X_TEST, 
                                Eigen::VectorXd   &Y_TEST){
    // build dataset for training and testing

    // split data into training and validation sets
    int num_data = BOPTI.size();
    int num_train = 0.8 * num_data;
    int num_val   = 0.1 * num_data;
    int num_test  = 0.1 * num_data; 
    std::cout << "\n================ build dataset ================" << std::endl;
    std::cout << "num_data: " << num_data << std::endl;
    std::cout << "num_train: " << num_train << std::endl;
    std::cout << "num_val: " << num_val << std::endl;
    std::cout << "===============================================\n" << std::endl;

    // resize X_TRAIN, Y_TRAIN, X_VAL, and Y_VAL
    X_TRAIN.resize(num_train, 5);
    Y_TRAIN.resize(num_train);
    X_VAL.resize(num_val, 5);
    Y_VAL.resize(num_val);
    X_TEST.resize(num_test, 5);
    Y_TEST.resize(num_test);
    
    // shuffle dataset
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(BOPTI.begin(), BOPTI.end(), g);


    // populate training sets
    for (int i = 0; i < num_train; ++i) {
        X_TRAIN(i, 0) = BOPTI[i].temp;
        X_TRAIN(i, 1) = BOPTI[i].rp;
        X_TRAIN(i, 2) = BOPTI[i].vp;
        X_TRAIN(i, 3) = BOPTI[i].uvi;
        X_TRAIN(i, 4) = BOPTI[i].uvt;
        Y_TRAIN(i)    = BOPTI[i].obj;
    }

    // populate validation sets
    for (int i = 0; i < num_val; ++i) {
        X_VAL(i, 0) = BOPTI[i + num_train].temp;
        X_VAL(i, 1) = BOPTI[i + num_train].rp;
        X_VAL(i, 2) = BOPTI[i + num_train].vp;
        X_VAL(i, 3) = BOPTI[i + num_train].uvi;
        X_VAL(i, 4) = BOPTI[i + num_train].uvt;
        Y_VAL(i)    = BOPTI[i + num_train].obj;
    }

    // populate test sets
    for (int i = 0; i < num_test; ++i) {
        X_TEST(i, 0) = BOPTI[i + num_train + num_val].temp;
        X_TEST(i, 1) = BOPTI[i + num_train + num_val].rp;
        X_TEST(i, 2) = BOPTI[i + num_train + num_val].vp;
        X_TEST(i, 3) = BOPTI[i + num_train + num_val].uvi;
        X_TEST(i, 4) = BOPTI[i + num_train + num_val].uvt;
        Y_TEST(i)    = BOPTI[i + num_train + num_val].obj;
    }

}

void BayesianOpt::gen_test_points(Eigen::MatrixXd &_x_sample){
    // initialize input variables
    std::random_device rd;                          // obtain a random number from hardware
    std::mt19937 gen(rd());                         // seed the generator
    std::uniform_real_distribution<> dis(0., 1.);   // define the range

    for (int ind = 0; ind < _x_sample.rows(); ++ind){
        _x_sample(ind, 0) = ((this->_c).max_temp - (this->_c).min_temp) * dis(gen) + (this->_c).min_temp; 
        _x_sample(ind, 1) = ((this->_c).max_rp   - (this->_c).min_rp)   * dis(gen) + (this->_c).min_rp;
        _x_sample(ind, 2) = ((this->_c).max_vp   - (this->_c).min_vp)   * dis(gen) + (this->_c).min_vp;
        _x_sample(ind, 3) = ((this->_c).max_uvi  - (this->_c).min_uvi)  * dis(gen) + (this->_c).min_uvi;
        _x_sample(ind, 4) = ((this->_c).max_uvt  - (this->_c).min_uvt)  * dis(gen) + (this->_c).min_uvt;

        
    }
}

void BayesianOpt::store_tot_data(std::vector<bopt> &BOPTI, int num_sims){
    std::cout << "\n--- storing data ---\n" << std::endl;
    std::ofstream my_file; 
    my_file.open(this->_file_path + "/tot_bopt.dat");
    my_file << "temp, rp, vp, uvi, uvt, obj, tn" << std::endl;
    
    for (int id = 0; id < num_sims; ++id){
        my_file << BOPTI[id].temp << ", " 
                << BOPTI[id].rp   << ", " 
                << BOPTI[id].vp   << ", " 
                << BOPTI[id].uvi  << ", " 
                << BOPTI[id].uvt  << ", " 
                << BOPTI[id].obj  << ", " 
                << _s.time_stepping << std::endl;
    }
    my_file.close();
}

// PUBLIC MEMBER FUNCTIONS
void BayesianOpt::load_data(std::vector<bopt> &BOPTI, bool validate) {
    this->_bopti    = BOPTI;
    this->_validate = validate;

    if (validate){
        // build dataset for training and testing
        build_dataset(BOPTI, *x_train, *y_train, *x_val, *y_val, *x_test, *y_test);
    }else{
        // build dataset for training only
        build_dataset(BOPTI, *x_train, *y_train); 
    }
}

void BayesianOpt::condition_model(bool pre_learned){
    // ADD ASSERTION TO CHECK IF DATA IS LOADED
    if (pre_learned){
        switch (_s.time_stepping){
            case 0: 
                _model_params = {0.99439,0.356547,0.000751229};   // obj_0 -> 673.344
                break;
            case 1: 
                _model_params = {0.994256,0.623914,0.000965578};  // obj_1 -> 422.003 
                break;
            case 2:
                _model_params = {0.940565,0.708302,0.000328992};  // obj_2 -> 397.977
                break;
            case 3: 
                _model_params = {0.956662, 0.78564, 0.00095118};  // obj_3 -> 487.76 
                break; 
        }

        // train with maximized log likelihood
        this->_model.train(*x_train, *y_train, *x_val, *y_val, _model_params);
        double cost = this->_model.validate();
        std::cout << "Validation Cost: " << cost << std::endl;

    }else{
        condition_model(); 
    }
}

void BayesianOpt::condition_model(){
    // train with maximized log likelihood
    if (this->_validate){
        this->_model.train(*x_train, *y_train, *x_val, *y_val);
    }; 
    this->_model.train(*x_train, *y_train);
}

void BayesianOpt::evaluate_model(){
    // ADD ASSERTION TO CHECK IF DATA IS LOADED
    
    if (_validate){
        std::cout << "--- _model validatation ---" << std::endl; 
        _model.validate();
    }
    else{
        std::cout << "--- _model prediction ---" << std::endl; 
        _model.predict(*x_test, false);
    }
}

void BayesianOpt::sample_posterior(){

    x_sample      ->resize(this->_num_sample, this->_n_dim);
    y_sample_mean ->resize(this->_num_sample);
    y_sample_std  ->resize(this->_num_sample);
    conf_bound    ->resize(this->_num_sample);

    gen_test_points(*x_sample);

    this->_model.predict(*x_sample, false);

    this->y_sample_mean->array() = this->_model.get_y_test().array();
    this->y_sample_std->array()  = this->_model.get_y_test_std().array();

}

void BayesianOpt::qUCB(bool _lcb){
    if (_lcb){
        this->conf_bound->array() = this->y_sample_mean->array() - 1.96 * this->y_sample_std->array();

        // sort conf_bound
        Eigen::VectorXi sorted_inds = Eigen::VectorXi::LinSpaced(this->conf_bound->size(), 0, this->conf_bound->size() - 1);

        std::sort(sorted_inds.data(), sorted_inds.data() + sorted_inds.size(),
                [this](int a, int b) { return (*this->conf_bound)(a) < (*this->conf_bound)(b); });

        *conf_bound = (*conf_bound)(sorted_inds);
        *x_sample   = (*x_sample)(sorted_inds, Eigen::all);
    }
    else{
        qUCB();
    }
}

void BayesianOpt::qUCB(){
    this->conf_bound->array() = this->y_sample_mean->array() - 1.96 * this->y_sample_std->array();

    // sort conf_bound
    Eigen::VectorXi sorted_inds = Eigen::VectorXi::LinSpaced(this->conf_bound->size(), 0, this->conf_bound->size() - 1);

    std::sort(sorted_inds.data(), sorted_inds.data() + sorted_inds.size(),
            [this](int a, int b) { return (*this->conf_bound)(a) > (*this->conf_bound)(b); });

    *conf_bound = (*conf_bound)(sorted_inds);
    *x_sample   = (*x_sample)(sorted_inds, Eigen::all);

}

void BayesianOpt::evaluate_samples(){
    this->_num_evals = omp_get_num_procs();
    std::vector<bopt> voxels_evals;
    #pragma omp parallel for
    for (int id = 0; id < this->_num_evals; ++id){

        bopt b; 
        b.temp = x_sample->coeff(id, 0);
        b.rp   = x_sample->coeff(id, 1);
        b.vp   = x_sample->coeff(id, 2);
        b.uvi  = x_sample->coeff(id, 3);
        b.uvt  = x_sample->coeff(id, 4);
        
        // perform simulation with top candidates
        Voxel voxel_sim(_s.tfinal,
                        _s.dt, 
                        _s.node, 
                        id, 
                        b.temp, 
                        b.uvi, 
                        b.uvt, 
                        this->_file_path, 
                        _s.save_voxel);

        voxel_sim.ComputeParticles(b.rp, 
                                   b.vp);
        
        b.obj = voxel_sim.obj; 

        #pragma omp critical
        {
            int thread_id = omp_get_thread_num();
            voxels_evals.push_back(b);
            std::cout << "Thread " << thread_id << ": i = " << id << std::endl;
        }
    }

    std::cout << "--- finished new evaluations ---" << std::endl;

    // concatenate data
    this->_bopti.insert(this->_bopti.end(), voxels_evals.begin(), voxels_evals.end());
    store_tot_data(this->_bopti, this->_bopti.size() + this->_num_evals);


}

void BayesianOpt::optimize(){

}

