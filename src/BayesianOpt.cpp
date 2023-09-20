#include "BayesianOpt.h"
#include "GaussianProcess.h"
#include "common.h"
#include <cmath>

/* default constructor */
BayesianOpt::BayesianOpt() {
    // default constructor
    constraints c;
    sim s; 
    s.time_stepping = 0;
    s.update_time_stepping_values();

    file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_opt/output_" + std::to_string(s.time_stepping);   // MACBOOK PRO
    // file_path = "/home/brian/Documents/berkeley/ugap_opt/output_" + std::to_string(s.time_stepping);         // LINUX CENTRAL COMPUTING

    GaussianProcess model = GaussianProcess("RBF", file_path); 
}

/* overload constructor */
BayesianOpt::BayesianOpt(GaussianProcess &_model,
                         int             &_n_dim,
                         constraints     &_c, 
                         sim             &_s, 
                         std::string     &_file_path) {
    // overload constructor
    n_dim     = _n_dim;
    c         = _c; 
    s         = _s; 
    model     = _model;
    file_path = _file_path;
}

/* destructor */
BayesianOpt::~BayesianOpt() {
    std::cout << "\n--- BayesianOpt Destroyed ---\n" << std::endl;
    // destructor

    // delete data matrices
    delete x_train; 
    delete y_train; 
    delete y_train_std; 

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
void BayesianOpt::build_dataset(std::vector<bopt> &_bopti, 
                                Eigen::MatrixXd   &_x_train,
                                Eigen::VectorXd   &_y_train) {
    // build dataset for training only
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

void BayesianOpt::build_dataset(std::vector<bopt> &_bopti,
                                Eigen::MatrixXd   &_x_train, 
                                Eigen::VectorXd   &_y_train,
                                Eigen::MatrixXd   &_x_val,   
                                Eigen::VectorXd   &_y_val){
    // build dataset for training and testing

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

void BayesianOpt::gen_test_points(Eigen::MatrixXd &_x_sample){
    // initialize input variables
    std::random_device rd;                          // obtain a random number from hardware
    std::mt19937 gen(rd());                         // seed the generator
    std::uniform_real_distribution<> dis(0., 1.);   // define the range

    for (int ind = 0; ind < _x_sample.rows(); ++ind){
        _x_sample(ind, 0) = ((this->c).max_temp - (this->c).min_temp) * dis(gen) + (this->c).min_temp; 
        _x_sample(ind, 1) = ((this->c).max_rp   - (this->c).min_rp)   * dis(gen) + (this->c).min_rp;
        _x_sample(ind, 2) = ((this->c).max_vp   - (this->c).min_vp)   * dis(gen) + (this->c).min_vp;
        _x_sample(ind, 3) = ((this->c).max_uvi  - (this->c).min_uvi)  * dis(gen) + (this->c).min_uvi;
        _x_sample(ind, 4) = ((this->c).max_uvt  - (this->c).min_uvt)  * dis(gen) + (this->c).min_uvt;

        
    }
}

void BayesianOpt::store_tot_data(std::vector<bopt> &_bopti, int num_sims){
    std::cout << "\n--- storing data ---\n" << std::endl;
    std::ofstream my_file; 
    my_file.open(this->file_path + "/tot_bopt.dat");
    my_file << "temp, rp, vp, uvi, uvt, obj, tn" << std::endl;
    
    for (int id = 0; id < num_sims; ++id){
        my_file << _bopti[id].temp << ", " 
                << _bopti[id].rp   << ", " 
                << _bopti[id].vp   << ", " 
                << _bopti[id].uvi  << ", " 
                << _bopti[id].uvt  << ", " 
                << _bopti[id].obj  << ", " 
                << s.time_stepping << std::endl;
    }
    my_file.close();
}

// PUBLIC MEMBER FUNCTIONS
void BayesianOpt::load_data(std::vector<bopt> &_bopti, bool _validate) {
    this->bopti    = _bopti;
    this->validate = _validate;

    if (_validate){
        // build dataset for training and testing
        build_dataset(_bopti, *x_train, *y_train, *x_test, *y_test);  
    }else{
        // build dataset for training only
        build_dataset(_bopti, *x_train, *y_train);                      
    }
}

void BayesianOpt::condition_model(bool pre_learned){
    // ADD ASSERTION TO CHECK IF DATA IS LOADED
    if (pre_learned){
        switch (s.time_stepping){
            case 0: 
                model_params = {0.99439,0.356547,0.000751229};   // obj_0 -> 673.344
                break;
            case 1: 
                model_params = {0.994256,0.623914,0.000965578};  // obj_1 -> 422.003 
                break;
            case 2:
                model_params = {0.940565,0.708302,0.000328992};  // obj_2 -> 397.977
                break;
            case 3: 
                model_params = {0.956662, 0.78564, 0.00095118};  // obj_3 -> 487.76 
                break; 
        }
    }else{
        condition_model(); 
    }

    // train with maximized log likelihood
    this->model.train(*x_train, *y_train, model_params);
}

void BayesianOpt::condition_model(){
    // ADD ASSERTION TO CHECK IF DATA IS LOADED
    // train with maximized log likelihood
    this->model.train(*x_train, *y_train);
}

void BayesianOpt::evaluate_model(){
    // ADD ASSERTION TO CHECK IF DATA IS LOADED
    
    if (validate){
        std::cout << "--- model validatation ---" << std::endl; 
        model.validate(*x_test, *y_test);
    }
    else{
        std::cout << "--- model prediction ---" << std::endl; 
        model.predict(*x_test, false);
    }
}

void BayesianOpt::sample_posterior(){

    x_sample      ->resize(this->num_sample, this->n_dim);
    y_sample_mean ->resize(this->num_sample);
    y_sample_std  ->resize(this->num_sample);
    conf_bound    ->resize(this->num_sample);

    gen_test_points(*x_sample);

    this->model.predict(*x_sample, false);

    this->y_sample_mean->array() = this->model.get_y_test().array();
    this->y_sample_std->array()  = this->model.get_y_test_std().array();

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
    this->num_evals = omp_get_num_procs();
    std::vector<bopt> voxels_evals;
    #pragma omp parallel for
    for (int id = 0; id < this->num_evals; ++id){
        bopt b; 
        b.temp = x_sample->coeff(id, 0);
        b.rp   = x_sample->coeff(id, 1);
        b.vp   = x_sample->coeff(id, 2);
        b.uvi  = x_sample->coeff(id, 3);
        b.uvt  = x_sample->coeff(id, 4);
        
        // perform simulation with top candidates
        Voxel voxel_sim(s.tfinal,
                        s.dt, 
                        s.node, 
                        id, 
                        b.temp, 
                        b.uvi, 
                        b.uvt, 
                        this->file_path, 
                        s.save_voxel);

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
    this->bopti.insert(this->bopti.end(), voxels_evals.begin(), voxels_evals.end());
    store_tot_data(this->bopti, this->bopti.size() + this->num_evals);


}

void BayesianOpt::optimize(){

}

