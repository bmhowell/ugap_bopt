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

    // initialize data matrices
    _x_train       = new Eigen::MatrixXd;
    _y_train       = new Eigen::VectorXd;
    _y_train_std   = new Eigen::VectorXd;

    _x_val         = new Eigen::MatrixXd;
    _y_val         = new Eigen::VectorXd;

    _x_test        = new Eigen::MatrixXd;
    _y_test        = new Eigen::VectorXd;
    _y_test_std    = new Eigen::VectorXd;

    // sampling
    _x_sample      = new Eigen::MatrixXd;
    _y_sample_mean = new Eigen::VectorXd;
    _y_sample_std  = new Eigen::VectorXd;
    _conf_bound    = new Eigen::VectorXd;


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

    // initialize data matrices
    _x_train       = new Eigen::MatrixXd;
    _y_train       = new Eigen::VectorXd;
    _y_train_std   = new Eigen::VectorXd;

    _x_val         = new Eigen::MatrixXd;
    _y_val         = new Eigen::VectorXd;

    _x_test        = new Eigen::MatrixXd;
    _y_test        = new Eigen::VectorXd;
    _y_test_std    = new Eigen::VectorXd;

    // sampling
    _x_sample      = new Eigen::MatrixXd;
    _y_sample_mean = new Eigen::VectorXd;
    _y_sample_std  = new Eigen::VectorXd;
    _conf_bound    = new Eigen::VectorXd;
}

/* destructor */
BayesianOpt::~BayesianOpt() {
    std::cout << "\n--- BayesianOpt Destroyed ---\n" << std::endl;
    // destructor

    // delete data matrices
    delete _x_train; 
    delete _y_train; 
    delete _y_train_std; 

    delete _x_val; 
    delete _y_val; 

    delete _x_test; 
    delete _y_test; 
    delete _y_test_std; 

    // sampling
    delete _x_sample;
    delete _y_sample_mean;
    delete _y_sample_std;
    delete _conf_bound;
}

// PRIVATE MEMBER FUNCTIONS
void BayesianOpt::build_dataset(std::vector<bopt> &bopti, 
                                Eigen::MatrixXd   &x_train,
                                Eigen::VectorXd   &y_train) {
    // build dataset for training only
    int num_data = bopti.size();

    x_train.resize(num_data, 5);
    y_train.resize(num_data);

    // populate training and validation sets
    for (int i = 0; i < num_data; ++i) {
        x_train(i, 0) = bopti[i].temp;
        x_train(i, 1) = bopti[i].rp;
        x_train(i, 2) = bopti[i].vp;
        x_train(i, 3) = bopti[i].uvi;
        x_train(i, 4) = bopti[i].uvt;
        y_train(i)    = bopti[i].obj;
    }
}

void BayesianOpt::build_dataset(std::vector<bopt> &bopti,
                                Eigen::MatrixXd   &x_train, 
                                Eigen::VectorXd   &y_train,
                                Eigen::MatrixXd   &x_val,   
                                Eigen::VectorXd   &y_val){
    // build dataset for training and testing

    // split data into training and validation sets
    int num_data = bopti.size();
    int num_train = 0.8 * num_data;
    int num_val   = 0.1 * num_data;
    
    std::cout << "\n================ build dataset ================" << std::endl;
    std::cout << "num_data: " << num_data << std::endl;
    std::cout << "num_train: " << num_train << std::endl;
    std::cout << "num_val: " << num_val << std::endl;
    std::cout << "===============================================\n" << std::endl;

    // resize x_train, y_train, x_val, and Y_VAL
    x_train.resize(num_train, 5);
    y_train.resize(num_train);
    x_val.resize(num_val, 5);
    y_val.resize(num_val);
    
    // shuffle dataset
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(bopti.begin(), bopti.end(), g);


    // populate training sets
    for (int i = 0; i < num_train; ++i) {
        x_train(i, 0) = bopti[i].temp;
        x_train(i, 1) = bopti[i].rp;
        x_train(i, 2) = bopti[i].vp;
        x_train(i, 3) = bopti[i].uvi;
        x_train(i, 4) = bopti[i].uvt;
        y_train(i)    = bopti[i].obj;
    }

    // populate validation sets
    for (int i = 0; i < num_val; ++i) {
        x_val(i, 0) = bopti[i + num_train].temp;
        x_val(i, 1) = bopti[i + num_train].rp;
        x_val(i, 2) = bopti[i + num_train].vp;
        x_val(i, 3) = bopti[i + num_train].uvi;
        x_val(i, 4) = bopti[i + num_train].uvt;
        y_val(i)    = bopti[i + num_train].obj;
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

void BayesianOpt::store_tot_data(std::vector<bopt> &bopti, int num_sims){
    std::cout << "\n--- storing data ---\n" << std::endl;
    std::ofstream my_file; 
    my_file.open(this->_file_path + "/tot_bopt.dat");
    my_file << "temp, rp, vp, uvi, uvt, obj, tn" << std::endl;
    
    for (int id = 0; id < num_sims; ++id){
        my_file << bopti[id].temp << ", " 
                << bopti[id].rp   << ", " 
                << bopti[id].vp   << ", " 
                << bopti[id].uvi  << ", " 
                << bopti[id].uvt  << ", " 
                << bopti[id].obj  << ", " 
                << _s.time_stepping << std::endl;
    }
    my_file.close();
}

// PUBLIC MEMBER FUNCTIONS
void BayesianOpt::load_data(std::vector<bopt> &bopti, bool validate) {
    this->_bopti    = bopti;
    this->_validate = validate;

    if (validate){
        // build dataset for training and testing
        build_dataset(bopti, *_x_train, *_y_train, *_x_val, *_y_val);
    }else{
        // build dataset for training only
        build_dataset(bopti, *_x_train, *_y_train); 
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
        this->_model.train(*_x_train, *_y_train, _model_params);
    }else{
        condition_model(); 
    }
}

void BayesianOpt::condition_model(){
    // train with maximized log likelihood
    this->_model.train(*_x_train, *_y_train);
}

void BayesianOpt::evaluate_model(){
    // ADD ASSERTION TO CHECK IF DATA IS LOADED
    
    if (_validate){
        std::cout << "--- _model validatation ---" << std::endl; 
        _model.validate(*_x_val, *_y_val);
    }
    else{
        // error: throw exception
        throw std::invalid_argument("Error: scale validation data before validating");
    }
}

void BayesianOpt::sample_posterior(){

    _x_sample      ->resize(this->_num_sample, this->_n_dim);
    _y_sample_mean ->resize(this->_num_sample);
    _y_sample_std  ->resize(this->_num_sample);
    _conf_bound    ->resize(this->_num_sample);

    gen_test_points(*_x_sample);

    this->_model.predict(*_x_sample, false);

    this->_y_sample_mean->array() = this->_model.get_y_test().array();
    this->_y_sample_std->array()  = this->_model.get_y_test_std().array();

}

void BayesianOpt::qUCB(bool _lcb){
    if (_lcb){
        this->_conf_bound->array() = this->_y_sample_mean->array() - 1.96 * this->_y_sample_std->array();

        // sort _conf_bound
        Eigen::VectorXi sorted_inds = Eigen::VectorXi::LinSpaced(this->_conf_bound->size(), 0, this->_conf_bound->size() - 1);

        std::sort(sorted_inds.data(), sorted_inds.data() + sorted_inds.size(),
                [this](int a, int b) { return (*this->_conf_bound)(a) < (*this->_conf_bound)(b); });

        *_conf_bound = (*_conf_bound)(sorted_inds);
        *_x_sample   = (*_x_sample)(sorted_inds, Eigen::all);
    }
    else{
        qUCB();
    }
}

void BayesianOpt::qUCB(){
    this->_conf_bound->array() = this->_y_sample_mean->array() - 1.96 * this->_y_sample_std->array();

    // sort _conf_bound
    Eigen::VectorXi sorted_inds = Eigen::VectorXi::LinSpaced(this->_conf_bound->size(), 0, this->_conf_bound->size() - 1);

    std::sort(sorted_inds.data(), sorted_inds.data() + sorted_inds.size(),
            [this](int a, int b) { return (*this->_conf_bound)(a) > (*this->_conf_bound)(b); });

    *_conf_bound = (*_conf_bound)(sorted_inds);
    *_x_sample   = (*_x_sample)(sorted_inds, Eigen::all);

}

void BayesianOpt::evaluate_samples(){
    this->_num_evals = omp_get_num_procs();
    std::vector<bopt> voxels_evals;
    #pragma omp parallel for
    for (int id = 0; id < this->_num_evals; ++id){

        bopt b; 
        b.temp = _x_sample->coeff(id, 0);
        b.rp   = _x_sample->coeff(id, 1);
        b.vp   = _x_sample->coeff(id, 2);
        b.uvi  = _x_sample->coeff(id, 3);
        b.uvt  = _x_sample->coeff(id, 4);
        
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

