#include "GaussianProcess.h"
#include "common.h"
#include <cmath>

/* default constructor */
GaussianProcess::GaussianProcess() {

    // MEMBER VARIABLES
    _kernel         = "RBF";            // covariance kernel specification
    _trained        = false;            // flag to indicate if GP has been trained
    _train_scaled   = false;            // flag to indicate if training data has been scaled
    _test_scaled    = false;            // flag to indicate if validation data has been scaled
    _val_scaled     = false;            // flag to indicate if validation data has been scaled

    // initialize hyper params
    _l      = 1.0;
    _sf     = 0.76;
    _sn     = 0.000888;
    _p      = 1.0;
    _a      = 1.0;
    _lml    = -1000.0;

    _file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_opt/output";
}

/* overload constructor */
GaussianProcess::GaussianProcess(std::string KERNEL, 
                                 std::string FILE_PATH){
    _kernel    = KERNEL;
    _trained   = false;
    _file_path = FILE_PATH;

    // initialize hyper params
    _l      = 1.0;
    _sf     = 0.76;
    _sn     = 0.000888;
    _p      = 1.0;
    _a      = 1.0;
    _lml    = -1000.0;

}

/* destructor function */
GaussianProcess::~GaussianProcess() {

}

// PUBLIC MEMBER FUNCTIONS
void GaussianProcess::train(Eigen::MatrixXd& X_TRAIN, Eigen::VectorXd& Y_TRAIN){
    std::cout << "\n===== TUNING GP PARAMETERS======" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    _trained = true;
    this->scale_data(X_TRAIN, Y_TRAIN);

    // maximize marginal likelihood 
    this->gen_tune_param();

    // compute _lml, _alpha and _L
    double lml_ = this->compute_lml(_l, _sf, _sn, _p, _a);
    if (!std::isnan(lml_)) { _lml = lml_; } else { _lml = -1000.0; }

    // compute covariance matrix
    this->kernelGP(_x_train, _x_train, _l, _sf, _p, _a);
    _Ky = _Cov; 

    // add noise to covariance matrix
    for (int i = 0; i < _x_train.rows(); i++){
        _Ky(i, i) += _sn;
    }
    _y_train_std = _Ky.diagonal().array().sqrt();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count() / 1e6;
    std::cout << "--- Parameter tunning/Training time: " << duration / 60 << "min ---" << std::endl;
    std::cout << "--- Parameter Tuning Complete ---\n" << std::endl;
    
    std::cout << "\nlog_marginal_likelihood: " << _lml << std::endl;
    std::cout << "    final length: "     << _l << std::endl;
    std::cout << "    final sigma:  "     << _sf << std::endl;
    std::cout << "    final noise:  "     << _sn << std::endl;
    std::cout << "    final period: "     << _p << std::endl;
    std::cout << "    final alpha:  "     << _a << std::endl;
}

void GaussianProcess::train(Eigen::MatrixXd& X_TRAIN, Eigen::VectorXd& Y_TRAIN,
                            std::vector<double>& model_param){

    // auto start = std::chrono::high_resolution_clock::now();

    // function overloading to take into account previously learned parameters 
    _trained = true; 
    scale_data(X_TRAIN, Y_TRAIN);
    
    // unpack model parameters
    _l     = model_param[0];
    _sf    = model_param[1];
    _sn    = model_param[2];
    _p     = model_param[3];
    _a     = model_param[4];
    
    // compute _lml, _alpha and _L
    _lml = compute_lml(_l, _sf, _sn, _p, _a);
    
    std::cout << "===== TUNING GP COMPLETE======\n" << std::endl;
}

double GaussianProcess::validate(Eigen::MatrixXd& X_VAL, Eigen::VectorXd& Y_VAL){

    if (!_trained){
        throw std::invalid_argument("Error: train GP before validating");
    }

    this->scale_data(X_VAL, Y_VAL, true);
    // compute covariance matrix
    kernelGP(_x_train, _x_train, _l, _sf, _p, _a);
    _Ky          = _Cov; 
    // add noise to covariance matrix
    for (int i = 0; i < _x_train.rows(); i++){
        _Ky(i, i) += _sn;
    }

    // Compute Cholesky decomposition of _Ky
    Eigen::LLT<Eigen::MatrixXd> lltOfKy(_Ky);
    if (lltOfKy.info() == Eigen::NumericalIssue) {
        // Handle numerical issues with Cholesky decomposition (e.g. !PD)
    }

    // Solve for _alpha using Cholesky factorization
    _alpha = lltOfKy.solve(_y_train);
    _L     = lltOfKy.matrixL();

    // initialize covariance sub matrices
    Eigen::MatrixXd Ks(_x_train.rows(), _x_val.rows());          // ∈ ℝ (m x _l)
    Eigen::MatrixXd Kss(_x_val.rows(), _x_val.rows());           // ∈ ℝ (_l x _l))

    kernelGP(_x_train, _x_val, _l, _sf, _p, _a);
    Ks = _Cov;
    Eigen::VectorXd y_val_out = Ks.transpose() * _alpha;                             // eq. 2.25

    _error_val = (_y_val - y_val_out).norm();

    std::cout << "ERROR: " << _error_val << std::endl;
    return _error_val; 
}

void GaussianProcess::predict(Eigen::MatrixXd& X_TEST, bool compute_std){
    
    // throw error if training has not occured
    if (!_trained){
        throw std::invalid_argument("Error: train GP before predicting");
        return;
    }


    scale_data(X_TEST);

    // initialize covariance sub matrices
    Eigen::MatrixXd Ks(_x_train.rows(), _x_test.rows());          // ∈ ℝ (m x _l)
    Eigen::MatrixXd Kss(_x_test.rows(), _x_test.rows());          // ∈ ℝ (_l x _l))

    // compute mean _y_test ∈ ℝ (_l x m) -> see Algorithm 2.1 in Rasmussen & Williams
    kernelGP(_x_train, _x_test, _l, _sf, _p, _a);
    Ks = _Cov;
    _y_test = Ks.transpose() * _alpha;                                  // eq. 2.25
    unscale_data(_y_test);

    if (compute_std){
        // compute variance: _V  ∈ ℝ (_l x _l)
        kernelGP(_x_test,  _x_test, _l, _sf, _p, _a);
        Kss = _Cov;
        _V = _L.triangularView<Eigen::Lower>().solve(Ks); 
        _Cov = Kss - _V.transpose() * _V;   
        _y_test_std = _Cov.diagonal().array();

        // check for negative variance bc numerical instability
        for (int i = 0; i < _y_test_std.size(); ++i){
            if (_y_test_std(i) < 0){
                _y_test_std(i) = 0;
            }
        }

        // scale _y_test by the variance of the training data 
        // see: https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/gaussian_process/_gpr.py#L26
        // line 433
        _y_test_std *= _y_std * _y_std;
        _y_test_std  = _y_test_std.array().sqrt();
    }
}; 

/* accessor functions */
Eigen::MatrixXd GaussianProcess::get_Cov(){
    return _Cov;
};

Eigen::VectorXd GaussianProcess::get_y_test(){
    return _y_test;
};

Eigen::VectorXd GaussianProcess::get_y_train_std(){
    return _y_train_std;
};

Eigen::VectorXd GaussianProcess::get_y_test_std(){
    return _y_test_std;
};

std::vector<int> GaussianProcess::get_candidates(){

    Eigen::VectorXd y_tot     = Eigen::VectorXd::Zero(_y_train.size() + _y_test.size());
    Eigen::VectorXd y_tot_std = Eigen::VectorXd::Zero(_y_train.size() + _y_test.size());
    Eigen::VectorXd y_max     = Eigen::VectorXd::Zero(_y_train.size() + _y_test.size());

    y_tot << _y_train, _y_test;
    y_tot_std << _y_train_std, _y_test_std;
    
    // compute upper bound confidence interval
    y_max = y_tot + 1.96 * y_tot_std;     

    return _candidates;
}

double GaussianProcess::get_length_param(){
    return _l;
}

double GaussianProcess::get_sigma_param(){
    return _sf;
}

double GaussianProcess::get_noise_param(){
    return _sn;
}

double GaussianProcess::get_period_param(){
    return _p;
}

double GaussianProcess::get_alpha_param(){
    return _a;
}

// PRIVATE MEMBER FUNCTIONS
/* infrastructure functions */
void GaussianProcess::kernelGP(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, 
                               double &length, double &sigma, double &p, double &a){
    if (_kernel == "RBF"){
        if (X.rows() != Y.rows()){
            
            // kernel construction algorithm for non-symmetric matrices
            _Cov = Eigen::MatrixXd::Zero(X.rows(), Y.rows());
            for (int i = 0; i < X.rows(); i++){
                for (int j = 0; j < Y.rows(); j++){
                    // eq 2.31 in Rasmussen & Williams
                    _Cov(i, j) = sigma * exp( -(X.row(i) - Y.row(j)).squaredNorm() / 2 / (length * length) );
                }
            }
        } else{
            // kernel construction algorithm for symmetric matrices
            _Cov = Eigen::MatrixXd::Zero(X.rows(), Y.rows());
            for (int i = 0; i < X.rows(); i++){
                for (int j = i; j < Y.rows(); j++){
                    // eq 2.31 in Rasmussen & Williams
                    _Cov(i, j) = sigma * exp( -(X.row(i) - Y.row(j)).squaredNorm() / 2 / (length * length) );
                    _Cov(j, i) = _Cov(i, j);
                }
            }
        }
    } else if (_kernel == "RQK") {
        double c_inv = 0.5 / (length * length * a);
        if (X.rows() != Y.rows()){
            // kernel construction algorithm for non-symmetric matrices
            _Cov = Eigen::MatrixXd::Zero(X.rows(), Y.rows());
            for (int i = 0; i < X.rows(); i++){
                for (int j = 0; j < Y.rows(); j++){
                    _Cov(i, j) = sigma * pow(exp(1 + (X.row(i) - Y.row(j)).squaredNorm() * c_inv ), -a);
                }
            }
        } else{
            // kernel construction algorithm for symmetric matrices
            _Cov = Eigen::MatrixXd::Zero(X.rows(), Y.rows());
            for (int i = 0; i < X.rows(); i++){
                for (int j = i; j < Y.rows(); j++){
                    _Cov(i, j) = sigma * pow(exp(1 + (X.row(i) - Y.row(j)).squaredNorm() * c_inv ), -a);
                    _Cov(j, i) = _Cov(i, j);
                }
            }
        }
    }
    
    
    else if (_kernel == "LOC_PER") {

        double l_inv     = 1. / (length * length);
        double p_inv     = 1. / p;
        if (X.rows() != Y.rows()){
            
            // kernel construction algorithm for non-symmetric matrices
            _Cov = Eigen::MatrixXd::Zero(X.rows(), Y.rows());
            for (int i = 0; i < X.rows(); i++){
                for (int j = 0; j < Y.rows(); j++){
                    double sqrd_exp = sigma * 0.5 * exp( -(X.row(i) - Y.row(j)).squaredNorm() * l_inv );
                    double periodic = exp( -2.0 * l_inv * (M_PI * (X.row(i) - Y.row(j)) * p_inv).array().sin().square().sum() );
                    _Cov(i, j) = sqrd_exp * periodic;
                }
            }
        } else{
            // kernel construction algorithm for symmetric matrices
            _Cov = Eigen::MatrixXd::Zero(X.rows(), Y.rows());
            for (int i = 0; i < X.rows(); i++){
                for (int j = i; j < Y.rows(); j++){
                    double sqrd_exp = sigma * 0.5 * exp( -(X.row(i) - Y.row(j)).squaredNorm() * l_inv );
                    double periodic = exp( -2.0 * l_inv * (M_PI * (X.row(i) - Y.row(j)) * p_inv).array().sin().square().sum() );
                    _Cov(i, j) = sqrd_exp * periodic;
                    _Cov(j, i) = _Cov(i, j);
                }
            }
        }
    } else{
        std::cout << "Kernel Error: choose appropriate kernel --> {RBF, RQK, LOC_PER}" << std::endl;
    }

}

void GaussianProcess::scale_data(Eigen::MatrixXd& X_VAL,   Eigen::VectorXd& Y_VAL, bool VAL){
    // scale data to zero mean and unit variance

    if (_trained){_val_scaled   = true;}
    else{throw std::invalid_argument("Error: train GP before scaling validation data");}

    _x_val   = (X_VAL.rowwise() - _x_mean.transpose()).array().rowwise() / _x_std.transpose().array();
    _y_val   = (Y_VAL.array() - _y_mean) / _y_std;

}

void GaussianProcess::scale_data(Eigen::MatrixXd& X, Eigen::VectorXd& Y){
    // scale data to zero mean and unit variance

    _train_scaled = true; 

    _x_mean = X.colwise().mean();
    _x_std  = ((X.rowwise() - _x_mean.transpose()).array().square().colwise().sum() / (X.rows() - 1)).sqrt();

    _y_mean = Y.mean();
    _y_std = (Y.array() - _y_mean).square().sum() / (Y.size() - 1);
    _y_std = sqrt(_y_std);

    _x_train = (X.rowwise() - _x_mean.transpose()).array().rowwise() / _x_std.transpose().array();
    _y_train = (Y.array() - _y_mean) / _y_std;

}

void GaussianProcess::scale_data(Eigen::MatrixXd &X_TEST) {
    // Scale validation data using mean and standard deviation from training data
    _test_scaled = true; 
    _x_test = (X_TEST.rowwise() - _x_mean.transpose()).array().rowwise() / _x_std.transpose().array();
}

void GaussianProcess::unscale_data(Eigen::VectorXd& Y_TEST){
    // map scaled _y_test back to original scale
    _test_scaled = false;
    _y_test = Y_TEST.array() * _y_std + _y_mean;
}

double GaussianProcess::compute_lml(double &length, double &sigma, double &noise, double &period, double &a){
    
    // compute covariance matrix
    kernelGP(_x_train, _x_train, length, sigma, period, a);
    _Ky = _Cov; 

    // add noise to covariance matrix
    for (int i = 0; i < _x_train.rows(); i++){
        _Ky(i, i) += noise;
    }
    _y_train_std = _Ky.diagonal().array().sqrt();

    
    // Compute Cholesky decomposition of _Ky
    Eigen::LLT<Eigen::MatrixXd> lltOfKy(_Ky);
    if (lltOfKy.info() == Eigen::NumericalIssue) {
        // Handle numerical issues with Cholesky decomposition
        // For example, matrix is not positive definite
    }

    // Solve for _alpha using Cholesky factorization
    _alpha = lltOfKy.solve(_y_train);
    _L     = lltOfKy.matrixL();

    return -0.5 * (_y_train).transpose() * _alpha - 0.5 * _L.diagonal().array().log().sum();
}

void GaussianProcess::sort_data(Eigen::MatrixXd& PARAM){
    // Custom comparator for sorting by the last column in descending order
    auto comparator = [&PARAM](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return a(PARAM.cols() - 1) > b(PARAM.cols() - 1);
    };

    // Convert Eigen matrix to std::vector of Eigen::VectorXd
    std::vector<Eigen::VectorXd> rows;
    for (int i = 0; i < PARAM.rows(); ++i) {
        rows.push_back(PARAM.row(i));
    }

    // Sort using the custom comparator
    std::sort(rows.begin(), rows.end(), comparator);

    // Copy sorted rows back to Eigen matrix
    for (int i = 0; i < PARAM.rows(); ++i) {
        PARAM.row(i) = rows[i];
    }
}

void GaussianProcess::gen_tune_param(){
    double c_length[2] = {1e-2, 10.0};                              // length scale parameter bounds
    double c_sigma[2]  = {1e-2, 1.0};                               // signal noise variance bounds
    double c_noise[2]  = {1e-7, 1e-3};                              // noise variance bounds
    double c_period[2] = {1e-2, 10.0};                              // period bounds
    double c_a[2]      = {1e-2, 10.0};                              // large/small length scale ratio bounds

    int pop = 24;                                                   // population size
    int P   = 4;                                                    // number of parents
    int C   = 4;                                                    // number of children
    int G   = 100;                                                  // number of generations
    double lam_1, lam_2;                                            // genetic algorith paramters

    Eigen::MatrixXd param;
    if (_kernel == "RBF") {
        param = Eigen::MatrixXd(pop, 4);                             // ∈ ℝ (population x param + obj)
    } else if (_kernel == "LOC_PER" || _kernel == "RQK") {
        param = Eigen::MatrixXd(pop, 5);                             // ∈ ℝ (population x param + obj)
    } else {
        Eigen::MatrixXd param(pop, 4);                              // ∈ ℝ (population x param + obj)
        throw std::invalid_argument("Error: choose appropriate kernel --> {RBF, LOC_PER, RQK}");
        return;
    }

    // initialize input variables
    std::random_device rd;                                          // Obtain a random seed from the hardware
    std::mt19937 gen(rd());                                         // Seed the random number generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)

    // best guess value
    param(0, 0) = _l;
    param(0, 1) = _sf;
    param(0, 2) = _sn;
    if (_kernel == "LOC_PER") { param(0, 3) = _p; }
    if (_kernel == "RQK") { param(0, 3) = _a; }
    
    // initialize parameter vectors for remaining rows
    for (int i = 1; i < param.rows(); ++i){
        param(i, 0) = c_length[0] + (c_length[1] - c_length[0]) * distribution(gen);
        param(i, 1) = c_sigma[0]  + (c_sigma[1]  - c_sigma[0])  * distribution(gen);
        param(i, 2) = c_noise[0]  + (c_noise[1]  - c_noise[0])  * distribution(gen);
        if (_kernel == "LOC_PER") { param(i, 3) = c_period[0] + (c_period[1] - c_period[0]) * distribution(gen); }
        if (_kernel == "RQK") { param(i, 3) = c_a[0] + (c_a[1] - c_a[0]) * distribution(gen); }
    }

    // loop over generations
    Eigen::MatrixXd x_test_sub = _x_train.topRows(10);
    Eigen::VectorXd y_test_sub = _y_train.head(10);
    std::vector<double> top_performer; 
    std::vector<double> avg_parent; 
    std::vector<double> avg_total; 
    std::vector<double> cost;
    double lml_output;

    for (int g = 0; g < G; ++g){

        // loop over population
        for (int i = 0; i < pop; ++i){
            // compute negative log-likelihood
            if (_kernel == "RBF") {
                lml_output  = this->compute_lml(param(i, 0), param(i, 1), param(i, 2), _p, _a);
            } else if (_kernel == "LOC_PER") {
                lml_output  = this->compute_lml(param(i, 0), param(i, 1), param(i, 2), param(i, 3), _a);
            } else if (_kernel == "RQK") {
                lml_output  = this->compute_lml(param(i, 0), param(i, 1), param(i, 2), _p, param(i, 3));
            } else {
                throw std::invalid_argument("Error: choose appropriate kernel --> {RBF, RQK, LOC_PER}");
                return;
            }

            // check for nan
            if (!std::isnan(lml_output)) {
                param(i, param.cols()-1) = lml_output;
            } else {
                param(i, param.cols()-1) = -1000.0;
            }
        }

        this->sort_data(param);

        // track top and average performers
        top_performer.push_back(param(0, param.cols() - 1));
        avg_parent.push_back(param.col(param.cols() - 1).head(P).mean());
        avg_total.push_back(param.col(param.cols() - 1).mean());

        if (g < G - 1){
            // mate the top performing parents
            for (int i = 0; i < P; i+=2){
                lam_1 = distribution(gen);
                lam_2 = distribution(gen);
                param.row(i + C)        = lam_1 * param.row(i) + (1 - lam_1) * param.row(i+1);
                param.row(i + C + 1)    = lam_2 * param.row(i) + (1 - lam_2) * param.row(i+1);
            }

            // initialize parameter vectors for remaining rows
            for (int i = P+C; i < param.rows(); ++i){
                param(i, 0) = c_length[0] + (c_length[1] - c_length[0]) * distribution(gen);
                param(i, 1) = c_sigma[0]  + (c_sigma[1]  -  c_sigma[0]) * distribution(gen);
                param(i, 2) = c_noise[0]  + (c_noise[1]  -  c_noise[0]) * distribution(gen);
                if (_kernel == "LOC_PER") { param(i, 3) = c_period[0] + (c_period[1] - c_period[0]) * distribution(gen); }
                if (_kernel == "RQK") { param(i, 3) = c_a[0] + (c_a[1] - c_a[0]) * distribution(gen); }
            }
        }
    }

    // store data
    std::cout << "--- storing data ---\n" << std::endl;
    std::ofstream store_params, store_performance, store_cost; 
    store_params.open(_file_path      + "/params.txt");
    store_performance.open(_file_path + "/gp_lml.txt");

    // create file headers
    store_performance << "top_performer, avg_parent, avg_total" << std::endl;
    if (_kernel == "RBF") {
        store_params      << "length, sigma, noise, lml"                 << std::endl;
    } else if (_kernel == "RQK") {
        store_params      << "length, sigma, noise, alpha, lml"          << std::endl;
    } else if (_kernel == "LOC_PER") {
        store_params      << "length, sigma, noise, period, lml"         << std::endl;
    } else {
        throw std::invalid_argument("Error: choose appropriate kernel --> {RBF, LOC_PER}");
        return;
    }

    // store data
    for (int i = 0; i < top_performer.size(); ++i){
        store_performance << top_performer[i] << "," << avg_parent[i] << "," << avg_total[i] << std::endl;
        if (i < param.rows()){
            if (_kernel == "RBF") {
                store_params << param(i, 0) 
                             << "," << param(i, 1) 
                             << "," << param(i, 2) 
                             << "," << param(i, 3) 
                             << std::endl;
            } else if (_kernel == "RQK" || _kernel == "LOC_PER") {
                store_params << param(i, 0) 
                             << "," << param(i, 1) 
                             << "," << param(i, 2) 
                             << "," << param(i, 3)
                             << "," << param(i, 4)
                             << std::endl;
            } else {
                throw std::invalid_argument("Error: choose appropriate kernel --> {RBF, RQK, LOC_PER}");
                return;
            }
        }
    }
    store_params.close();
    store_performance.close();

    // save best parameters to object
    if (_kernel == "RBF") {
        _l    = param(0, 0);  // length scale
        _sf   = param(0, 1);  // signal noise variance
        _sn   = param(0, 2);  // noise variance
        _lml  = param(0, 3);  // negative log-likelihood
    } else if (_kernel == "RQK") {
        _l    = param(0, 0);  // length scale
        _sf   = param(0, 1);  // signal noise variance
        _sn   = param(0, 2);  // noise variance
        _a    = param(0, 3);  // large/small length scale ratio
        _lml  = param(0, 4);  // negative log-likelihood
    } else if (_kernel == "LOC_PER") {
        _l    = param(0, 0);  // length scale
        _sf   = param(0, 1);  // signal noise variance
        _sn   = param(0, 2);  // noise variance
        _p    = param(0, 3);  // period
        _lml  = param(0, 4);  // negative log-likelihood
    } else {
        throw std::invalid_argument("Error: choose appropriate kernel --> {RBF, LOC_PER}");
        return;
    }

}
