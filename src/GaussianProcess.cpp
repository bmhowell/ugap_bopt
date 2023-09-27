#include "GaussianProcess.h"
#include "common.h"
#include <cmath>
/*

    need to load GP with validation data to track error during GA calibration

    
*/
// #include "helper_functions.h"

/* default constructor */
GaussianProcess::GaussianProcess() {

    // MEMBER VARIABLES
    kernel         = "RBF";            // covariance kernel specification
    trained        = false;            // flag to indicate if GP has been trained
    train_scaled   = false;            // flag to indicate if training data has been scaled
    test_scaled    = false;            // flag to indicate if validation data has been scaled
    val_scaled     = false;            // flag to indicate if validation data has been scaled

    file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_opt/output";
}


/* overload constructor */
GaussianProcess::GaussianProcess(std::string KERNEL, 
                                 std::string FILE_PATH){
    kernel    = KERNEL;
    trained   = false;
    file_path = FILE_PATH; 
}


/* destructor function */
GaussianProcess::~GaussianProcess() {

}

// PRIVATE MEMBER FUNCTIONS
/* infrastructure functions */
void GaussianProcess::kernelGP(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, double& length, double& sigma){
    if (kernel == "RBF"){
        if (X.rows() != Y.rows()){
            
            // kernel construction algorithm for non-symmetric matrices
            Cov = Eigen::MatrixXd::Zero(X.rows(), Y.rows());
            for (int i = 0; i < X.rows(); i++){
                for (int j = 0; j < Y.rows(); j++){
                    // eq 2.31 in Rasmussen & Williams
                    Cov(i, j) = sigma * exp( -(X.row(i) - Y.row(j)).squaredNorm() / 2 / (length * length) );
                }
            }
        } else{
            // kernel construction algorithm for symmetric matrices
            Cov = Eigen::MatrixXd::Zero(X.rows(), Y.rows());
            for (int i = 0; i < X.rows(); i++){
                for (int j = i; j < Y.rows(); j++){
                    // eq 2.31 in Rasmussen & Williams
                    Cov(i, j) = sigma * exp( -(X.row(i) - Y.row(j)).squaredNorm() / 2 / (length * length) );
                    Cov(j, i) = Cov(i, j);
                }
            }
        }
    } else{
        std::cout << "Kernel Error: choose appropriate kernel --> {RBF, }" << std::endl;
    }

}

void GaussianProcess::scale_data(Eigen::MatrixXd& X_TRAIN, Eigen::VectorXd& Y_TRAIN, 
                                 Eigen::MatrixXd& X_VAL,   Eigen::VectorXd& Y_VAL){
    // scale data to zero mean and unit variance

    train_scaled = true;
    val_scaled   = true; 

    x_mean = X_TRAIN.colwise().mean();  
    x_std  = ((X_TRAIN.rowwise() - x_mean.transpose()).array().square().colwise().sum() / (X_TRAIN.rows() - 1)).sqrt();

    y_mean = Y_TRAIN.mean();
    y_std = (Y_TRAIN.array() - y_mean).square().sum() / (Y_TRAIN.size() - 1);
    y_std = sqrt(y_std);

    x_train = (X_TRAIN.rowwise() - x_mean.transpose()).array().rowwise() / x_std.transpose().array();
    y_train = (Y_TRAIN.array() - y_mean) / y_std;

    // scale validation data with training mean and std
    x_val   = (X_VAL.rowwise() - x_mean.transpose()).array().rowwise() / x_std.transpose().array();
    y_val   = (Y_VAL.array() - y_mean) / y_std;

}


void GaussianProcess::scale_data(Eigen::MatrixXd& X, Eigen::VectorXd& Y){
    // scale data to zero mean and unit variance

    train_scaled = true; 

    x_mean = X.colwise().mean();
    x_std  = ((X.rowwise() - x_mean.transpose()).array().square().colwise().sum() / (X.rows() - 1)).sqrt();

    y_mean = Y.mean();
    y_std = (Y.array() - y_mean).square().sum() / (Y.size() - 1);
    y_std = sqrt(y_std);

    x_train = (X.rowwise() - x_mean.transpose()).array().rowwise() / x_std.transpose().array();
    y_train = (Y.array() - y_mean) / y_std;

}


void GaussianProcess::scale_data(Eigen::MatrixXd& X_TEST) {
    // Scale validation data using mean and standard deviation from training data
    test_scaled = true; 
    x_test = (X_TEST.rowwise() - x_mean.transpose()).array().rowwise() / x_std.transpose().array();
}


void GaussianProcess::unscale_data(Eigen::VectorXd& Y_TEST){
    // map scaled y_test back to original scale
    test_scaled = false;
    y_test = Y_TEST.array() * y_std + y_mean;
}


double GaussianProcess::compute_lml(double& length, double& sigma, double& noise){
    
    // compute covariance matrix
    kernelGP(x_train, x_train, length, sigma);
    Ky          = Cov; 

    // add noise to covariance matrix
    for (int i = 0; i < x_train.rows(); i++){
        Ky(i, i) += noise;
    }
    y_train_std = Ky.diagonal().array().sqrt();

    
    // Compute Cholesky decomposition of Ky
    Eigen::LLT<Eigen::MatrixXd> lltOfKy(Ky);
    if (lltOfKy.info() == Eigen::NumericalIssue) {
        // Handle numerical issues with Cholesky decomposition
        // For example, matrix is not positive definite
    }

    // Solve for alpha using Cholesky factorization
    alpha = lltOfKy.solve(y_train);
    L     = lltOfKy.matrixL();

    return -0.5 * (y_train).transpose() * alpha - 0.5 * L.diagonal().array().log().sum();
}

void GaussianProcess::sort_data(Eigen::MatrixXd& PARAM){
    // Custom comparator for sorting by the fourth column in descending order
    auto comparator = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return a(3) > b(3);
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

void GaussianProcess::gen_opt(double& L, double& SF, double& SN){
    double c_length[2] = {1e-3, 1000.0};                           // length scale parameter bounds
    double c_sigma[2]  = {1e-3, 1.0};                               // signal noise variance bounds
    double c_noise[2]  = {1e-10, 1e-3};                             // noise variance bounds

    int pop = 24;                                                   // population size
    int P   = 4;                                                    // number of parents
    int C   = 4;                                                    // number of children
    int G   = 100;                                                 // number of generations
    double lam_1, lam_2;                                            // genetic algorith paramters
    Eigen::MatrixXd param(pop, 4);                                  // ∈ ℝ (population x param + obj)

    // initialize input variables
    std::random_device rd;                                          // Obtain a random seed from the hardware
    std::mt19937 gen(rd());                                         // Seed the random number generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)

    // initialize parameter vectors 
    for (int i = 0; i < param.rows(); ++i){
        param(i, 0) = c_length[0] + (c_length[1] - c_length[0]) * distribution(gen);
        param(i, 1) = c_sigma[0]  + (c_sigma[1]  - c_sigma[0])  * distribution(gen);
        param(i, 2) = c_noise[0]  + (c_noise[1]  - c_noise[0])  * distribution(gen);
    }

    // loop over generations
    Eigen::MatrixXd x_test_sub = x_train.topRows(10);
    Eigen::VectorXd y_test_sub = y_train.head(10);
    std::vector<double> top_performer; 
    std::vector<double> avg_parent; 
    std::vector<double> avg_total; 
    std::vector<double> cost; 

    for (int g = 0; g < G; ++g){
        
        std::cout << "generation: " << g << std::endl;

        // loop over population
        for (int i = 0; i < pop; ++i){

            // compute negative log-likelihood
            param(i, 3) = compute_lml(param(i, 0), param(i, 1), param(i, 2));
        }

        this->sort_data(param);

        // track top and average performers
        top_performer.push_back(param(0, 3));
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
            }

            std::cout << "top performer: " << param(0, 3) << std::endl;
            std::cout << "    length: "    << param(0, 0) << std::endl;
            std::cout << "    sigma:  "    << param(0, 1) << std::endl;
            std::cout << "    noise:  "    << param(0, 2) << std::endl;

            // evaluate loss function of gaussian process with top performer 
            if (val_scaled){
                l = param(0, 0);
                sf = param(0, 1);
                sn = param(0, 2);

                cost.push_back(this->validate()); 
            }
            
            std::cout << "\n====================================\n" << std::endl;
        }
    }

    // store data
    std::cout << "--- storing data ---\n" << std::endl;
    std::ofstream store_params, store_performance, store_cost; 
    store_params.open(file_path      + "/params.txt");
    store_performance.open(file_path + "/performance.txt");
    store_cost.open(file_path        + "/cost.txt");

    // write to file
    store_params      << "length, sigma, noise"                 << std::endl;
    store_performance << "top_performer, avg_parent, avg_total" << std::endl;
    store_cost        << "cost"                                 << std::endl;
    for (int i = 0; i < top_performer.size(); ++i){
        store_performance << top_performer[i] << "," << avg_parent[i] << "," << avg_total[i] << std::endl;
        if (i < param.rows()){
            store_params << param(i, 0) << "," << param(i, 1) << "," << param(i, 2) << std::endl;
        }
        if (i < cost.size()){
            store_cost << cost[i] << "," << std::endl;
        }
    }
    store_params.close();
    store_performance.close();
    std::cout << "--- data saved ---\n" << std::endl;

    // save best parameters to object
    l    = param(0, 0);  // length scale
    sf   = param(0, 1);  // signal noise variance
    sn   = param(0, 2);  // noise variance
    lml  = param(0, 3);  // negative log-likelihood
}

// PUBLIC MEMBER FUNCTIONS
void GaussianProcess::train(Eigen::MatrixXd& X_TRAIN, Eigen::VectorXd& Y_TRAIN){
    std::cout << "\n--- Training Gaussian Process ---\n" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    trained = true;
    scale_data(X_TRAIN, Y_TRAIN);

    // maximize marginal likelihood 
    gen_opt(l, sf, sn);

    // compute lml, alpha and L
    lml = compute_lml(l, sf, sn);

    // compute covariance matrix
    kernelGP(x_train, x_train, l, sf);
    Ky = Cov; 

    // add noise to covariance matrix
    for (int i = 0; i < x_train.rows(); i++){
        Ky(i, i) += sn;
    }
    y_train_std = Ky.diagonal().array().sqrt();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count() / 1e6;
    std::cout << "--- Parameter tunning/Training time time: " << duration / 60 << "min ---" << std::endl;
    std::cout << "--- Parameter Tuning Complete ---\n" << std::endl;
    
    std::cout << "\nlog_marginal_likelihood: " << lml << std::endl;
    std::cout << "    final length: "     << l << std::endl;
    std::cout << "    final sigma:  "     << sf << std::endl;
    std::cout << "    final noise:  "     << sn << std::endl;
}

void GaussianProcess::train(Eigen::MatrixXd& X_TRAIN, Eigen::VectorXd& Y_TRAIN,
                            Eigen::MatrixXd& X_VAL,   Eigen::VectorXd& Y_VAL, 
                            std::vector<double>& model_param){

    auto start = std::chrono::high_resolution_clock::now();

    // function overloading to take into account previously learned parameters 
    trained = true; 
    scale_data(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL);
    
    // unpack model parameters
    l = model_param[0];
    sf = model_param[1];
    sn = model_param[2];
    
    // compute lml, alpha and L
    lml = compute_lml(l, sf, sn);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count() / 1e6;
    std::cout << "--- Training time: " << duration / 60 << " min ---" << std::endl;
    std::cout << "\nlml: " << lml << std::endl;
    std::cout << "   l = " << l << std::endl;
    std::cout << "   sf = " << sf << std::endl;
    std::cout << "   sn = " << sn << std::endl;
    std::cout << "--- ----------------- ---: " << std::endl;
}


void GaussianProcess::train(Eigen::MatrixXd& X_TRAIN, Eigen::VectorXd& Y_TRAIN,
                            Eigen::MatrixXd& X_VAL,   Eigen::VectorXd& Y_VAL){
    std::cout << "\n--- Training Gaussian Process ---\n" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    trained = true;
    scale_data(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL);

    // maximize marginal likelihood 
    gen_opt(l, sf, sn);

    // compute lml, alpha and L
    lml = compute_lml(l, sf, sn);

    // compute covariance matrix
    kernelGP(x_train, x_train, l, sf);
    Ky = Cov; 

    // add noise to covariance matrix
    for (int i = 0; i < x_train.rows(); i++){
        Ky(i, i) += sn;
    }
    y_train_std = Ky.diagonal().array().sqrt();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count() / 1e6;
    std::cout << "--- Parameter tunning/Training time time: " << duration / 60 << "min ---" << std::endl;
    std::cout << "--- Parameter Tuning Complete ---\n" << std::endl;
    
    std::cout << "\nlog_marginal_likelihood: " << lml << std::endl;
    std::cout << "    final length: "     << l << std::endl;
    std::cout << "    final sigma:  "     << sf << std::endl;
    std::cout << "    final noise:  "     << sn << std::endl;
}

double GaussianProcess::validate(){

    if (!trained){
        throw std::invalid_argument("Error: train GP before validating");
    }

    // compute covariance matrix
    kernelGP(x_train, x_train, l, sf);
    Ky          = Cov; 
    // add noise to covariance matrix
    for (int i = 0; i < x_train.rows(); i++){
        Ky(i, i) += sn;
    }

    // Compute Cholesky decomposition of Ky
    Eigen::LLT<Eigen::MatrixXd> lltOfKy(Ky);
    if (lltOfKy.info() == Eigen::NumericalIssue) {
        // Handle numerical issues with Cholesky decomposition (e.g. !PD)
    }

    // Solve for alpha using Cholesky factorization
    alpha = lltOfKy.solve(y_train);
    L     = lltOfKy.matrixL();

    // initialize covariance sub matrices
    Eigen::MatrixXd Ks(x_train.rows(), x_val.rows());          // ∈ ℝ (m x l)
    Eigen::MatrixXd Kss(x_val.rows(), x_val.rows());          // ∈ ℝ (l x l))

    kernelGP(x_train, x_val, l, sf);
    Ks = Cov;
    Eigen::VectorXd y_val_out = Ks.transpose() * alpha;                             // eq. 2.25

    error_val = (y_val - y_val_out).norm();

    std::cout << "ERROR: " << error_val << std::endl;
    return error_val; 
}

void GaussianProcess::predict(Eigen::MatrixXd& X_TEST, bool compute_std){
    
    // throw error if training has not occured
    if (!trained){
        throw std::invalid_argument("Error: train GP before predicting");
        return;
    }


    scale_data(X_TEST);

    // initialize covariance sub matrices
    Eigen::MatrixXd Ks(x_train.rows(), x_test.rows());          // ∈ ℝ (m x l)
    Eigen::MatrixXd Kss(x_test.rows(), x_test.rows());          // ∈ ℝ (l x l))

    // compute mean y_test ∈ ℝ (l x m) -> see Algorithm 2.1 in Rasmussen & Williams
    kernelGP(x_train, x_test, l, sf);
    Ks = Cov;
    y_test = Ks.transpose() * alpha;                                  // eq. 2.25
    unscale_data(y_test);

    if (compute_std){
        // compute variance: V  ∈ ℝ (l x l)
        kernelGP(x_test,  x_test, l, sf);
        Kss = Cov;
        V = L.triangularView<Eigen::Lower>().solve(Ks); 
        Cov = Kss - V.transpose() * V;   
        y_test_std = Cov.diagonal().array();

        // check for negative variance bc numerical instability
        for (int i = 0; i < y_test_std.size(); ++i){
            if (y_test_std(i) < 0){
                y_test_std(i) = 0;
            }
        }

        // scale y_test by the variance of the training data 
        // see: https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/gaussian_process/_gpr.py#L26
        // line 433
        y_test_std *= y_std * y_std;
        y_test_std  = y_test_std.array().sqrt();
    }
}; 

/* accessor functions */
Eigen::MatrixXd GaussianProcess::get_Cov(){
    return Cov;
};


Eigen::VectorXd GaussianProcess::get_y_test(){
    return y_test;
};


Eigen::VectorXd GaussianProcess::get_y_train_std(){
    return y_train_std;
};


Eigen::VectorXd GaussianProcess::get_y_test_std(){
    return y_test_std;
};

std::vector<int> GaussianProcess::get_candidates(){

    Eigen::VectorXd y_tot     = Eigen::VectorXd::Zero(y_train.size() + y_test.size());
    Eigen::VectorXd y_tot_std = Eigen::VectorXd::Zero(y_train.size() + y_test.size());
    Eigen::VectorXd y_max     = Eigen::VectorXd::Zero(y_train.size() + y_test.size());

    y_tot << y_train, y_test;
    y_tot_std << y_train_std, y_test_std;
    
    // compute upper bound confidence interval
    y_max = y_tot + 1.96 * y_tot_std;     

    return candidates;
}


