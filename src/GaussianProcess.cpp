
#include "GaussianProcess.h"
// #include "AcquisitionFunction.h"

/* default constructor */
GaussianProcess::GaussianProcess() {

    // MEMBER VARIABLES
    kernel   = "RBF";                  // covariance kernel specification
    trained  = false;                  // flag to indicate if GP has been trained

    file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/materials_opt/output";
}

/* overload constructor */
GaussianProcess::GaussianProcess(std::string KERNEL, std::string FILE_PATH){
    kernel    = KERNEL;
    trained   = false;
    file_path = FILE_PATH; 
}

/* destructor function */
GaussianProcess::~GaussianProcess() {

}

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

void GaussianProcess::scale_data(Eigen::MatrixXd& X, Eigen::VectorXd& Y){
    // scale data to zero mean and unit variance
    //      - l: number of test points
    //      - m: number of data points
    //      - n: number of variables
    Eigen::VectorXd mean = X.colwise().mean();
    Eigen::VectorXd std  = ((X.rowwise() - mean.transpose()).array().square().colwise().sum() / (X.rows() - 1)).sqrt();
    x_train = (X.rowwise() - mean.transpose()).array().rowwise() / std.transpose().array();

    double y_mean = Y.mean();
    double y_stddev = (Y.array() - y_mean).square().sum() / (Y.size() - 1);
    y_stddev = (y_stddev);
    y_train = (Y.array() - y_mean) / y_stddev;

}



double GaussianProcess::compute_neg_log_likelihood(double& length, double& sigma, double& noise){
    
    // compute covariance matrix
    kernelGP(x_train, x_train, length, sigma);
    Ky = Cov;

    // add noise to covariance matrix
    for (int i = 0; i < x_train.rows(); i++){
        Ky(i, i) += noise;
    }

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

void GaussianProcess::train(Eigen::MatrixXd& X_TRAIN, Eigen::VectorXd& Y_TRAIN){
    std::cout << "\n--- Training Gaussian Process ---\n" << std::endl;

    trained = true; 
    scale_data(X_TRAIN, Y_TRAIN);

    double c_length[2] = {1e-3, 1000.0};                           // length scale parameter bounds
    double c_sigma[2]  = {1e-3, 1.0};                               // signal noise variance bounds
    double c_noise[2]  = {1e-10, 1e-3};                             // noise variance bounds

    int pop = 24;                                                   // population size
    int P   = 4;                                                    // number of parents
    int C   = 4;                                                    // number of children
    int G   = 10;                                                   // number of generations
    double lam_1, lam_2;                                            // genetic algorith paramters
    Eigen::MatrixXd param(pop, 4);                                  // ∈ ℝ (population x param + obj)

    // initialize input variables
    std::random_device rd;                                          // Obtain a random seed from the hardware
    std::mt19937 gen(rd());                                         // Seed the random number generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)


    // initialize parameter vectors
    param(0, 0) = 18.0225; 
    param(0, 1) = 0.507717; 
    param(0, 2) = 4.25249e-05; 
    for (int i = 1; i < param.rows(); ++i){
        param(i, 0) = c_length[0] + (c_length[1] - c_length[0]) * distribution(gen);
        param(i, 1) = c_sigma[0]  + (c_sigma[1]  - c_sigma[0])  * distribution(gen);
        param(i, 2) = c_noise[0]  + (c_noise[1]  - c_noise[0])  * distribution(gen);
    }

    // loop over generations
    for (int g = 0; g < G; ++g){
        std::cout << "generation: " << g << std::endl;
        // loop over population
        for (int i = 0; i < pop; ++i){
            // compute negative log-likelihood
            param(i, 3) = compute_neg_log_likelihood(param(i, 0), param(i, 1), param(i, 2));
        }

        // Custom comparator for sorting by the fourth column in descending order
        auto comparator = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            return a(3) > b(3);
        };

        // Convert Eigen matrix to std::vector of Eigen::VectorXd
        std::vector<Eigen::VectorXd> rows;
        for (int i = 0; i < param.rows(); ++i) {
            rows.push_back(param.row(i));
        }

        // Sort using the custom comparator
        std::sort(rows.begin(), rows.end(), comparator);

        // Copy sorted rows back to Eigen matrix
        for (int i = 0; i < param.rows(); ++i) {
            param.row(i) = rows[i];
        }

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
            param(i, 1) = c_sigma[0]  + (c_sigma[1]  - c_sigma[0])  * distribution(gen);
            param(i, 2) = c_noise[0]  + (c_noise[1]  - c_noise[0])  * distribution(gen);
        }

        std::cout << "top performer: " << param(0, 3) << std::endl;
        std::cout << "    length: " << param(0, 0) << std::endl;
        std::cout << "    sigma:  " << param(0, 1) << std::endl;
        std::cout << "    noise:  " << param(0, 2) << std::endl;

    }

    l  = param(0, 0);
    sf = param(0, 1);
    sn = param(0, 2);

    std::cout << "\nneg_log_likelihood: " << param(0, 3) << std::endl;
    std::cout << "    final length: " << param(0, 0) << std::endl;
    std::cout << "    final sigma:  " << param(0, 1) << std::endl;
    std::cout << "    final noise:  " << param(0, 2) << std::endl;
    std::cout << "\n--- Parameter Tuning Complete ---\n" << std::endl;


}

void GaussianProcess::predict(Eigen::MatrixXd& X_TEST, char save){
    
    
    // throw error if training has not occured
    if (!trained){
        throw std::invalid_argument("Error: train GP before predicting");
        return;
    }

    x_test = X_TEST;

    // initialize covariance sub matrices
    Eigen::MatrixXd Ks(x_train.rows(), x_test.rows());          // ∈ ℝ (m x l)
    Eigen::MatrixXd Kss(x_test.rows(), x_test.rows());          // ∈ ℝ (l x l))

    // compute required covariance matrices
    kernelGP(x_train, x_test, l, sf);
    Ks = Cov;
    kernelGP(x_test, x_test, l, sf);
    Kss = Cov;

    // compute mean y_test ∈ ℝ (l x m) -> see Algorithm 2.1 in Rasmussen & Williams
    y_test = Ks.transpose() * alpha;                                  // eq. 2.25

    // compute variance: V  ∈ ℝ (l x l)
    Eigen::MatrixXd V = Ky.llt().matrixL().solve(Ks);                 // eq. 2.26
    // Eigen::MatrixXd V = L.solve(Ks);                                  // eq. 2.26
    Cov = Kss - V.transpose() * V;                                    // eq. 2.26   

}

/* accessor functions */
std::string GaussianProcess::get_kernel() const {
    return kernel;
};

// float GaussianProcess::get_lengthScale() const {
//     return length;
// }

// float GaussianProcess::get_signalNoise() const {
//     return sigma;
// }

Eigen::MatrixXd& GaussianProcess::get_Cov(){
    return Cov;
};

Eigen::VectorXd& GaussianProcess::get_y_test(){
    return y_test;
};


