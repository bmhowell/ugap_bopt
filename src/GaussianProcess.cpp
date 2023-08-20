
#include "GaussianProcess.h"
// #include "AcquisitionFunction.h"

/* default constructor */
GaussianProcess::GaussianProcess() {

    // MEMBER VARIABLES
    length   = 1.0f;                   // length scale parameter
    sigma    = 1.0f;                   // signal noise variance
    noise    = 1e-8;                   // noise parameter
    kernel   = "RBF";                  // covariance kernel specification
    trained  = false;                  // flag to indicate if GP has been trained

    file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/materials_opt/output";
}

/* overload constructor */
GaussianProcess::GaussianProcess(float L, float SF, double N, std::string KERNEL, std::string FILE_PATH){
    length    = L;
    sigma     = SF;
    noise     = N;  
    kernel    = KERNEL;
    file_path = FILE_PATH; 
}

/* destructor function */
GaussianProcess::~GaussianProcess() {

}

/* infrastructure functions */
void GaussianProcess::kernelGP(Eigen::MatrixXd* X, Eigen::MatrixXd* Y){
    if (kernel == "RBF"){
        if (X->rows() != Y->rows()){
            
            // kernel construction algorithm for non-symmetric matrices
            Cov = Eigen::MatrixXd::Zero(X->rows(), Y->rows());
            for (int i = 0; i < X->rows(); i++){
                for (int j = 0; j < Y->rows(); j++){
                    // eq 2.31 in Rasmussen & Williams
                    Cov(i, j) = sigma * exp( -(X->row(i) - Y->row(j)).squaredNorm() / 2 / (length * length) );
                }
            }
        } else{
            // kernel construction algorithm for symmetric matrices
            Cov = Eigen::MatrixXd::Zero(X->rows(), Y->rows());
            for (int i = 0; i < X->rows(); i++){
                for (int j = i; j < Y->rows(); j++){
                    // eq 2.31 in Rasmussen & Williams
                    Cov(i, j) = sigma * exp( -(X->row(i) - Y->row(j)).squaredNorm() / 2 / (length * length) );
                    Cov(j, i) = Cov(i, j);
                }
            }
        }
    } else{
        std::cout << "Kernel Error: choose appropriate kernel --> {RBF, }" << std::endl;
    }

}

void GaussianProcess::generate_random_points(int num_sample, int x_size, float mean, float stddev, float scale){

    std::random_device rd;
    std::mt19937 gen(rd());

    m_x_sample_distribution = Eigen::MatrixXd::Zero(num_sample, x_size);
    for (unsigned int i = 0; i < num_sample; ++i){
        std::normal_distribution<float> d(mean, stddev);
        for(unsigned int j = 0; j < x_size; ++j) {
            m_x_sample_distribution(i, j) = d(gen) * scale;
        }
    }
}

void GaussianProcess::train(Eigen::MatrixXd* X_TRAIN, Eigen::VectorXd* Y_TRAIN){
    std::cout << "\n--- Training Gaussian Process ---\n" << std::endl;

    trained = true; 
    x_train = X_TRAIN; 
    y_train = Y_TRAIN;

    // compute covariance matrix
    kernelGP(x_train, x_train);
    Ky = Cov;
    // add noise to covariance matrix
    for (int i = 0; i < (*x_train).rows(); i++){
        Ky(i, i) += noise;
    }

    // alpha = Ky.llt().solve(*y_train);                 // eq. 2.25

    // double neg_log_likelihood = -0.5 * (*y_train).transpose() * alpha - Ky.llt().matrixL().diagonal().array().log().sum();

    // Compute Cholesky decomposition of Ky
    Eigen::LLT<Eigen::MatrixXd> lltOfKy(Ky);
    if (lltOfKy.info() == Eigen::NumericalIssue) {
        // Handle numerical issues with Cholesky decomposition
        // For example, matrix is not positive definite
    }

    // Solve for alpha using Cholesky factorization
    alpha = lltOfKy.solve(*y_train);
    L = lltOfKy.matrixL();

    // Compute negative log-likelihood
    neg_log_likelihood = -0.5 * (*y_train).transpose() * alpha - 0.5 * L.diagonal().array().log().sum();


    std::cout << "\nneg_log_likelihood: " << neg_log_likelihood << std::endl;
    std::cout << "\n--- Parameter Tuning Complete ---\n" << std::endl;
}



void GaussianProcess::predict(Eigen::MatrixXd* X_TEST, char save){
    
    
    // throw error if training has not occured
    if (!trained){
        throw std::invalid_argument("Error: train GP before predicting");
        return;
    }

    x_test = X_TEST;

    // initialize covariance sub matrices
    Eigen::MatrixXd Ks((*x_train).rows(), (*x_test).rows());          // ∈ ℝ (m x l)
    Eigen::MatrixXd Kss((*x_test).rows(), (*x_test).rows());          // ∈ ℝ (l x l))

    // compute required covariance matrices
    kernelGP(x_train, x_test);
    Ks = Cov;
    kernelGP(x_test, x_test);
    Kss = Cov;

    // compute mean y_test ∈ ℝ (l x m) -> see Algorithm 2.1 in Rasmussen & Williams
    Eigen::MatrixXd alpha = Ky.llt().solve(*y_train);                 // eq. 2.25
    y_test = Ks.transpose() * alpha;                                  // eq. 2.25

    // compute variance: V  ∈ ℝ (l x l)
    Eigen::MatrixXd V = Ky.llt().matrixL().solve(Ks);                 // eq. 2.26
    Cov = Kss - V.transpose() * V;                                    // eq. 2.26   

}

/* accessor functions */
std::string GaussianProcess::get_kernel() const {
    return kernel;
};

float GaussianProcess::get_lengthScale() const {
    return length;
}

float GaussianProcess::get_signalNoise() const {
    return sigma;
}

Eigen::MatrixXd& GaussianProcess::get_Cov(){
    return Cov;
};

Eigen::VectorXd& GaussianProcess::get_y_test(){
    return y_test;
};


