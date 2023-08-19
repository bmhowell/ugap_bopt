
#include "GaussianProcess.h"
// #include "AcquisitionFunction.h"

/* default constructor */
GaussianProcess::GaussianProcess() {

    // MEMBER VARIABLES
    m_l = 1.0f;                        // length scale parameter
    m_sf = 1.0f;                       // signal noise variance
    m_kernel = "RBF";                  // covariance kernel specification

    file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/materials_opt/output";
}

/* overload constructor */
GaussianProcess::GaussianProcess(float L, float SF, std::string KERNEL, std::string FILE_PATH){
    m_l = L;
    m_sf = SF;
    m_kernel = KERNEL;
    file_path = FILE_PATH; 
}

/* destructor function */
GaussianProcess::~GaussianProcess() {

}

/* infrastructure functions */
void GaussianProcess::kernelGP(Eigen::MatrixXd* X, Eigen::MatrixXd* Y){
    if (m_kernel == "RBF"){
        if (X->rows() != Y->rows()){
            // kernel construction algorithm for non-symmetric matrices
            Cov = Eigen::MatrixXd::Zero(X->rows(), Y->rows());
            for (int i = 0; i < X->rows(); i++){
                for (int j = 0; j < Y->rows(); j++){
                    Cov(i, j) = (m_sf * m_sf) * exp( -(X->row(i) - Y->row(j)).squaredNorm() / 2 / (m_l * m_l) );
                    if (i == j){
                        // add noise cholesky noise to diagonal elements to ensure non-singular
                        Cov(i, j) += 1e-8;
                    }
                }
            }
        } else{
            // kernel construction algorithm for symmetric matrices
            double cov_value;
            Cov = Eigen::MatrixXd::Zero(X->rows(), Y->rows());
            for (int i = 0; i < X->rows(); i++){
                for (int j = i; j < Y->rows(); j++){
                    cov_value = (m_sf * m_sf) * exp( -(X->row(i) - Y->row(j)).squaredNorm() / 2 / (m_l * m_l) );
                    Cov(i, j) = cov_value;
                    Cov(j, i) = cov_value;
                    if (i == j){
                        // add noise cholesky noise to diagonal elements to ensure non-singular
                        Cov(i, j) += 1e-8;
                    }
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
    x_train = X_TRAIN; 
    y_train = Y_TRAIN;

    // maximize the negative log-likehood using a genetic algorithm


}

void GaussianProcess::predict(Eigen::MatrixXd* X_TEST, char save){

    x_test = X_TEST; 

    // initialize covariance sub matrices                            
    Eigen::MatrixXd Ky((*x_train).rows(), (*x_train).rows());         // ∈ ℝ (m x m)
    Eigen::MatrixXd Ks((*x_train).rows(), (*x_test).rows());          // ∈ ℝ (m x l)
    Eigen::MatrixXd Kss((*x_test).rows(), (*x_test).rows());          // ∈ ℝ (l x l))

    // compute required covariance matrices
    kernelGP(x_train, x_train);
    Ky = Cov;
    kernelGP(x_train, x_test);
    Ks = Cov;
    kernelGP(x_test, x_test);
    Kss = Cov;


    // compute mean: Mu ∈ ℝ (l x m)
    Eigen::MatrixXd alpha;
    alpha = Ky.llt().solve(*y_train);
    y_test = Ks.transpose() * alpha;
    std::cout << "\ny_test: \n" << y_test << std::endl; 

    // compute variance: V  ∈ ℝ (l x l)
    Eigen::MatrixXd V = Ky.llt().matrixL().solve(Ks);
    Cov = Kss - V.transpose() * V;
}

/* accessor functions */
std::string GaussianProcess::get_kernel() const {
    return m_kernel;
};

float GaussianProcess::get_lengthScale() const {
    return m_l;
}

float GaussianProcess::get_signalNoise() const {
    return m_sf;
}

Eigen::MatrixXd& GaussianProcess::get_Cov(){
    return Cov;
};

Eigen::VectorXd& GaussianProcess::get_Mu(){
    return y_test;
};

/* mutator functions */
void GaussianProcess::set_lengthScale(float l) {
    m_l = l;
}

void GaussianProcess::set_signalNoise(float sf){
    m_sf = sf;
}

void GaussianProcess::set_kernel(std::string kernel) {
    m_kernel = kernel;
}
