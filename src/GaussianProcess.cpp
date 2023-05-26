
#include "GaussianProcess.h"
// #include "AcquisitionFunction.h"

/* default constructor */
GaussianProcess::GaussianProcess() {

    // MEMBER VARIABLES
    m_l = 1.0f;                        // length scale parameter
    m_sf = 1.0f;                       // signal noise variance
    m_kernel = "RBF";                  // covariance kernel specification

}

/* overload constructor */
GaussianProcess::GaussianProcess(float l, float sf, std::string kernel){
    l = l;
    sf = sf;
    kernel = kernel;

    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "welcome to bhoptimisation.bopt!" << std::endl;
    std::cout << "     - For to produce an unconditioned GP plot, call the method 'unconditioned_GP(xSize_=50, nSamples_=5, plot_=1)'.";
    std::cout << std::endl;
    std::cout << "     - To condition a GP, call the method 'condition_GP()'." << std::endl;
    std::cout << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << std::endl;

}

/* destructor function */
GaussianProcess::~GaussianProcess() {

}

/* optimization functions */
void GaussianProcess::unconditionedGP(){

    // generate an initial sample vector of evenly spaced points between -10 and 10
    const int num_sample = 5;                                       // number of sample points
    const  int x_size = 25;                                        // number of x points \in [-10, 10]
    double delta = (10.0 - (-10.0)) / (x_size - 1);                 // increment step size

    // generate lin space between -10 and 10
    m_x_points = Eigen::VectorXd::Zero(x_size);
    for(int i = 0 ; i < x_size; i++){
        m_x_points(i) = (-10 + delta * i);
    }

    // define zero mean
    Eigen::VectorXd Mu = Eigen::VectorXd::Zero(x_size);             // ∈ ℝ (m)

    // construct RBF covariance kernel using Eigen::Matrix
    double cov_value;
    Eigen::MatrixXd Cov(x_size, x_size);                            // ∈ ℝ (m x m)
    for (int i = 0; i < m_x_points.size(); i++){
        for (int j = i; j < m_x_points.size(); j++){
            if (i != j){
                cov_value = (m_sf * m_sf) * exp(-pow(sqrt(pow((m_x_points[i] - m_x_points[j]), 2)), 2) / 2 / (m_l * m_l));
                Cov(i, j) = cov_value;
                Cov(j, i) = cov_value;
            } else{
                // add noise Cholesky Noise to diagonal elements to ensure non-singular
                Cov(i, j) = 1. + 1e-6;
            }
        }
    }

    // cholesky decomposition
    Eigen::MatrixXd L(x_size, x_size);
    L = Cov.llt().matrixL();

    // sample the distribution
    generate_random_points(num_sample, x_size, 0., 1., 1.);
    Eigen::MatrixXd f_sample(num_sample, x_size);
    f_sample = m_x_sample_distribution * L;
    for (short i = 0; i < num_sample; i++){
        f_sample.row(i) += Mu;
    }

    saveUnconditionedData(m_x_points, f_sample, Mu, Cov);
}

void GaussianProcess::kernelGP(Eigen::MatrixXd& X, Eigen::MatrixXd& Y){
    if (m_kernel == "RBF"){
        if (X.cols() != Y.cols()){
            // kernel construction algorithm for non-symmetric matrices
            m_Cov = Eigen::MatrixXd::Zero(X.cols(), Y.cols());
            for (int i = 0; i < X.cols(); i++){
                for (int j = 0; j < Y.cols(); j++){
                    m_Cov(i, j) = (m_sf * m_sf) * exp( -(X.col(i) - Y.col(j)).squaredNorm() / 2 / (m_l * m_l) );
                    if (i == j){
                        // add noise cholesky noise to diagonal elements to ensure non-singular
                        m_Cov(i, j) += 1e-6;
                    }
                }
            }
        } else{
            // kernel construction algorithm for symmetric matrices
            double cov_value;
            m_Cov = Eigen::MatrixXd::Zero(X.cols(), Y.cols());
            for (int i = 0; i < X.cols(); i++){
                for (int j = i; j < Y.cols(); j++){
                    cov_value = (m_sf * m_sf) * exp( -(X.col(i) - Y.col(j)).squaredNorm() / 2 / (m_l * m_l) );
                    m_Cov(i, j) = cov_value;
                    m_Cov(j, i) = cov_value;
                    if (i == j){
                        // add noise cholesky noise to diagonal elements to ensure non-singular
                        m_Cov(i, j) += 1e-6;
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

void GaussianProcess::predict(Eigen::MatrixXd& x_test, Eigen::MatrixXd& x_train, Eigen::VectorXd& y_train, char save, std::string file_path){

    std::cout << "\n--- GAUSSIAN PROCESS ---" << std::endl;
    std::cout << "x_test: \n" << x_test << std::endl;
    std::cout << "x_train: \n" << x_train << std::endl;
    std::cout << "y_train: \n" << y_train.transpose() << std::endl;

    // compute required covariance matrices
    Eigen::MatrixXd Ky(x_train.rows(), x_train.cols());         // ∈ ℝ (m x m)
    Eigen::MatrixXd Ks(x_train.rows(), x_test.cols());          // ∈ ℝ (m x l)
    Eigen::MatrixXd Kss(x_test.rows(), x_test.cols());          // ∈ ℝ (l x l))

    kernelGP(x_train, x_train);
    Ky = m_Cov;
    kernelGP(x_train, x_test);
    Ks = m_Cov;
    kernelGP(x_test, x_test);
    Kss = m_Cov;

    // METHOD 1
    std::chrono::steady_clock sc;
    auto start = sc.now();

    // compute mean: Mu ∈ ℝ (l x m)
    Eigen::MatrixXd alpha;
    alpha = Ky.llt().solve(y_train);
    m_mu = Ks.transpose() * alpha;

    // compute variance: V  ∈ ℝ (l x l)
    Eigen::MatrixXd V = Ky.llt().matrixL().solve(Ks);
    m_Cov = Kss - V.transpose() * V;

    // compute uncertainty ∈ ℝ (l)
    Eigen::VectorXd uncertainty(m_Cov.cols());
    for (unsigned int i = 0; i < m_Cov.cols(); ++i){
        uncertainty(i) = 2 * sqrt(m_Cov(i, i));
    }

    // display results
    std::cout << "\nx_test: \n" << x_test << std::endl;
    std::cout << "\nMu: \n" << m_mu.transpose() << std::endl;
    std::cout << "\nuncertainty = \n" << uncertainty.transpose() << std::endl;
    std::cout << "\nMu - uncertainty = \n" << (m_mu - uncertainty).transpose() << std::endl;
    std::cout << "\nMu + uncertainty = \n" << (m_mu + uncertainty).transpose() << std::endl;

    if (save == 'y'){
        std::cout << "--- saving prediction results --- " << std::endl;
        // save results
        std::ofstream save_x_test;
        std::ofstream save_mu;
        std::ofstream save_uncertainty;

        save_x_test.open(file_path); 
        save_mu.open(file_path);
        save_uncertainty.open(file_path);

        // save_x_test.open("/Users/brianhowell/Desktop/Berkeley/MSOL/BayesianOptimisationCPP/plots/data/conditionedGp/save_x_test.dat");
        // save_Mu.open("/Users/brianhowell/Desktop/Berkeley/MSOL/BayesianOptimisationCPP/plots/data/conditionedGp/save_Mu.dat");
        // save_uncertainty.open("/Users/brianhowell/Desktop/Berkeley/MSOL/BayesianOptimisationCPP/plots/data/conditionedGp/save_uncertainty.dat");

        for (unsigned int i = 0; i < x_test.rows(); ++i){
            for (unsigned int j = 0; j < x_test.cols(); ++j){
                if (j < x_test.cols() - 1){
                    save_x_test << x_test(i, j) << " ";
                } else{
                    save_x_test << x_test(i, j);
                }
            }
            save_x_test << std::endl;
        }

        for (short i = 0; i < m_mu.rows(); ++i){
            for(short j = 0; j < m_mu.cols(); ++j){
                if (i < m_mu.cols() - 1){
                    save_mu << m_mu(i, j) << " ";
                    save_uncertainty << uncertainty(i) << " ";
                } else{
                    save_mu << m_mu(i, j);
                    save_uncertainty << uncertainty(i);
                }
            }
        }
        save_x_test.close();
        save_mu.close();
        save_uncertainty.close();
    }

    auto end = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(end - start);
    std::cout << "\nMethod 1: " << time_span.count() << " s" << std::endl;
    std::cout << "\n----------------------------------------\n" << std::endl;

}

/* saving data functions */
void GaussianProcess::saveUnconditionedData(Eigen::VectorXd& X, Eigen::MatrixXd& Y, Eigen::VectorXd& Mu, Eigen::MatrixXd& Cov) {
    // initialize output data streams
    std::ofstream saveX;
    std::ofstream saveY;
    std::ofstream saveMu;
    std::ofstream saveCov;

    saveX.open("/Users/brianhowell/Desktop/Berkeley/MSOL/BayesianOptimisationCPP/plots/data/unconditionedGp/saveX.dat");
    saveY.open("/Users/brianhowell/Desktop/Berkeley/MSOL/BayesianOptimisationCPP/plots/data/unconditionedGP/saveY.dat");
    saveMu.open("/Users/brianhowell/Desktop/Berkeley/MSOL/BayesianOptimisationCPP/plots/data/unconditionedGP/saveMu.dat");
    saveCov.open("/Users/brianhowell/Desktop/Berkeley/MSOL/BayesianOptimisationCPP/plots/data/unconditionedGP/saveCov.dat");

    // save mean
    saveMu << Mu;
    saveMu.close();

    // save covariance
    for (unsigned int j = 0; j < Cov.cols(); j++){
        for (unsigned int i = 0; i < Cov.rows(); i++){
            if (i < Cov.rows()){
                saveCov << Cov(i, j) << " ";
            } else{
                saveCov << Cov(i, j);
            }
        }
        saveCov << std::endl;
    }
    saveCov.close();

    // save X data:
    saveX << X;
    saveX.close();

    // save Y data
    std::string yText = "y";
    for (unsigned int i = 0; i < Y.rows(); i++){
        if (i < Y.rows() - 1){
            saveY << yText + std::to_string(i + 1) << " ";
        } else{
            saveY << yText + std::to_string(i + 1);
        }
    }
    saveY << std::endl;

    for (unsigned int j = 0; j < Y.cols(); j++){
        for (unsigned int i = 0; i < Y.rows(); i++){
            if (i < Y.rows() - 1){
                saveY << Y(i, j) << " ";
            } else{
                saveY << Y(i, j);
            }
        }
        saveY << std::endl;
    }
    saveY.close();

    std::cout << "--- saving to file complete ---" << std::endl;

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
    return m_Cov;
};

Eigen::VectorXd& GaussianProcess::get_Mu(){
    return m_mu;
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
