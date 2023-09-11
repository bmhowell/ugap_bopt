#ifndef BAYESIANOPTIMISATIONCPP_GAUSSIANPROCESS_H
#define BAYESIANOPTIMISATIONCPP_GAUSSIANPROCESS_H
#include "common.h"

class GaussianProcess {

private:
    // MEMBER VARIABLES
    std::string kernel;                                   // covariance kernel specification
    std::string file_path;                                // file path to store data
    
    bool trained, train_scaled, val_scaled;               // flag to indicate if GP has been trained

    // initialize covariance matrices
    //      - l --> number of samples
    //      - m --> number of data points
    //      - n --> number of variables
    Eigen::MatrixXd Cov;                                  // ∈ ℝ (m x m) | ∈ ℝ (m x l) | ∈ ℝ (l x l) ⊂ conditionGP() | ⊂ unconditionedGP()
    Eigen::MatrixXd Ky;                                   // ∈ ℝ (m x m) ⊂ train() | predict();

    // vector and matrix for cholesky decomposition
    Eigen::VectorXd alpha;                                // ∈ ℝ (m)     ⊂ train() | predict()
    Eigen::MatrixXd L;                                    // ∈ ℝ (m x m) ⊂ train() | predict()

    // data
    Eigen::MatrixXd x_train; 
    Eigen::VectorXd y_train;
    Eigen::MatrixXd x_test; 
    Eigen::VectorXd y_test;
    Eigen::VectorXd y_test_std;
    Eigen::VectorXd y_test_u;
    Eigen::VectorXd y_test_l;  

    // scaling
    Eigen::VectorXd x_mean, x_std;                        // ∈ ℝ (m)     ⊂ scale_data()
    double y_mean, y_std;                                 // ∈ ℝ         ⊂ scale_data()
    double error_val;                                     // ∈ ℝ         ⊂ validate()
    
    // learned parameters
    double l, sf, sn, lml;



public:
    /* default constructor */
    GaussianProcess();

    /* overload constructor */
    GaussianProcess(std::string KERNEL, std::string FILE_PATH);

    /* destructor function */
    ~GaussianProcess();

    /* optimization functions */
    void kernelGP(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, double& length, double& sigma);
    /*  description:
     *      kernel construction currently equipped with the following kernels:
     *          - radial basis function --> "RBF"
     *
     *      - l: number of test points
     *      - m: number of data points
     *      - n: number of variables
     *
     *  input:
     *      - X: Dataset 1 ∈ ℝ (n x m)
     *
     *        [[X1_1, X1_2, ... X1_3],
     *         [X2_1, X2_2, ... X2_3],
     *         [X3_1, X3_2, ... X3_3]]
     *
     *      - Y: dataset 2 ∈ ℝ (n x m)
     *
     *         [[Y1_1, Y1_2, ... Y1_3],
     *          [Y2_1, Y2_2, ... Y2_3],
     *          [Y3_1, Y3_2, ... Y3_3]]
     *
     *  output:
     *      - K: covariance matrix ∈ ℝ (m x m)
     *      - Mu: average predicted output ∈ ℝ (l x m)
     *
    */
    
    /* scaling data */
    void scale_data(Eigen::MatrixXd& X, Eigen::VectorXd& Y);

    void scale_data(Eigen::MatrixXd& X_TEST);

    void unscale_data(Eigen::VectorXd& Y_TEST);

    /* model selection - compute negative log likelihood */
    double compute_lml(double& length, double& sigma, double& noise);
    /* description:
        - choice of optimization method
        - model parameters: 
            - length scale: l
            - signal variance: σ_f^2
            - noise parameter: σ_n^2
    */
           
    /* training */
    void train(Eigen::MatrixXd& X_TRAIN, Eigen::VectorXd& Y_TRAIN);

    void train(Eigen::MatrixXd& X_TRAIN, Eigen::VectorXd& Y_TRAIN, std::vector<double>& model_param); 
    /* 
        Either: 
            - perform model selection
            - or use pre-defined model parameters       
    */

    /* validation */
    void validate(Eigen::MatrixXd& X_VAL, Eigen::VectorXd& Y_VAL);

    /* inference */
    void predict(Eigen::MatrixXd& X_TEST);
    /*  Conditioning the GP:
     *
     *    - l: number of sample points
     *    - m: number of training data points
     *    - n: number of variables
     *
     *  input data:
     *          - testX:     ∈ ℝ (l x l)         - number of variables
     *          - trainX:    ∈ ℝ (m x m)         - training data X
     *          - trainY:    ∈ ℝ (m)             - training data Y
     *          - save:      if 'y' -> save
     *  output data:
     *          - Mu:        ∈ ℝ (l)             - mean output Y
     *          - K:         ∈ ℝ (l x l)         - covariance matrix
     *
     *  description:
     *          this function enables the user to condition the GP.
     *          the details of these equations can be found in Rasmussen page 16, equations 2.22-24:
     *
     *          mean:
     *              Bar* = K* @ [K + variance * I]^-1 @ y_train
     *
     *          covariance:
     *              Cov(f*): K** - K* @ [K + variance * I]^-1 @ K*
     *              ** Note: the covariance between the outputs is written as a function of inputs
     *                  -> cov(f(x_p), f(x_q)) = k(xp, xq) = exp(-1/2 * norm(x_p - x_q, 2)^2)
     *                  -> Rasmussen page 14, equation 2.16
     *
     *
     */

    void sort_data(Eigen::MatrixXd& PARAM);
    /* 
        Implements sorting algorithm to rank top performers 
        for genetic algorithm. 
    */ 

    void gen_opt(double& l, double& sf, double& sn);
    /* 
        Genetic algorithm used for maximization of marginal log likelihood. 
    */ 

    /* accessor functions */

    Eigen::MatrixXd get_Cov();

    Eigen::VectorXd get_y_test();

    Eigen::VectorXd get_y_test_std();

    Eigen::VectorXd get_y_test_u();

    Eigen::VectorXd get_y_test_l();


};

#endif //BAYESIANOPTIMISATIONCPP_GAUSSIANPROCESS_H
