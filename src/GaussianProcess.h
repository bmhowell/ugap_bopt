// Copyright 2023 Brian Howell
// MIT License
// Project: BayesOpt

#ifndef SRC_GAUSSIANPROCESS_H_
#define SRC_GAUSSIANPROCESS_H_
#include "common.h"

class GaussianProcess {

private:
    // MEMBER VARIABLES
    std::string _kernel;                                   // covariance kernel specification
    std::string _file_path;                                // file path to store data
    
    bool _trained, _train_scaled, _val_scaled, _test_scaled;  // flag to indicate if GP has been trained

    // initialize covariance matrices
    //      - _l --> number of samples
    //      - m --> number of data points
    //      - n --> number of variables
    Eigen::MatrixXd _Cov;                                  // ∈ ℝ (m x m) | ∈ ℝ (m x _l) | ∈ ℝ (_l x _l) ⊂ conditionGP() | ⊂ unconditionedGP()
    Eigen::MatrixXd _Ky;                                   // ∈ ℝ (m x m) ⊂ train() | predict();

    // vector and matrix for cholesky decomposition
    Eigen::VectorXd _alpha;                                // ∈ ℝ (m)     ⊂ train() | predict() | compute_lml()
    Eigen::MatrixXd _L;                                    // ∈ ℝ (m x m) ⊂ train() | predict() | compute_lml()
    Eigen::MatrixXd _V;

    // data
    Eigen::MatrixXd _x_train; 
    Eigen::VectorXd _y_train;
    Eigen::VectorXd _y_train_std;

    Eigen::MatrixXd _x_val; 
    Eigen::VectorXd _y_val; 

    Eigen::MatrixXd _x_test; 
    Eigen::VectorXd _y_test;
    Eigen::VectorXd _y_test_std;

    // bayesian optimization output
    std::vector<int> _candidates; 

    // scaling
    Eigen::VectorXd _x_mean, _x_std;                        // ∈ ℝ (m)     ⊂ scale_data()
    double _y_mean, _y_std;                                 // ∈ ℝ         ⊂ scale_data()
    double _error_val;                                     // ∈ ℝ         ⊂ validate()
    
    // learned parameters
    double _l, _sf, _sn, _lml, _p;

    // MEMBER FUNCTIONS

    void kernelGP(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, 
                  double &length, double &sigma, double &p);
    /*  description:
     *      kernel construction currently equipped with the following kernels:
     *          - radial basis function --> "RBF"
     *
     *      - _l: number of test points
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
     *      - Mu: average predicted output ∈ ℝ (_l x m)
     *
    */
    
    /* scaling data */
    void scale_data(Eigen::MatrixXd &X_VAL,   Eigen::VectorXd &Y_VAL, bool VAL);

    void scale_data(Eigen::MatrixXd &X, Eigen::VectorXd &Y);

    void scale_data(Eigen::MatrixXd &X_TEST);

    void unscale_data(Eigen::VectorXd &Y_TEST);

    /* model selection - compute negative log likelihood */
    double compute_lml(double &length, double &sigma, double &noise, double &p);
    /* description:
        - choice of optimization method
        - model parameters: 
            - length scale: _l
            - signal variance: σ_f^2
            - noise parameter: σ_n^2
    */

    void sort_data(Eigen::MatrixXd &PARAM);
    /* 
        Implements sorting algorithm to rank top performers 
        for genetic algorithm. 
    */ 

    void gen_tune_param();
    /* 
        Genetic algorithm used for maximization of marginal log likelihood. 
    */ 



public:
    /* default constructor */
    GaussianProcess();

    /* overload constructor */
    GaussianProcess(std::string KERNEL, std::string FILE_PATH);

    /* destructor function */
    ~GaussianProcess();
           
    /* training functions*/
    void train(Eigen::MatrixXd &X_TRAIN, Eigen::VectorXd &Y_TRAIN);

    void train(Eigen::MatrixXd &X_TRAIN, Eigen::VectorXd &Y_TRAIN,
               std::vector<double> &model_param); 
    
    // void train(Eigen::MatrixXd &X_TRAIN, Eigen::VectorXd &Y_TRAIN,
    //            Eigen::MatrixXd &X_VAL,   Eigen::VectorXd &Y_VAL); 
    /* 
        Either: 
            - perform model selection
            - or use pre-defined model parameters       
    */

    /* validation */
    double validate(Eigen::MatrixXd &X_VAL, Eigen::VectorXd &Y_VAL);

    /* inference */
    void predict(Eigen::MatrixXd &X_TEST, bool compute_std = true);

    /*  Conditioning the GP:
     *
     *    - _l: number of sample points
     *    - m: number of training data points
     *    - n: number of variables
     *
     *  input data:
     *          - testX:     ∈ ℝ (_l x _l)         - number of variables
     *          - trainX:    ∈ ℝ (m x m)         - training data X
     *          - trainY:    ∈ ℝ (m)             - training data Y
     *          - save:      if 'y' -> save
     *  output data:
     *          - Mu:        ∈ ℝ (_l)             - mean output Y
     *          - K:         ∈ ℝ (_l x _l)         - covariance matrix
     *
     *  description:
     *          this function enables the user to condition the GP.
     *          the details of these equations can be found in Rasmussen page 16, equations 2.22-24:
     *
     *          mean:
     *              Bar* = K* @ [K + variance * I]^-1 @ _y_train
     *
     *          covariance:
     *              _Cov(f*): K** - K* @ [K + variance * I]^-1 @ K*
     *              ** Note: the covariance between the outputs is written as a function of inputs
     *                  -> cov(f(x_p), f(x_q)) = k(xp, xq) = exp(-1/2 * norm(x_p - x_q, 2)^2)
     *                  -> Rasmussen page 14, equation 2.16
     *
     *
     */


    /* accessor functions */

    Eigen::MatrixXd get_Cov();

    Eigen::VectorXd get_y_test();

    Eigen::VectorXd get_y_train_std(); 

    Eigen::VectorXd get_y_test_std();

    std::vector<int> get_candidates();

    double get_length_param();

    double get_sigma_param();
    
    double get_noise_param();
};

#endif // SRC_GAUSSIANPROCESS_H_
