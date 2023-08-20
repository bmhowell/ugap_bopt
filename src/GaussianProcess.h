#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <random>
#include <vector>
#include <stdexcept>
#define EIGEN_USE_BLAS
#include <Eigen/Dense>

#pragma once
#ifndef BAYESIANOPTIMISATIONCPP_GAUSSIANPROCESS_H
#define BAYESIANOPTIMISATIONCPP_GAUSSIANPROCESS_H


class GaussianProcess {

private:
    // MEMBER VARIABLES

    // parameters
    float length;                                         // length scale parameter
    float sigma;                                          // signal noise variance
    double noise;                                         // noise parameter

    std::string kernel;                                   // covariance kernel specification
    std::string file_path;                                // file path to store data
    
    bool trained;                                         // flag to indicate if GP has been trained

    // initialize covariance matrices
    //      - l --> number of samples
    //      - m --> number of data points
    //      - n --> number of variables
    Eigen::MatrixXd m_x_sample_distribution;              // ∈ ℝ (l x m) ⊂ generate_random_points()
    Eigen::VectorXd m_x_points;                           // ∈ ℝ (m x m) ⊂ unconditionedGP(): -> x_test
    Eigen::MatrixXd Cov;                                  // ∈ ℝ (m x m) | ∈ ℝ (m x l) | ∈ ℝ (l x l) ⊂ conditionGP() | ⊂ unconditionedGP()
    Eigen::MatrixXd Ky;                                   // ∈ ℝ (m x m) ⊂ train() | predict()

    Eigen::VectorXd alpha; 

    double neg_log_likelihood;                            // ∈ ℝ (1)     ⊂ train()

    // data
    Eigen::MatrixXd* x_train; 
    Eigen::VectorXd* y_train;
    Eigen::MatrixXd* x_test; 
    Eigen::VectorXd  y_test;                              // ∈ ℝ (m)


public:
    /* default constructor */
    GaussianProcess();

    /* overload constructor */
    GaussianProcess(float L, float SF, double N, std::string KERNEL, std::string FILE_PATH);

    /* destructor function */
    ~GaussianProcess();

    /* optimization functions */
    void kernelGP(Eigen::MatrixXd* X, Eigen::MatrixXd* Y);
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

    void generate_random_points(int num_sample, int x_size, float mean, float stddev, float scale);

    /* training */
    void train(Eigen::MatrixXd* X_TRAIN, Eigen::VectorXd* Y_TRAIN);
    /* Model selection
        automatic bias-variance trade off
            - model parameters: 
                - noise parameter: σ^2
                - length scale: l
        
        using a gentic algorithm
    */

    

    /* inference */
    void predict(Eigen::MatrixXd* X_TEST, char save);
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

    /* accessor functions */
    std::string get_kernel() const;

    float get_lengthScale() const;

    float get_signalNoise() const;

    Eigen::MatrixXd& get_Cov();

    Eigen::VectorXd& get_Mu();

    Eigen::VectorXd& get_y_test();

};

#endif //BAYESIANOPTIMISATIONCPP_GAUSSIANPROCESS_H
