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
    float m_l;                                            // length scale parameter
    float m_sf;                                           // signal noise variance
    std::string m_kernel;                                 // covariance kernel specification

    // initialize covariance matrices
    //      - l --> number of samples
    //      - m --> number of data points
    //      - n --> number of variables

    // ⊂ generate_random_points()
    Eigen::MatrixXd m_x_sample_distribution;              // ∈ ℝ (l x m)

    // ⊂ unconditionedGP(): -> x_test
    Eigen::VectorXd m_x_points;                           // ∈ ℝ (m x m)

    // ⊂ conditionGP() | ⊂ unconditionedGP()
    Eigen::MatrixXd m_Cov;                                // ∈ ℝ (m x m) | ∈ ℝ (m x l) | ∈ ℝ (l x l)

    // ⊂ conditionGP() | ⊂ unconditionedGP()
    Eigen::VectorXd m_mu;                                 // ∈ ℝ (m)


public:
    /* default constructor */
    GaussianProcess();

    /* overload constructor */
    GaussianProcess(float, float, std::string);

    /* destructor function */
    ~GaussianProcess();

    /* optimization functions */
    void unconditionedGP();
    /*
     *   This function samples from a multivariate distribution with zero mean
     *   and no data. The purpose of this function is to demonstrate a 1D plot
     *   of what an unconditioned GP is to look like.
     *
     */

    void kernelGP(Eigen::MatrixXd& X, Eigen::MatrixXd& Y);
    /*  description:
     *      kernel construction currently equipped with the following kernels:
     *          - radial basis function --> "RBF"
     *
     *      - l: number of sample points
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

    void predict(Eigen::MatrixXd& x_test, Eigen::MatrixXd& x_train, Eigen::VectorXd& y_train, char save, std::string file_path);
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


    /* saving data functions */
    void saveUnconditionedData(Eigen::VectorXd& X, Eigen::MatrixXd& Y,
                               Eigen::VectorXd& Mu, Eigen::MatrixXd& Cov);

    /* accessor functions */
    std::string get_kernel() const;

    float get_lengthScale() const;

    float get_signalNoise() const;

    Eigen::MatrixXd& get_Cov();

    Eigen::VectorXd& get_Mu();

    /* mutator functions  */
    void set_lengthScale(float l_);

    void set_signalNoise(float sf_);

    void set_kernel(std::string kernel_);

};

#endif //BAYESIANOPTIMISATIONCPP_GAUSSIANPROCESS_H
