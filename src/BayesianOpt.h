// Copyright 2023 Brian Howell
// MIT License
// Project: BayesOpt

#ifndef SRC_BAYESIANOPT_H_
#define SRC_BAYESIANOPT_H_
#include <string>
#include <vector>
#include "common.h"
#include "GaussianProcess.h"

#include "Voxel.h"

class BayesianOpt {
 public:
   // CONSTRUCTORS
    BayesianOpt();
    BayesianOpt(GaussianProcess &model,
                const int       &n_dim,
                constraints     &c,
                sim             &s,
                std::string     &file_path);
    ~BayesianOpt();
 
    // PUBLIC MEMBER FUNCTIONS
    void load_data(std::vector<bopt> &bopti, const bool validate);

    void condition_model(const bool pre_learned);
    void condition_model();

    void evaluate_model();

    void sample_posterior();

    void qUCB(const bool _lcb);
    void qUCB();

    void evaluate_samples();

    void optimize();

private:
    // MEMBER VARIABLES

    // opt constraints, sim settings, and GP model
    constraints         _c;
    sim                 _s;
    GaussianProcess     _model;
    std::string         _file_path;

    // gp model parameters
    std::vector<double> _model_params;
    bool                _validate;
    int                 _n_dim;
    int                 _num_sample = 100;
    int                 _num_evals;


    // data matrices
    std::vector<bopt> _bopti;
    Eigen::MatrixXd *_x_train;
    Eigen::VectorXd *_y_train;
    Eigen::VectorXd *_y_train_std;

    Eigen::MatrixXd *_x_val;
    Eigen::VectorXd *_y_val;

    Eigen::MatrixXd *_x_test;
    Eigen::VectorXd *_y_test;
    Eigen::VectorXd *_y_test_std;

    // sampling
    Eigen::MatrixXd *_x_smpl;
    Eigen::VectorXd *_y_sample_mean;
    Eigen::VectorXd *_y_sample_std;
    Eigen::VectorXd *_conf_bound;

    // top performers
    Eigen::MatrixXd *_x_top;
    Eigen::VectorXd *_y_top;

    // PRIVATE MEMBER FUNCTIONS
    void build_dataset(std::vector<bopt> &bopti,
                       Eigen::MatrixXd   &x_train,
                       Eigen::VectorXd   &y_train);

    void build_dataset(std::vector<bopt> &bopti,
                                Eigen::MatrixXd   &x_train,
                                Eigen::VectorXd   &y_train,
                                Eigen::MatrixXd   &x_val,
                                Eigen::VectorXd   &y_val);

    void gen_test_points(Eigen::MatrixXd &x_smpl);

    void store_tot_data(std::vector<bopt> &bopti, int num_sims);

    
};
#endif  // SRC_BAYESIANOPT_H_
