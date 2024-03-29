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

    void qLCB(int iter);

    void evaluate_samples(int obj_fn, double wts[4]);

    void optimize(int obj_fn, double wts[4]);

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
    int                 _num_sample;
    int                 _num_evals;
    int                 _init_data_size;
    int                 _current_iter;


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

    // performance
    std::vector<double> _top_obj_pi;
    std::vector<double> _top_obj_pidot;
    std::vector<double> _top_obj_mdot;
    std::vector<double> _top_obj_m;
    std::vector<double> _top_obj;
    std::vector<double> _avg_obj;
    std::vector<double> _avg_top_obj;
    std::vector<double> _cost;

    // parameters over time
    std::vector<double> _tot_length;
    std::vector<double> _tot_sigma;
    std::vector<double> _tot_noise;
    std::vector<double> _tot_period;
    std::vector<double> _tot_alpha;
    std::vector<double> _tot_temp;
    std::vector<double> _tot_rp;
    std::vector<double> _tot_vp;
    std::vector<double> _tot_uvi;
    std::vector<double> _tot_uvt;


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

    void save_cost();

    void sort_data();

    double inv_decay_schdl(int iter);

    double exp_decay_schdl(int iter);
};
#endif  // SRC_BAYESIANOPT_H_
