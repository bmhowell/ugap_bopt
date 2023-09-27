#ifndef BAYESIANOPT_H
#define BAYESIANOPT_H
#include "common.h"
#include "GaussianProcess.h"

#include "Voxel.h"

class BayesianOpt {

private: 
    // MEMBER VARIABLES

    // opt constraints, sim settings, and GP model
    constraints     _c;
    sim             _s;
    GaussianProcess _model;
    std::string     _file_path;

    // gp model parameters
    std::vector<double> _model_params; 
    bool                _validate; 
    int                 _n_dim; 
    int                 _num_sample = 100; 
    int                 _num_evals; 
    

    // data 
    std::vector<bopt> _bopti;
    Eigen::MatrixXd* x_train     = new Eigen::MatrixXd;
    Eigen::VectorXd* y_train     = new Eigen::VectorXd;
    Eigen::VectorXd* y_train_std = new Eigen::VectorXd;

    Eigen::MatrixXd* x_val       = new Eigen::MatrixXd;
    Eigen::VectorXd* y_val       = new Eigen::VectorXd;

    Eigen::MatrixXd* x_test      = new Eigen::MatrixXd; 
    Eigen::VectorXd* y_test      = new Eigen::VectorXd;
    Eigen::VectorXd* y_test_std  = new Eigen::VectorXd;

    // sampling 
    Eigen::MatrixXd *x_sample      = new Eigen::MatrixXd;
    Eigen::VectorXd *y_sample_mean = new Eigen::VectorXd;
    Eigen::VectorXd *y_sample_std  = new Eigen::VectorXd; 
    Eigen::VectorXd *conf_bound    = new Eigen::VectorXd;

    // PRIVATE MEMBER FUNCTIONS
    void build_dataset(std::vector<bopt> &BOPTI, 
                       Eigen::MatrixXd   &X_TRAIN,
                       Eigen::VectorXd   &Y_TRAIN);

    void build_dataset(std::vector<bopt> &BOPTI,
                                Eigen::MatrixXd   &X_TRAIN, 
                                Eigen::VectorXd   &Y_TRAIN,
                                Eigen::MatrixXd   &X_VAL,   
                                Eigen::VectorXd   &Y_VAL, 
                                Eigen::MatrixXd   &X_TEST, 
                                Eigen::VectorXd   &Y_TEST);

    void gen_test_points(Eigen::MatrixXd &_x_sample); 

    void store_tot_data(std::vector<bopt> &BOPTI, int num_sims); 

public:

    // CONSTRUCTORS
    BayesianOpt();
    BayesianOpt(GaussianProcess &_model, 
                int             &_n_dim,
                constraints     &_c, 
                sim             &_s, 
                std::string     &_file_path);
    ~BayesianOpt();


    // PUBLIC MEMBER FUNCTIONS
    void load_data(std::vector<bopt> &BOPTI, bool validate); 
    
    void condition_model(bool pre_learned); 
    void condition_model(); 
    
    void evaluate_model(); 

    void sample_posterior(); 

    void qUCB(bool _lcb); 
    void qUCB(); 

    void evaluate_samples(); 

    void optimize();


}; 

#endif  //BAYESIANOPT_H