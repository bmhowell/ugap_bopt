#ifndef BAYESIANOPT_H
#define BAYESIANOPT_H
#include "common.h"
#include "GaussianProcess.h"

#include "Voxel.h"

class BayesianOpt {

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
    Eigen::MatrixXd *_x_sample;
    Eigen::VectorXd *_y_sample_mean;
    Eigen::VectorXd *_y_sample_std;
    Eigen::VectorXd *_conf_bound;

    // PRIVATE MEMBER FUNCTIONS
    void build_dataset(std::vector<bopt> &bopti, 
                       Eigen::MatrixXd   &x_train,
                       Eigen::VectorXd   &y_train);

    void build_dataset(std::vector<bopt> &bopti,
                                Eigen::MatrixXd   &x_train, 
                                Eigen::VectorXd   &y_train,
                                Eigen::MatrixXd   &x_val,   
                                Eigen::VectorXd   &y_val);

    void gen_test_points(Eigen::MatrixXd &_x_sample); 

    void store_tot_data(std::vector<bopt> &bopti, int num_sims); 

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
    void load_data(std::vector<bopt> &bopti, bool validate); 
    
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