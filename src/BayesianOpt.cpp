// Copyright 2023 Brian Howell
// MIT License
// Project: BayesOpt

#include <cmath>
#include <algorithm>
#include "Voxel.h"
#include "BayesianOpt.h"
#include "GaussianProcess.h"
#include "common.h"

/* default constructor */
BayesianOpt::BayesianOpt() {
    // default constructor
    constraints _c;
    sim _s;
    _s.time_stepping = 0;
    _s.update_time_stepping_values();

    // MACBOOK PRO
    _file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_opt/output_"
                + std::to_string(_s.time_stepping);
    // LINUX CENTRAL COMPUTING
    // _file_path = "/home/brian/Documents/berkeley/ugap_opt/output_"
    //             + std::to_string(_s.time_stepping);

    GaussianProcess _model = GaussianProcess("RBF", _file_path);

    // initialize data matrices
    _x_train       = new Eigen::MatrixXd;
    _y_train       = new Eigen::VectorXd;
    _y_train_std   = new Eigen::VectorXd;

    _x_val         = new Eigen::MatrixXd;
    _y_val         = new Eigen::VectorXd;

    _x_test        = new Eigen::MatrixXd;
    _y_test        = new Eigen::VectorXd;
    _y_test_std    = new Eigen::VectorXd;

    // sampling
    _x_smpl        = new Eigen::MatrixXd;
    _y_sample_mean = new Eigen::VectorXd;
    _y_sample_std  = new Eigen::VectorXd;
    _conf_bound    = new Eigen::VectorXd;
    
    // number of sample points for acq fn
    _num_sample = 250;

    // number of prev evals
    _init_data_size = 0;
    _current_iter   = 0;
}

/* overload constructor */
BayesianOpt::BayesianOpt(GaussianProcess &model,
                         const int       &n_dim,
                         constraints     &c,
                         sim             &s,
                         std::string     &file_path) {
    // overload constructor
    _n_dim     = n_dim;
    _c         = c;
    _s         = s;
    _model     = model;
    _file_path = file_path;

    // initialize data matrices
    _x_train       = new Eigen::MatrixXd;
    _y_train       = new Eigen::VectorXd;
    _y_train_std   = new Eigen::VectorXd;

    _x_val         = new Eigen::MatrixXd;
    _y_val         = new Eigen::VectorXd;

    _x_test        = new Eigen::MatrixXd;
    _y_test        = new Eigen::VectorXd;
    _y_test_std    = new Eigen::VectorXd;

    // sampling
    _x_smpl        = new Eigen::MatrixXd;
    _y_sample_mean = new Eigen::VectorXd;
    _y_sample_std  = new Eigen::VectorXd;
    _conf_bound    = new Eigen::VectorXd;

    // number of sample points for acq fn
    _num_sample = 250;

    // number of prev evals
    _init_data_size = 0;
    _current_iter   = 0;
}

/* destructor */
BayesianOpt::~BayesianOpt() {
    std::cout << "\n--- BayesianOpt Destroyed ---\n" << std::endl;
    // destructor

    // delete data matrices
    delete _x_train;
    delete _y_train;
    delete _y_train_std;

    delete _x_val;
    delete _y_val;
    delete _x_test;
    delete _y_test;
    delete _y_test_std;

    // sampling
    delete _x_smpl;
    delete _y_sample_mean;
    delete _y_sample_std;
    delete _conf_bound;
}

// PUBLIC MEMBER FUNCTIONS
void BayesianOpt::load_data(std::vector<bopt> &bopti, const bool validate) {
    _bopti          = bopti;
    _validate       = validate;
    _init_data_size = bopti.size();
    std::cout << "INIT DATA SIZE: " << _init_data_size << std::endl;

    if (validate) {
        // build dataset for training and testing
        this->build_dataset(bopti, *_x_train, *_y_train, *_x_val, *_y_val);
    } else {
        // build dataset for training only
        this->build_dataset(bopti, *_x_train, *_y_train);
    }
}

void BayesianOpt::condition_model(const bool pre_learned) {
    // ADD ASSERTION TO CHECK IF DATA IS LOADED
    if (pre_learned) {
        switch (_s.time_stepping) {
            case 0:
                // obj_0 -> 673.344
                _model_params = {0.99439, 0.356547, 0.000751229};
                break;
            case 1:
                // obj_1 -> 422.003
                _model_params = {0.994256, 0.623914, 0.000965578};
                break;
            case 2:
                // obj_2 -> 397.977
                _model_params = {0.940565, 0.708302, 0.000328992};
                break;
            case 3:
                // obj_3 -> 487.76
                _model_params = {0.956662, 0.78564, 0.00095118};
                break;
        }

        // train with maximized log likelihood
        _model.train(*_x_train, *_y_train, _model_params);
    } else {
        condition_model();
    }
}

void BayesianOpt::condition_model() {
    // train with maximized log likelihood
    _model.train(*_x_train, *_y_train);
}

void BayesianOpt::evaluate_model() {
    // use GP mvalidation to evaluate model
    if (_validate) {
        std::cout << "--- _model validatation ---" << std::endl;
        _model.validate(*_x_val, *_y_val);
    } else {
        throw std::invalid_argument("Error: validation data not selected");
    }
}

void BayesianOpt::sample_posterior() {
    // resize matrices
    _x_smpl        ->resize(_num_sample, _n_dim);
    _y_sample_mean ->resize(_num_sample);
    _y_sample_std  ->resize(_num_sample);
    _conf_bound    ->resize(_num_sample);

    // generate uniformly random points
    this->gen_test_points(*_x_smpl);

    // predict mean and std
    _model.predict(*_x_smpl, true);

    // store mean and std
    _y_sample_mean->array() = _model.get_y_test().array();
    _y_sample_std->array()  = _model.get_y_test_std().array();
}

void BayesianOpt::qLCB(int iter) {
    double beta = exp_decay_schdl(iter);
    // double beta = inv_decay_schdl(iter);
    std::cout << "beta: " << beta << std::endl;
    _conf_bound->array() = _y_sample_mean->array()
                         - beta * _y_sample_std->array();

    // get top candidates
    Eigen::VectorXi inds_s;
    inds_s = Eigen::VectorXi::LinSpaced(_conf_bound->size(),
                                        0,
                                        _conf_bound->size() - 1);

    std::sort(inds_s.data(), inds_s.data() + inds_s.size(),
             [this](int a, int b) {
                return (*_conf_bound)(a) < (*_conf_bound)(b);
                });

    // sort top samples
    *_conf_bound = (*_conf_bound)(inds_s);
    *_x_smpl     = (*_x_smpl)(inds_s, Eigen::all);
}

void BayesianOpt::evaluate_samples(int obj_fn) {
    // evaluate top candidates
    _num_evals = omp_get_num_procs();
    std::vector<bopt> voxels_evals;

    #pragma omp parallel for
    for (int id = 0; id < _num_evals; ++id) {
        // pull data from _x_smpl
        bopt b;
        b.temp = _x_smpl->coeff(id, 0);
        b.rp   = _x_smpl->coeff(id, 1);
        b.vp   = _x_smpl->coeff(id, 2);
        b.uvi  = _x_smpl->coeff(id, 3);
        b.uvt  = _x_smpl->coeff(id, 4);

        // init guess weights for multi-obj fn
        double default_weights[4] = {0.1, 0.25, 0.25, 0.4};
        double pareto_weights[4]  = {3.56574286e-09, 2.42560512e-03, 2.80839829e-01, 7.14916061e-01};
        // perform simulation with top candidates
        Voxel voxel_sim(_s.tfinal,
                        _s.dt,
                        _s.node,
                        id,
                        b.temp,
                        b.uvi,
                        b.uvt,
                        _file_path,
                        true);

        voxel_sim.computeParticles(b.rp,
                                   b.vp);
        voxel_sim.simulate(_s.method, _s.save_voxel, obj_fn, pareto_weights);

        b.obj_pi    = voxel_sim.getObjPI();
        b.obj_pidot = voxel_sim.getObjPIDot();
        b.obj_mdot  = voxel_sim.getObjMDot();
        b.obj_m     = voxel_sim.getObjM();

        b.obj       = voxel_sim.getObjective();
        
        #pragma omp critical
        {
            int thread_id = omp_get_thread_num();
            if (!std::isnan(b.obj)) {
                voxels_evals.push_back(b);
                _cost.push_back(b.obj);
                std::cout << "Thread " << thread_id << std::endl;
                std::cout << " | b.obj:   "  << b.obj       << std::endl
                          << " | b.pi:    "  << b.obj_pi    << std::endl
                          << " | b.pidot: "  << b.obj_pidot << std::endl
                          << " | b.mdot:  "  << b.obj_mdot  << std::endl
                          << " | b.m:     "  << b.obj_m     << std::endl
                          << " | b.temp:  "  << b.temp      << std::endl
                          << " | b.rp:    "  << b.rp        << std::endl
                          << " | b.vp:    "  << b.vp        << std::endl
                          << " | b.uvi:   "  << b.uvi       << std::endl
                          << " | b.uvt:   "  << b.uvt       << std::endl;
                std::cout << "-------------------\n" << std::endl;
            } else {
                std::cout << "\nWARNING: NaN detected\n" << std::endl;
                std::cout << "Thread " << thread_id << std::endl;
                std::cout << "-------------------\n" << std::endl;
            }
        }
    }

    std::cout << "--- finished new evaluations ---" << std::endl;

    // sort cost
    std::sort(_cost.begin(), _cost.end());

    // store top performers
    _top_obj.push_back(_cost[0]);

    // average top five performers and total performers

    _avg_top_obj.push_back(std::accumulate(_cost.begin(), _cost.begin() + 5, 0.0) / 5);
    _avg_obj.push_back(std::accumulate(_cost.begin(), _cost.end(), 0.0) / _cost.size());

    // concatenate data
    _bopti.insert(_bopti.end(), voxels_evals.begin(), voxels_evals.end());
    
    // find index associated with lowest cost for batch
    int ind;
    double min_cost = _bopti[0].obj;
    for (int i = _bopti.size()-omp_get_num_procs(); i < _bopti.size(); ++i) {
        if (_bopti[i].obj < min_cost) {
            min_cost = _bopti[i].obj;
            ind = i;
        }
    }

    // store GP parameters over time
    _tot_length.push_back(_model.get_length_param());
    _tot_sigma.push_back(_model.get_sigma_param());
    _tot_noise.push_back(_model.get_noise_param());
    
    // store best candidates over time
    _tot_temp.push_back(_bopti[ind].temp);
    _tot_rp.push_back(_bopti[ind].rp);
    _tot_vp.push_back(_bopti[ind].vp);
    _tot_uvi.push_back(_bopti[ind].uvi);
    _tot_uvt.push_back(_bopti[ind].uvt);

    // store each cost associated with best obj
    _top_obj_pi.push_back(_bopti[ind].obj_pi);
    _top_obj_pidot.push_back(_bopti[ind].obj_pidot);
    _top_obj_mdot.push_back(_bopti[ind].obj_mdot);
    _top_obj_m.push_back(_bopti[ind].obj_m);

}

void BayesianOpt::optimize(int obj_fn) {
    std::cout << "\n===== OPTIMIZING =====" << std::endl;
    // step 1: uniformly sample domain from gaussian process
    this->sample_posterior();
    
    // compute confidence bound (true indicates we are minimizing)
    this->qLCB(0);
    
    // evaluate top candidates
    this->evaluate_samples(obj_fn);

    // run loop
    for (int iter = 1; iter < 10; ++iter) {
        std::cout << "\n===== ITERATION " << iter << " =====" << std::endl;
        // update data
        this->build_dataset(_bopti, *_x_train, *_y_train);

        // train model on new data
        this->condition_model(false);

        // uniformly sample domain
        this->sample_posterior();

        // compute confidence bound
        this->qLCB(iter);

        // evaluate top candidates
        this->evaluate_samples(obj_fn);

        // print current best costs
        std::cout << "\ncurrent best costs: " << std::endl;
        for (int j = 0; j < 5; ++j) {
            std::cout << _cost[j] << ", ";
        }
        std::cout << std::endl;

        std::cout << "===== =========== =====" << std::endl;
    }

    // rebuild dataset
    this->build_dataset(_bopti, *_x_train, *_y_train);
    
    // save cost and sort data
    this->store_tot_data(_bopti, _bopti.size());
    this->save_cost();
    this->sort_data();
}

// PRIVATE MEMBER FUNCTIONS
void BayesianOpt::build_dataset(std::vector<bopt> &bopti,
                                Eigen::MatrixXd   &x_train,
                                Eigen::VectorXd   &y_train) {
    // build dataset for training only
    int num_data = bopti.size();

    // populate training and validation sets
    y_train.resize(num_data);
    x_train.resize(num_data, 5);
    for (int i = 0; i < num_data; ++i) {
        x_train(i, 0) = bopti[i].temp;
        x_train(i, 1) = bopti[i].rp;
        x_train(i, 2) = bopti[i].vp;
        x_train(i, 3) = bopti[i].uvi;
        x_train(i, 4) = bopti[i].uvt;
        y_train(i)    = bopti[i].obj;
        if (_current_iter == 0) {
            _cost.push_back(bopti[i].obj);
        }
    }

    if (_current_iter == 0) {
        // sort cost
        std::sort(_cost.begin(), _cost.end());

        // computer top, avg top, and avg cost
        _top_obj.push_back(_cost[0]);
        _avg_top_obj.push_back(std::accumulate(_cost.begin(), _cost.begin() + 5, 0.0) / 5);
        _avg_obj.push_back(std::accumulate(_cost.begin(), _cost.end(), 0.0) / _cost.size());
    }

    // increment current iteration 
    _current_iter++;
}

void BayesianOpt::build_dataset(std::vector<bopt> &bopti,
                                Eigen::MatrixXd   &x_train,
                                Eigen::VectorXd   &y_train,
                                Eigen::MatrixXd   &x_val,
                                Eigen::VectorXd   &y_val) {
    // build dataset for training and testing

    // split data into training and validation sets
    int num_data = bopti.size();
    int num_train = 0.8 * num_data;
    int num_val   = 0.1 * num_data;

    std::cout << "\n============== build dataset ==============" << std::endl;
    std::cout << "num_data: "  << num_data << std::endl;
    std::cout << "num_train: " << num_train << std::endl;
    std::cout << "num_val: "   << num_val << std::endl;
    std::cout << "===========================================\n" << std::endl;

    // resize x_train, y_train, x_val, and Y_VAL
    x_train.resize(num_train, 5);
    y_train.resize(num_train);
    x_val.resize(num_val, 5);
    y_val.resize(num_val);

    // shuffle dataset
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(bopti.begin(), bopti.end(), g);

    // populate training sets
    for (int i = 0; i < num_train; ++i) {
        x_train(i, 0) = bopti[i].temp;
        x_train(i, 1) = bopti[i].rp;
        x_train(i, 2) = bopti[i].vp;
        x_train(i, 3) = bopti[i].uvi;
        x_train(i, 4) = bopti[i].uvt;
        y_train(i)    = bopti[i].obj;
        if (_current_iter == 0) {
            _cost.push_back(bopti[i].obj);
        }
    }

    if (_current_iter == 0) {
        // sort cost
        std::sort(_cost.begin(), _cost.end());

        // computer top, avg top, and avg cost
        _top_obj.push_back(_cost[0]);
        _avg_top_obj.push_back(std::accumulate(_cost.begin(), _cost.begin() + 5, 0.0) / 5);
        _avg_obj.push_back(std::accumulate(_cost.begin(), _cost.end(), 0.0) / _cost.size());
    }

    // populate validation sets
    for (int i = 0; i < num_val; ++i) {
        x_val(i, 0) = bopti[i + num_train].temp;
        x_val(i, 1) = bopti[i + num_train].rp;
        x_val(i, 2) = bopti[i + num_train].vp;
        x_val(i, 3) = bopti[i + num_train].uvi;
        x_val(i, 4) = bopti[i + num_train].uvt;
        y_val(i)    = bopti[i + num_train].obj;
    }
    
    // increment iteration current
    _current_iter++;
}

void BayesianOpt::gen_test_points(Eigen::MatrixXd &x_smpl) {
    // initialize input variables
    std::random_device rd;                          // obtain a rnd num
    std::mt19937 gen(rd());                         // seed the generator
    std::uniform_real_distribution<> dis(0., 1.);   // define the range

    for (int ind = 0; ind < x_smpl.rows(); ++ind) {
        x_smpl(ind, 0) = ((_c).max_temp
                        - (_c).min_temp) * dis(gen)
                        + (_c).min_temp;
        x_smpl(ind, 1) = ((_c).max_rp
                        - (_c).min_rp)   * dis(gen)
                        + (_c).min_rp;
        x_smpl(ind, 2) = ((_c).max_vp
                        - (_c).min_vp)   * dis(gen)
                        + (_c).min_vp;
        x_smpl(ind, 3) = ((_c).max_uvi
                        - (_c).min_uvi)  * dis(gen)
                        + (_c).min_uvi;
        x_smpl(ind, 4) = ((_c).max_uvt
                        - (_c).min_uvt)  * dis(gen)
                        + (_c).min_uvt;
    }
}

void BayesianOpt::store_tot_data(std::vector<bopt> &bopti, int num_sims) {
    // convert bopti to Eigen matrix, sort according to _obj and save matrix
    Eigen::MatrixXd output;
    output.resize(num_sims, 10);
    for (int id = 0; id < num_sims; ++id) {
        output(id, 0) = bopti[id].temp;
        output(id, 1) = bopti[id].rp;
        output(id, 2) = bopti[id].vp;
        output(id, 3) = bopti[id].uvi;
        output(id, 4) = bopti[id].uvt;
        output(id, 5) = bopti[id].obj_pi;
        output(id, 6) = bopti[id].obj_pidot;
        output(id, 7) = bopti[id].obj_mdot;
        output(id, 8) = bopti[id].obj_m;
        output(id, 9) = bopti[id].obj;
    }

    // // sort rows of matrix according to last column
    // Eigen::VectorXi inds_s;
    // inds_s = Eigen::VectorXi::LinSpaced(num_sims,
    //                                     0,
    //                                     num_sims - 1);
    // std::sort(inds_s.data(), inds_s.data() + inds_s.size(),
    //         [&output](int a, int b) {
    //             std::cout << "output(a, 9): " << output(a, 9) << " | output(b, 9): " << output(b, 9) << std::endl;
    //             return output(a, 9) < output(b, 9);
    //             });

    // // sort matrix
    // output = output(inds_s, Eigen::all);

    
    std::cout << "\n--- storing data ---\n" << std::endl;
    std::ofstream my_file;
    my_file.open(_file_path + "/tot_bopt.txt");
    my_file << "temp, rp, vp, uvi, uvt, obj_pi, obj_pidot, obj_mdot, obj_m, obj" << std::endl;
    my_file << output.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
    my_file.close();
}

void BayesianOpt::save_cost() {
    std::ofstream my_file;

    // save best params over each iteration
    my_file.open(_file_path + "/bopt_params.txt");
    my_file << "length, " 
            << "sigma, "
            << "noise, "
            << "temp, "
            << "rp, "
            << "vp, "
            << "uvi, "
            << "uvt, "
            << "avg_obj, "
            << "avg_top_obj, "
            << "top_obj, "
            << "top_obj_pi, "
            << "top_obj_pidot, "
            << "top_obj_mdot, "
            << "top_obj_m" 
            << std::endl;
    for (int ind = 0; ind < _tot_length.size(); ++ind) {
        my_file << _tot_length[ind]     << ", "
                << _tot_sigma[ind]      << ", "
                << _tot_noise[ind]      << ", "
                << _tot_temp[ind]       << ", "
                << _tot_rp[ind]         << ", "
                << _tot_vp[ind]         << ", "
                << _tot_uvi[ind]        << ", "
                << _tot_uvt[ind]        << ", "
                << _avg_obj[ind]        << ", "
                << _avg_top_obj[ind]    << ", "
                << _top_obj[ind]        << ", "
                << _top_obj_pi[ind]     << ", "
                << _top_obj_pidot[ind]  << ", "
                << _top_obj_mdot[ind]   << ", "
                << _top_obj_m[ind]
                << std::endl;
    }
    my_file.close();
}

void BayesianOpt::sort_data() {
    // sort data
    Eigen::VectorXi inds_s;
    inds_s = Eigen::VectorXi::LinSpaced(_y_train->size(),
                                        0,
                                        _y_train->size() - 1);
    std::sort(inds_s.data(), inds_s.data() + inds_s.size(), 
            [this](int a, int b) {
                const double epsilon = 1e-6;
                return (*_y_train)(a) + epsilon < (*_y_train)(b);
        });

    // concatenate data for saving
    Eigen::MatrixXd tot_data(_x_train->rows(), _x_train->cols()+1);
    tot_data << (*_x_train)(inds_s, Eigen::all), (*_y_train)(inds_s);

    std::cout << "Final best values: " << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << " | b.obj:  " << tot_data.coeff(i, 5) << "\n"
                  << " | b.temp: " << tot_data.coeff(i, 0) << "\n"
                  << " | b.rp:   " << tot_data.coeff(i, 1) << "\n"
                  << " | b.vp:   " << tot_data.coeff(i, 2) << "\n"
                  << " | b.uvi:  " << tot_data.coeff(i, 3) << "\n"
                  << " | b.uvt:  " << tot_data.coeff(i, 4) << "\n"
                  << std::endl;
    }

    // save data
    std::ofstream my_file;
    my_file.open(_file_path + "/tot_data.txt");
    my_file << "temp, rp, vp, uvi, uvt, obj" << std::endl;
    my_file << tot_data;
    my_file.close();

}

double BayesianOpt::inv_decay_schdl(int iter) {
    // subtract of initial n data points
    // iter -= _init_data_size;

    return std::sqrt(1.96 / (1.0 + iter));
}

double BayesianOpt::exp_decay_schdl(int iter) {
    // subtract of initial n data points
    // iter -= _init_data_size;

    return std::sqrt(1.96*std::exp(-iter));
}


