#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H
#pragma once

#include <vector>
// #include <Eigen/Dense>  // Include necessary Eigen headers
#include "GaussianProcess.h"

// declare functions
int     find_arg_idx(int argc, char** argv, const char* option); 
double  gen_data(float tfinal, double dt, int node, int idsim, bopt &bopti, sim& simi, std::string file_path); 
void    bootstrap(sim &sim_settings, constraints &c, std::vector<bopt>* bopt, int num_sims, std::string file_path);
void    write_to_file(bopt &b, sim& sim_set, int id, std::string file_path); 
void    store_tot_data(std::vector<bopt>* bopti, sim& sim_set, int num_sims, std::string file_path); 
int     read_data(std::vector<bopt>* bopti, std::string file_path); 
void    to_eigen(std::vector<bopt>* data, Eigen::MatrixXd* X, Eigen::VectorXd* Y);
void    gen_test_points(constraints &c, Eigen::MatrixXd* X); 

#endif  // HELPER_FUNCTIONS_H