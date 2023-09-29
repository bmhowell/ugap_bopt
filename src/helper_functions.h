// Copyright 2023 Brian Howell
// MIT License
// Project: BayesOpt

#ifndef SRC_HELPER_FUNCTIONS_H_
#define SRC_HELPER_FUNCTIONS_H_
#include <vector>
#include <string>
#include "common.h"
#include "GaussianProcess.h"


// declare functions
int     find_arg_idx(int argc, char** argv, const char* option);

double  gen_data(float tfinal,
                 double dt,
                 int node,
                 int idsim,
                 bopt &bopti,
                 sim &simi,
                 std::string file_path,
                 bool multi_thread);

void    bootstrap(sim &sim_settings,
                  constraints &c,
                  std::vector<bopt> &bopt,
                  int num_sims,
                  std::string file_path,
                  bool multi_thread);

void    write_to_file(bopt &b,
                      sim &sim_set,
                      int id,
                      std::string file_path);

void    store_tot_data(std::vector<bopt> &bopti,
                       sim &sim_set,
                       int num_sims,
                       std::string file_path);

int     read_data(std::vector<bopt> &bopti,
                  std::string file_path); 

#endif  // SRC_HELPER_FUNCTIONS_H_
