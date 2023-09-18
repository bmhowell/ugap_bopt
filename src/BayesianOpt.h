#ifndef BAYESIANOPT_H
#define BAYESIANOPT_H
#include "common.h"
#include "GaussianProcess.h"

#include "Voxel.h"

class BayesianOpt {

private: 
    // MEMBER VARIABLES

public:

    // CONSTRUCTORS
    BayesianOpt();
    BayesianOpt(GaussianProcess &MODEL);
    ~BayesianOpt();


}; 

#endif  //BAYESIANOPT_H