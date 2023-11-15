// Copyright 2023 Brian Howell
// MIT License
// Project: BayesOpt

#ifndef UGAPDIFFUSION_VOXEL_H
#define UGAPDIFFUSION_VOXEL_H
#include "common.h"

class Voxel {

 private:
    // MEMBER VARIABLES

    // simulation parameters
    int    _sim_id;              // |   ---   |  simulation ID
    float  I0;                   // |  W/m^2  |  incident light intensity
    float  _t_final;             // |    s    |  final simulation time
    double _dt;                  // |    s    |  time step discretization
    int    _nodes;
    float  _uvt;                 // |    s    |  uv exposure time
    
    double _timer;               // |    s    |  timer for simulation
    float _coord_map_const;

    double _theta0;              // |    K    |  initial temperature
    int _interfacial_nodes;      // |   ---   |  interfacial thickness parameter
    double _len_block;           // |    m    |  sample length
    double _h;                   // |    m    |  spatial discretization
    int _n_vol_nodes;            // | unitless|  total number of nodes in RVE
    int _n_plane_nodes;          // | unitless|  total number of nodes in plane of RVE

    // formulation - wt. percent
    float _percent_PI;           // |   wt.%  | weight percent of photo initiator
    float _percent_PEA;          // |   wt.%  | weight percent of PEA
    float _percent_HDDA;         // |   wt.%  | weight percent of HDDA
    float _percent_8025D;        // |   wt.%  | weight percent of 8025D
    float _percent_8025E;        // |   wt.%  | weight percent of 8025E
    float _percent_E4396;        // |   wt.%  | weight percent of HDDA
    float _percent_M;            // |   wt.%  | weight percent of monomer

    // physical properties
    // densities and molecular weights
    int _rho_PEA;                // | kg/m^3  | density of PEA (estimated)
    int _rho_HDDA;               // | kg/m^3  | density of HDDA (estimated)
    int _rho_E4396;              // | kg/m^3  | density of EBECRYL 4396
    float _rho_M;                // | kg/m^3  | weighted average density of monomer
    float _rho_P;                // | kg/m^3  | weighted average density of polymer
    int _rho_UGAP;               // | kg/m^3  | estimated density of UGAP
    int _rho_nacl;               // | kg/m^3  | estimated density of NaCl

    float _mw_PEA;               // |  kg/mol | molecular weight of PEA
    float _mw_HDDA;              // |  kg/mol | molecular weight of HDDA
    float _mw_M;                 // |  kg/mol | weighted average molecular weight of monomer
    float _mw_PI;                // |  kg/mol | molecular weight of photo initiator
    float _basis_wt;             // |   kg    | arbitrary starting ink weight
    float _basis_vol;            // |   m^3   | arbitrary starting ink volume
    float _mol_PI;               // |   mol   | required PI for basis weight
    float _mol_M;                // |   mol   | required monomer for basis weight
    float _c_M0;                 // | mol/m^3 | inital concentration of monomer
    float _c_PI0;                // | mol/m^3 | inital concentration of photoinitiator
    float _c_NaCl;               // | mol/m^3 | concentration of NaCl

    // diffusion parameters
    double _Dm0;                 // |  m^2/s  | diffusion constant pre-exponential, monomer (taki lit.)
    float _Am;                   // | unitless| diffusion constant parameter, monomer (shanna lit.)

    // bowman reaction parameters
    float _Rg;                   // | J/mol K | universal gas constant
    float _alpha_P;              // |   1/K   | coefficent of thermal expansion, polymerization (taki + bowman lit.)
    float _alpha_M;              // |   1/K   | coefficent of thermal expansion, monomer (taki + bowman lit.)
    float _theta_gP;             // |    K    | glass transition temperature, polymer UGAP (measured TgA)
    float _theta_gM;             // |    K    | glass transition temperature, monomer (Taki lit.)
    float _k_P0;                 // |m^3/mol s| true kinetic constant, polymerization (taki lit.)
    float _E_P;                  // |  J/mol  | activation energy, polymerization (lit.)
    float _A_Dp;                 // | unitless| diffusion parameter, polymerization (lit.)
    float _f_cp;                 // | unitless| critical free volume, polymerization (lit.)
    float _k_T0;                 // |m^3/mol s| true kinetic constant, termination (taki lit.)
    float _E_T;                  // |  J/mol  | activation energy, termination (bowman lit.)
    float _A_Dt;                 // | unitless| activation energy, termination (taki lit.)
    float _f_ct;                 // | unitless| critical free volume, termination (taki lit.)
    float _R_rd;                 // |  1/mol  | reaction diffusion parameter (taki lit.)

    float _k_I0;                 // |  s^-1   | primary radical rate constant
    float _A_I;                  // | unitless| activation energy, initiation (bowman lit. 1)
    float _f_ci;                 // | unitless| critical free volume, initiation (bowman lit. 1)
    float _E_I;                  // |  J/mol  | activation energy, initiation (bowman lit. 1)

    // thermal properties
    float _dHp;                  // |  W/mol  | heat of polymerization of acrylate monomers
    int _Cp_nacl;                // | J/kg/K  | heat capacity of NaCl
    float _Cp_pea;               // | J/mol/K | heat capacity of PEA @ 298K - https://polymerdatabase.com/polymer%20physics/Cp%20Table.html
    float _Cp_hdda;              // | J/mol/K | solid heat capacity of HDDA - https://webbook.nist.gov/cgi/cbook.cgi?ID=C629118&Units=SI&Mask=1F
    float _K_thermal_nacl;

    // SHANNA PARAMETERS
    int _Cp_shanna;             // | J/kg/K  | shanna's heat capacity
    float _K_thermal_shanna;    // | W/m/K   | shanna's thermal conductivity

    // photo initiator properties
    float _eps;                 // |m^3/mol m| initiator absorbtivity
    float _eps_nacl;            // |m^3/mol m| NaCl absorbtivity
    float _phi;                 // | unitless| quantum yield inititation

    // numerical method parameters: backward euler
    float _tol;
    int _thresh;

    // initialize cube_coord, nonboundary_nodes and boundary_nodes
    int _current_coords[3]; 

    // initialize material properties and uv energy
    std::vector<double> _density, _heat_capacity, _therm_cond;
    std::vector<double> _f_free_volume, _uv_values;

    // initialize spatial concentrations, temperature, and rate constants
    std::vector<double> _c_PI, _c_PIdot, _c_Mdot, _c_M;
    std::vector<double> _theta;
    std::vector<double> _k_t, _k_p, _k_i, _diff_pdot, _diff_mdot, _diff_m, _diff_theta;

    // vectors and arrays for particle generation
    std::vector<int> _material_type;               // 0-resin, 1-particle
    std::vector<int> _particles_ind;               // vector holding total indices for each particle
    std::vector<int> _particle_interfacial_nodes;  // interfacial distance indices
    double _interfacial_thick = 1.0;

    // solution vectors
    std::vector<double> _total_time_steps;         // time discretization
    std::vector<double> _z_space;                  // spatial discretization

    // track average quantities
    std::vector<double> _c_PI_avg;
    std::vector<double> _c_PIdot_avg;
    std::vector<double> _c_Mdot_avg;
    std::vector<double> _c_M_avg;
    std::vector<double> _theta_avg;
    
    // data outputs
    std::ofstream _print_sim_config; 
    std::ofstream _print_density;
    std::ofstream _print_concentrations;
    std::ofstream _print_avg_concentrations;

    // optimization objective
    double    _obj;                                // |   ---   |  total obj function
    double    _obj_PI;                             // |   ---   |  PI obj function
    double    _obj_PIdot;                          // |   ---   |  PIdot obj function
    double    _obj_Mdot;                           // |   ---   |  Mdot obj function
    double    _obj_M;                              // |   ---   |  M obj function
    double    _obj_default;                        // |   ---   |  default total obj function

    // process parameters
    double    _vp;                                 // |   ---   |  volume fraction of particles
    double    _rp;                                 // |    m    |  radius of particles
    
    // output file path
    std::string _file_path;                        // set correct file path to output dependent on computer
    bool      _multi_thread; 


 public:
    
    /* overload constructor */
    Voxel(float tf,
          double dt,
          int n,
          int idsim,
          double temp,
          float uvi,
          float uvt,
          std::string file_path,
          bool multi_thread);

    /* destructor */
    ~Voxel();

    // helper functions
    double squaredDiff(double val_1, double val_2);
        // SquaredDiff - returns the squared difference between two values for l2 norm
    
    void uniqueVec(std::vector<int>& vec);
        // uniqueIntegers - returns a vector of unique integers

    void node2Coord(int node, int (&coord)[3]); 
        // Mapping node number to (i, j, k) coordinates
    
    int coord2Node(int (&coord)[3]);
        // Mapping (i, j, k) coordinates to node number

    // function declarations
    void computeParticles(double radius_1, double solids_loading);
        // ComputeParticles - adds particles to resin

    void computeRxnRateConstants();

    // equation 1
    double iRate(std::vector<double> &conc_PI, double I0, double z, int node) const;
        // PhotoinitiatorRate - right hand side of photoinitiator ODE


    // equation 2
    double piDotRate(std::vector<double> &conc_PIdot,
                              std::vector<double> &conc_PI,
                              std::vector<double> &conc_M,
                              double I0, double z, int node);


    // equation 3
    double mDotRate(std::vector<double> &conc_Mdot,
                    std::vector<double> &conc_PIdot,
                    std::vector<double> &conc_M,
                    int node);


    // equation 4
    double mRate(std::vector<double> &conc_M,
                 std::vector<double> &conc_Mdot,
                 std::vector<double> &conc_PIdot,
                 int node);

    // equation 5
    double tempRate(std::vector<double>     &temperature,
                        std::vector<double> &conc_M,
                        std::vector<double> &conc_Mdot,
                        std::vector<double> &conc_PI,
                        std::vector<double> &conc_PIdot,
                        double intensity, int node);

    // solve system simultaneously
    void solveSystem(std::vector<double> &c_PI_next,
                     std::vector<double> &c_PIdot_next,
                     std::vector<double> &c_Mdot_next,
                     std::vector<double> &c_M_next,
                     std::vector<double> &theta_next,
                     double I0, double dt, int method);

    // write solutions to files
    void config2File(double dt); 

    void density2File();

    void avgConcentrations2File(int counter, 
                                std::vector<double> &c_PI_next,
                                std::vector<double> &c_PIdot_next,
                                std::vector<double> &c_Mdot_next,
                                std::vector<double> &c_M_next,
                                std::vector<double> &theta_next,
                                double time); 

    void concentrations2File(int counter,
                             std::vector<double> &c_PI_next,
                             std::vector<double> &c_PIdot_next,
                             std::vector<double> &c_Mdot_next,
                             std::vector<double> &c_M_next,
                             std::vector<double> &theta_next,
                             double time);

    void nonBoundaries2File(int counter, 
                            std::vector<double> &c_PI_next,
                            std::vector<double> &c_PIdot_next,
                            std::vector<double> &c_Mdot_next,
                            std::vector<double> &c_M_next,
                            std::vector<double> &theta_next,
                            double time, 
                            int (&coords)[3]);


    void simulate(int method, int save_voxel, int obj_fn, double w[4]);
        // Simulate - runs simulation of UV curing kinetics

    double getObjPI();

    double getObjPIDot();

    double getObjMDot();

    double getObjM();

    double getObjective();

};


#endif //UGAPDIFFUSION_VOXEL_H
