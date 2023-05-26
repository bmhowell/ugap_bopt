
#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

#ifndef UGAPDIFFUSION_VOXEL_H
#define UGAPDIFFUSION_VOXEL_H


class Voxel {

private:
    // MEMBER VARIABLES

    // simulation parameters
    int    sim_id;                                          // |   ---   |  simulation ID
    float  I0;                                              // |  W/m^2  |  incident light intensity
    float  t_final;                                         // |    s    |  final simulation time
    double dt;                                              // |    s    |  time step discretization
    int    nodes;
    float  uvt;                                             // |    s    |  uv exposure time

    double timer;                                           // |    s    |  timer for simulation
    float total_time;                                       // | unitless|  total number of nodes
    float coord_map_const;

    double theta0;                                          // |    K    |  initial temperature
    int interfacial_nodes;                                  // |   ---   |  interfacial thickness parameter
    double len_block;                                       // |    m    |  sample length
    double h;                                               // |    m    |  spatial discretization
    int N_VOL_NODES;                                        // | unitless|  total number of nodes in RVE
    int N_PLANE_NODES;                                      // | unitless|  total number of nodes in plane of RVE

    // formulation - wt. percent
    float percent_PI;                                        // |   wt.%  | weight percent of photo initiator
    float percent_PEA;                                       // |   wt.%  | weight percent of PEA
    float percent_HDDA;                                      // |   wt.%  | weight percent of HDDA
    float percent_8025D;                                     // |   wt.%  | weight percent of 8025D
    float percent_8025E;                                     // |   wt.%  | weight percent of 8025E
    float percent_E4396;                                     // |   wt.%  | weight percent of HDDA
    float percent_M;                                         // |   wt.%  | weight percent of monomer

    // physical properties
    // densities and molecular weights
    int rho_PEA;                                              // | kg/m^3  | density of PEA (estimated)
    int rho_HDDA;                                             // | kg/m^3  | density of HDDA (estimated)
    int rho_E4396;                                            // | kg/m^3  | density of EBECRYL 4396
    float rho_M;                                              // | kg/m^3  | weighted average density of monomer
    float rho_P;                                              // | kg/m^3  | weighted average density of polymer
    int rho_UGAP;                                             // | kg/m^3  | estimated density of UGAP
    int rho_nacl;                                             // | kg/m^3  | estimated density of NaCl

    float mw_PEA;                                             // |  kg/mol | molecular weight of PEA
    float mw_HDDA;                                            // |  kg/mol | molecular weight of HDDA
    float mw_M;                                               // |  kg/mol | weighted average molecular weight of monomer
    float mw_PI;                                              // |  kg/mol | molecular weight of photo initiator
    float basis_wt;                                           // |   kg    | arbitrary starting ink weight
    float basis_vol;                                          // |   m^3   | arbitrary starting ink volume
    float mol_PI;                                             // |   mol   | required PI for basis weight
    float mol_M;                                              // |   mol   | required monomer for basis weight
    float c_M0;                                               // | mol/m^3 | inital concentration of monomer
    float c_PI0;                                              // | mol/m^3 | inital concentration of photoinitiator
    float c_NaCl;                                            // | mol/m^3 | concentration of NaCl

    // diffusion parameters
    double Dm0;                                               // |  m^2/s  | diffusion constant pre-exponential, monomer (taki lit.)
    float Am;                                                 // | unitless| diffusion constant parameter, monomer (shanna lit.)

    // bowman reaction parameters
    float Rg;                                                 // | J/mol K | universal gas constant
    float alpha_P;                                            // |   1/K   | coefficent of thermal expansion, polymerization (taki + bowman lit.)
    float alpha_M;                                            // |   1/K   | coefficent of thermal expansion, monomer (taki + bowman lit.)
    float theta_gP;                                           // |    K    | glass transition temperature, polymer UGAP (measured TgA)
    float theta_gM;                                           // |    K    | glass transition temperature, monomer (Taki lit.)
    float k_P0;                                               // |m^3/mol s| true kinetic constant, polymerization (taki lit.)
    float E_P;                                                // |  J/mol  | activation energy, polymerization (lit.)
    float A_Dp;                                               // | unitless| diffusion parameter, polymerization (lit.)
    float f_cp;                                               // | unitless| critical free volume, polymerization (lit.)
    float k_T0;                                               // |m^3/mol s| true kinetic constant, termination (taki lit.)
    float E_T;                                                // |  J/mol  | activation energy, termination (bowman lit.)
    float A_Dt;                                               // | unitless| activation energy, termination (taki lit.)
    float f_ct;                                               // | unitless| critical free volume, termination (taki lit.)
    float R_rd;                                               // |  1/mol  | reaction diffusion parameter (taki lit.)

    float k_I0;                                               // |  s^-1   | primary radical rate constant
    float A_I;                                                // | unitless| activation energy, initiation (bowman lit. 1)
    float f_ci;                                               // | unitless| critical free volume, initiation (bowman lit. 1)
    float E_I;                                                // |  J/mol  | activation energy, initiation (bowman lit. 1)

    // thermal properties
    float dHp;                                                // |  W/mol  | heat of polymerization of acrylate monomers
    int Cp_nacl;                                              // | J/kg/K  | heat capacity of NaCl
    float Cp_pea;                                             // | J/mol/K | heat capacity of PEA @ 298K - https://polymerdatabase.com/polymer%20physics/Cp%20Table.html
    float Cp_hdda;                                            // | J/mol/K | solid heat capacity of HDDA - https://webbook.nist.gov/cgi/cbook.cgi?ID=C629118&Units=SI&Mask=1F
    float K_thermal_nacl;

//    // SHANNA PARAMETERS
    int Cp_shanna;                                            // | J/kg/K  | shanna's heat capacity
    float K_thermal_shanna;                                   // | W/m/K   | shanna's thermal conductivity

    // photo initiator properties
    float eps;                                                // |m^3/mol m| initiator absorbtivity
    float eps_nacl;                                           // |m^3/mol m| NaCl absorbtivity
    float phi;                                                // | unitless| quantum yield inititation

    // numerical method parameters: backward euler
    float tol;
    int thresh;

    // initialize cube_coord, nonboundary_nodes and boundary_nodes
    int current_coords[3]; 

    // initialize material properties and uv energy
    std::vector<double> density, heat_capacity, therm_cond, f_free_volume, uv_values;


    // initialize spatial concentrations, temperature, and rate constants
    std::vector<double> c_PI, c_PIdot, c_Mdot, c_M;
    std::vector<double> theta;
    std::vector<double> k_t, k_p, k_i, diff_pdot, diff_mdot, diff_m, diff_theta;

    // vectors and arrays for particle generation
    std::vector<int> material_type;                             // 0-resin, 1-particle
    std::vector<int> particles_ind;                             // vector holding total indices for each particle
    std::vector<int> particle_interfacial_nodes;                // interfacial distance indices
    double interfacial_thick = 1.0;

    // solution vectors
    std::vector<double> total_time_steps;                   // time discretization
    std::vector<double> z_space;                            // spatial discretization


    // data outputs
    std::ofstream print_sim_config; 
    std::ofstream print_density;
    std::ofstream print_concentrations;
    std::ofstream print_avg_concentrations;

public:

    // output file path
    std::string file_path;                                  // set correct file path to output dependent on computer

    /* overload constructor */
    Voxel(float TF, double DT, int N, int IDSIM, double temp, float UVI, float UVT, std::string FILE_PATH);

    /* destructor */
    ~Voxel();

    // helper functions
    double SquaredDiff(double val_1, double val_2);
        // SquaredDiff - returns the squared difference between two values for l2 norm
    
    void UniqueVec(std::vector<int>& vec);
        // uniqueIntegers - returns a vector of unique integers

    void Node2Coord(int node, int (&coord)[3]); 
        // Mapping node number to (i, j, k) coordinates
    
    int Coord2Node(int (&coord)[3]);
        // Mapping (i, j, k) coordinates to node number

    // function declarations
    void ComputeParticles(double radius_1, double solids_loading);
        // ComputeParticles - adds particles to resin

    void ComputeRxnRateConstants();

    // equation 1
    double IRate(std::vector<double> &conc_PI, double I0, double z, int node) const;
        // PhotoinitiatorRate - right hand side of photoinitiator ODE


    // equation 2
    double PIdotRate(std::vector<double> &conc_PIdot,
                              std::vector<double> &conc_PI,
                              std::vector<double> &conc_M,
                              double I0, double z, int node);


    // equation 3
    double MdotRate(std::vector<double> &conc_Mdot,
                    std::vector<double> &conc_PIdot,
                    std::vector<double> &conc_M,
                    int node);


    // equation 4
    double MRate(std::vector<double> &conc_M,
                 std::vector<double> &conc_Mdot,
                 std::vector<double> &conc_PIdot,
                 int node);

    // equation 5
    double TempRate(std::vector<double>     &temperature,
                        std::vector<double> &conc_M,
                        std::vector<double> &conc_Mdot,
                        std::vector<double> &conc_PI,
                        std::vector<double> &conc_PIdot,
                        double intensity, int node);

    // solve system simultaneously
    void SolveSystem(std::vector<double> &c_PI_next,
                     std::vector<double> &c_PIdot_next,
                     std::vector<double> &c_Mdot_next,
                     std::vector<double> &c_M_next,
                     std::vector<double> &theta_next,
                     double I0, double dt, int method);

    // write solutions to files
    void Config2File(double dt); 

    void Density2File();

    void AvgConcentrations2File(int counter, 
                                std::vector<double> &c_PI_next,
                                std::vector<double> &c_PIdot_next,
                                std::vector<double> &c_Mdot_next,
                                std::vector<double> &c_M_next,
                                std::vector<double> &theta_next,
                                double time); 

    void Concentrations2File(int counter,
                             std::vector<double> &c_PI_next,
                             std::vector<double> &c_PIdot_next,
                             std::vector<double> &c_Mdot_next,
                             std::vector<double> &c_M_next,
                             std::vector<double> &theta_next,
                             double time);

    void NonBoundaries2File(int counter, 
                            std::vector<double> &c_PI_next,
                            std::vector<double> &c_PIdot_next,
                            std::vector<double> &c_Mdot_next,
                            std::vector<double> &c_M_next,
                            std::vector<double> &theta_next,
                            double time, 
                            int (&coords)[3]);


    void Simulate(int method, int save_voxel);
        // Simulate - runs simulation of UV curing kinetics

};


#endif //UGAPDIFFUSION_VOXEL_H
