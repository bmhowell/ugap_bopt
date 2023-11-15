// Copyright 2023 Brian Howell
// MIT License
// Project: BayesOpt

#include "Voxel.h"

Voxel::Voxel(float tf,
             double dt,
             int n,
             int idsim,
             double temp,
             float uvi,
             float uvt,
             std::string file_path,
             bool MULTI_THREAD){

    // MEMBER VARIABLES
    // representative volume element RVE simulation parameters
    _t_final = tf;                                               // |    s    |  final time
    _dt      = dt;                                               // |    s    |  time step
    _nodes   = n;                                                // | unitless|  total number of nodes
    _sim_id  = idsim;                                            // |   ---   |  simulation id
    _theta0  = temp;                                             // |    K    | initial and ambient temperature
    I0      = uvi;                                               // |  W/m^2  |  UV intensity
    _uvt     = uvt;                                              // |    s    | uv exposure time
    _obj     = 1000.;                                            // |   ---   |  objective function

    _multi_thread = MULTI_THREAD;
    
    // set file path
    _file_path = file_path;                                      // |   ---   |  file path

    _interfacial_nodes = 1;                                      // |   ---   |  interfacial thickness parameter
    _len_block         = 0.00084;                                // |    m    |  sample length
    _h                 = (double) _len_block / (_nodes - 1);     // |    m    |  spatial discretization
    _n_vol_nodes       = _nodes * _nodes * _nodes;               // | unitless|  total number of nodes in RVE
    _n_plane_nodes     = _nodes * _nodes;                        // | unitless|  total number of nodes in plane of RVE
    _coord_map_const   = _len_block / _nodes; 

    // formulation - wt. percent
    _percent_PI    = 0.0333;                                     // |   wt.%  | weight percent of photo initiator
    _percent_PEA   = 0.15;                                       // |   wt.%  | weight percent of PEA
    _percent_HDDA  = 0.0168;                                     // |   wt.%  | weight percent of HDDA
    _percent_8025D = 0.4084;                                     // |   wt.%  | weight percent of 8025D
    _percent_8025E = 0.0408;                                     // |   wt.%  | weight percent of 8025E
    _percent_E4396 = 0.3507;                                     // |   wt.%  | weight percent of HDDA
    _percent_M = _percent_PEA + _percent_HDDA;                   // |   wt.%  | weight percent of monomer

    // physical properties
    // densities and molecular weights
    _rho_PEA = 1020;                                             // | kg/m^3  | _density of PEA (estimated)
    _rho_HDDA = 1020;                                            // | kg/m^3  | _density of HDDA (estimated)
    _rho_E4396 = 1100;                                           // | kg/m^3  | _density of EBECRYL 4396
    _rho_M = 0.899 * _rho_PEA + 0.101 * _rho_HDDA;               // | kg/m^3  | weighted average _density of monomer
    _rho_P = 0.03 * _rho_HDDA
            + 0.29 * _rho_PEA
            + 0.68 * _rho_E4396;                                 // | kg/m^3  | weighted average _density of polymer
    _rho_UGAP = 1840;                                            // | kg/m^3  | estimated _density of UGAP
    _rho_nacl = 2170;                                            // | kg/m^3  | estimated _density of NaCl

    _mw_PEA = 0.19221;                                           // |  kg/mol | molecular weight of PEA
    _mw_HDDA = 0.226;                                            // |  kg/mol | molecular weight of HDDA
    _mw_M = 0.899 * _mw_PEA + 0.101 * _mw_HDDA;                  // |  kg/mol | weighted average molecular weight of monomer
    _mw_PI = 0.4185;                                             // |  kg/mol | molecular weight of photo initiator

    _basis_wt = 0.5;                                             // |   kg    | arbitrary starting ink weight
    _basis_vol = _basis_wt / _rho_UGAP;                          // |   m^3   | arbitrary starting ink volume
    _mol_PI = _basis_wt * _percent_PI / _mw_PI;                  // |   mol   | required PI for basis weight
    _mol_M = _basis_wt * _percent_M / _mw_M;                     // |   mol   | required monomer for basis weight
    _c_M0 = _mol_M / _basis_vol;                                 // | mol/m^3 | initial concentration of monomer
    _c_PI0 = _mol_PI / _basis_vol;                               // | mol/m^3 | initial concentration of photoinitiator
    _c_NaCl = 37241.4;                                           // | mol/m^3 | concentration of NaCl

    // diffusion properties
    _Dm0 = 2.36e-6;                                              // |  m^2/s  | diffusion c pre-exponential, monomer (shanna lit.)
    _Am = 0.66;                                                  // | unitless| diffusion constant parameter, monomer (bowman lit.)

    // bowman reaction parameters
    _Rg       = 8.3145;                                          // | J/mol K | universal gas constant
    _alpha_P  = 0.000075;                                        // |   1/K   | coefficent of thermal expansion, polymerization (taki + bowman lit.)
    _alpha_M  = 0.0005;                                          // |   1/K   | coefficent of thermal expansion, monomer (taki + bowman lit.)
    _theta_gP = 236.75;                                          // |    K    | glass transition temperature, polymer UGAP (DMA MEASURED)
    _theta_gM = 213.0;                                           // |    K    | glass transition temperature, monomer (bowman)

    _k_P0 = 1.6e3;                                               // |m^3/mol s| true kinetic constant, polymerization (bowman lit) / 
    _E_P  = 18.23e3;                                             // |  J/mol  | activation energy, polymerization (bowman lit. 1) /
    _A_Dp = 0.66;                                                // | unitless| diffusion parameter, polymerization (bowman lit.)
    _f_cp = 4.2e-2;                                              // | unitless| critical free volume, polymerization (SHANNA lit.)
    
    _k_T0 = 3.6e3;                                               // |m^3/mol s| true kinetic constant, termination (shanna lit.)/ 
    _E_T  = 2.94e3;                                              // |  J/mol  | activation energy, termination (bowman lit.)
    _A_Dt = 1.2;                                                 // | unitless| activation energy, termination (bowman lit. MODIFIED?)
    _f_ct = 6.0e-2;                                              // | unitless| critical free volume, termination (bowman lit.) / 
    _R_rd = 11.;                                                 // |m^3/mol  | reaction diffusion parameter (taki lit.)

    _k_I0 = 5.1e-4;                                              // |m^3/mol s| primary radical rate constant (TRIAL/ERROR METHOD) / 
    _E_I  = 18.23e3;                                             // |  J/mol  | activation energy, initiation (bowman lit.)
    _A_I  = 0.66;                                                // | unitless| diffusion parameter, initiation (bowman lit.)
    _f_ci = 0.042;                                               // | unitless| critical free volume, initiation (bowman lit.)

    // thermal properties
    _dHp            = 5.48e4;                                    // |  J/mol  | heat of polymerization of acrylate monomers (bowman lit.)
    _Cp_nacl        = 880;                                       // | J/kg/K  | heat capacity of NaCl
    _Cp_pea         = 180.3;                                     // | J/mol/K | heat capacity of PEA @ 298K - https://polymerdatabase.com/polymer%20physics/Cp%20Table.html
    _Cp_pea        /= _mw_PEA;                                   // | J/kg/K  | convert units
    _Cp_hdda        = 202.9;                                     // | J/mol/K | solid heat capacity of HDDA - https://webbook.nist.gov/cgi/cbook.cgi?ID=C629118&Units=SI&Mask=1F
    _Cp_hdda       /= _mw_HDDA;                                  // | J/kg/K  | convert units
    _K_thermal_nacl = 0.069;                                     // | W/m/K   | thermal conductivity

    _Cp_shanna        = 1700;                                    // | J/kg/K  | shanna's heat capacity
    _K_thermal_shanna = 0.2;                                     // | W/m/K   | shanna's thermal conductivity

    // photo initiator properties
    _eps      = 9.66e-1;                                         // |m^3/mol m| initiator absorbtivity
    _eps_nacl = 7e-4;                                            // |   1/m   | NaCl absorbtivity
    _phi      = 0.6;                                             // | unitless| quantum yield inititation

    // numerical method parameters: backward euler
    _tol = 5e-2;
    _thresh = 50;

    // spatial discretization -> [0., h, 2*h, ..., L]
    double z_increment = 0.0;
    for (int i=0; i<_nodes; i++){
        _z_space.push_back(z_increment);
        z_increment += _h;
    }

    /* INITIALIZE VOXEL VALUES */
    // UV gradient
    std::fill_n(std::back_inserter(_uv_values),     _n_vol_nodes, 0.);

    // material properties
    std::fill_n(std::back_inserter(_density),       _n_vol_nodes, _rho_UGAP);
    std::fill_n(std::back_inserter(_heat_capacity), _n_vol_nodes, _Cp_shanna);
    std::fill_n(std::back_inserter(_therm_cond),    _n_vol_nodes, _K_thermal_shanna);
    std::fill_n(std::back_inserter(_material_type), _n_vol_nodes, 1);
    std::fill_n(std::back_inserter(_f_free_volume), _n_vol_nodes, 0.);

    // concentrations
    std::fill_n(std::back_inserter(_c_PI),          _n_vol_nodes, _c_PI0);
    std::fill_n(std::back_inserter(_c_M),           _n_vol_nodes, _c_M0);
    std::fill_n(std::back_inserter(_c_PIdot),       _n_vol_nodes, 0.);
    std::fill_n(std::back_inserter(_c_Mdot),        _n_vol_nodes, 0.);
    
    // diffusion values
    std::fill_n(std::back_inserter(_diff_pdot),     _n_vol_nodes, 0.);
    std::fill_n(std::back_inserter(_diff_mdot),     _n_vol_nodes, 0.);
    std::fill_n(std::back_inserter(_diff_m)   ,     _n_vol_nodes, 0.);
    std::fill_n(std::back_inserter(_diff_theta),    _n_vol_nodes, 0.);

    // temperature
    std::fill_n(std::back_inserter(_theta),         _n_vol_nodes, _theta0);

    // rate constants
    std::fill_n(std::back_inserter(_k_t),           _n_vol_nodes, _k_T0);
    std::fill_n(std::back_inserter(_k_p),           _n_vol_nodes, _k_P0);
    std::fill_n(std::back_inserter(_k_i),           _n_vol_nodes, _k_I0);

    if (!_multi_thread){
        std::cout << "==================================" << std::endl;
        std::cout << "Initial concentrations (mol/m^3): " << std::endl;
        std::cout << "_c_M0: "  << _c_M0   << std::endl;
        std::cout << "_c_PI0: " << _c_PI0  << std::endl;
        std::cout << "_theta0 " << _theta0 << std::endl;
        std::cout << "==================================\n" << std::endl;
    }

    // initialize system
    _current_coords[0] = 0;
    _current_coords[1] = 0;
    _current_coords[2] = 0;

}

// Destructor
Voxel::~Voxel() {
    std::cout << "Voxel destructor called" << std::endl;
}

// helper functions
double Voxel::squaredDiff(double val_1, double val_2){
    return (val_1 - val_2) * (val_1 - val_2);
}


void Voxel::uniqueVec(std::vector<int>& vec) {
    std::sort(vec.begin(), vec.end());
    auto last = std::unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());
}


void Voxel::node2Coord(int node, int (&coord)[3]){
    /*
        Given a 3D cube of size m by n by p, and integer coordinates (i, j, k)
        where 0 <= i < m, 0 <= j < n, and 0 <= k < p, the grid number can be 
        calculated as follows:
    */
    coord[2] = node % _nodes;    // compute k
    int temp = node / _nodes; 
    coord[1] = temp % _nodes;    // compute j
    coord[0] = temp / _nodes;    // compute i
    
}


int Voxel::coord2Node(int (&coord)[3]){
    /*
        Given a 3D cube of size m by n by p, and integer coordinates (i, j, k)
        where 0 <= i < m, 0 <= j < n, and 0 <= k < p, the grid number can be 
        calculated as follows:
    */
    return coord[0] * _nodes * _nodes + coord[1] * _nodes + coord[2];
}


void Voxel::computeParticles(double radius_1, double solids_loading) {
    
    // total number of host nodes for particles
    double n_particle_nodes = std::round(_n_vol_nodes * solids_loading);
    double particle_distance, node_particle, rand_loc;
    int node;
    _rp = radius_1;
    _vp = solids_loading; 

    if (!_multi_thread){
        std::cout << "\n--------- ------- ---------" << std::endl;
        std::cout << "--- GENERATING PARTICLES ---" << std::endl;
        std::cout << "--------- ------- ---------" << std::endl;
    }
    
    int counter1 = 0;
    int particle_coords[3] = {0, 0, 0}; 
    int tot_part_nodes = 0; 
    while ((tot_part_nodes < n_particle_nodes) && (counter1 < 10000000)){
        // step 1: choose random node as particle
        std::random_device rd{};
        std::mt19937  gen{rd()};


        std::uniform_real_distribution<> dist{0, 1};
        rand_loc = dist(gen);

        // step 2: generate random seed location for particle
        node_particle = ceil(_n_vol_nodes * rand_loc) - 1;
        this->node2Coord(node_particle, particle_coords);
        
        // step 3: find nodes that are within the distance of the seed location
        for (int node = 0; node < _n_vol_nodes; node++){
            this->node2Coord(node, _current_coords);
            particle_distance = sqrt(   squaredDiff(_current_coords[0] * _coord_map_const, particle_coords[0] * _coord_map_const)
                                      + squaredDiff(_current_coords[1] * _coord_map_const, particle_coords[1] * _coord_map_const)
                                      + squaredDiff(_current_coords[2] * _coord_map_const, particle_coords[2] * _coord_map_const) );

            // check if node is the particle radius range
            if (particle_distance <= radius_1){
                _particles_ind.push_back(node);
            }

            // check if node is in interfacial region (between resin and particle)
            else if (_interfacial_thick != 0 and particle_distance <= radius_1 + _interfacial_thick * _h){
                _particle_interfacial_nodes.push_back(node);
            }

            else{
                continue;
            }

        }

        // ensure vectors contain no duplicated nodes
        this->uniqueVec(_particles_ind);
        this->uniqueVec(_particle_interfacial_nodes);

        // assign interfacial material properties
        for (int i = 0; i < _particle_interfacial_nodes.size(); i++){
            _material_type[_particle_interfacial_nodes[i]]    = 2;

            // thermal properties
            _density[_particle_interfacial_nodes[i]]          = (_rho_nacl + _rho_UGAP) / 2;
            _heat_capacity[_particle_interfacial_nodes[i]]    = (_Cp_nacl + _Cp_shanna) / 2;
            _therm_cond[_particle_interfacial_nodes[i]]       = (_K_thermal_nacl + _K_thermal_shanna) / 2;

            // reaction properties
            _k_t[_particle_interfacial_nodes[i]]              = _k_T0;
            _k_p[_particle_interfacial_nodes[i]]              = _k_P0;
            _c_PI[_particle_interfacial_nodes[i]]             = _c_PI0 / 2;
            _c_PIdot[_particle_interfacial_nodes[i]]          = 0.;
            _c_Mdot[_particle_interfacial_nodes[i]]           = 0.;
            _c_M[_particle_interfacial_nodes[i]]              = _c_M0 / 2;
        }

        // assign particle material properties
        for (int i = 0; i < _particles_ind.size(); i++){

            // assign type particle
            _material_type[_particles_ind[i]]                 = 0;

            // thermal properties
            _density[_particles_ind[i]]                       = _rho_nacl;
            _heat_capacity[_particles_ind[i]]                 = _Cp_nacl;
            _therm_cond[_particles_ind[i]]                    = _K_thermal_nacl;

            // reaction properties
            _k_t[_particles_ind[i]]                           = 0.;
            _k_p[_particles_ind[i]]                           = 0.;
            _c_PI[_particles_ind[i]]                          = 0.;
            _c_PIdot[_particles_ind[i]]                       = 0.;
            _c_Mdot[_particles_ind[i]]                        = 0.;
            _c_M[_particles_ind[i]]                           = 0.;
        }


        // update total number of particle nodes
        tot_part_nodes = _particles_ind.size() + _particle_interfacial_nodes.size() / 2;  

        counter1++;
        if (counter1 >= 10000000){
            std::cout << "--- PARTICLE ITERATION THRESHOLD ---" << std::endl;
        }
        if (tot_part_nodes >= n_particle_nodes && !_multi_thread){
            std::cout << "_n_vol_nodes: "    << _n_vol_nodes                          << std::endl;
            std::cout << "tot_part_nodes: " << tot_part_nodes                       << std::endl;
            std::cout << "solids loading: " << (1.0 * tot_part_nodes / _n_vol_nodes) << std::endl;
        }
    }

    if (!_multi_thread){
        std::cout << "-------------------------------------------"                          << std::endl;
        std::cout << "number of particles generated: "  << counter1                         << std::endl;
        std::cout << "solids loading: "                 << n_particle_nodes / _n_vol_nodes   << std::endl;
        std::cout << "n_particle_nodes: "               << n_particle_nodes                 << std::endl;
        std::cout << "_particles_ind.size(): "           << _particles_ind.size()             << std::endl;
        std::cout << "-------------------------------------------"                          << std::endl;
    }
}


void Voxel::computeRxnRateConstants() {
    /*
     *   @updateVec - updates the reaction rate constants:
     *                          - _f_free_volume
     *                          - _k_p
     *                          - _k_t
     *                for every voxel.
     */

    // initialize intermediate values
    double vT, phi_M, phi_P, k_tr, denom;

    // loop through all voxels
    for (int i=0; i<_n_vol_nodes; i++){

        // compute reaction rate constants for resin
        if (_material_type[i] != 0){

            // bowman (1) equation 20
            vT = _c_M[i]*_mw_M/_rho_M + (_c_M0-_c_M[i])*_mw_M/_rho_P;

            // bowman (1) equation 22
            phi_M = _c_M[i]*_mw_M/_rho_M/vT;

            // bowman (1) equation 23
            phi_P = (_c_M0-_c_M[i])*_mw_M/_rho_P/vT;

            // bowman (1) equation 24
            _f_free_volume[i] = 0.025 + _alpha_M*phi_M*(_theta[i]-_theta_gM) + _alpha_P*phi_P*(_theta[i]-_theta_gP);

            // compute temperature dependent rate constants | bowman (1) equation 17
            _k_p[i] = _k_P0*exp(-_E_P / _Rg / _theta[i]) / (1 + exp(_A_Dp * (1/_f_free_volume[i] - 1/_f_cp)));
            _k_i[i] = _k_I0; 

            // bowman (1) equation 18
            k_tr   = _R_rd * _k_p[i] * _c_M[i];
            denom  = (k_tr / (_k_T0*exp(-_E_T/_Rg/_theta[i])) + exp(-_A_Dt*(1/_f_free_volume[i] - 1/_f_ct)));
            _k_t[i] = _k_T0*exp(-_E_T/_Rg/_theta[i]) / (1+1/denom);
            

        }else{
            // compute reaction rate constants for particles
            _f_free_volume[i] = 0.;
            _k_t[i]           = 0.;
            _k_p[i]           = 0.;
            _k_i[i]           = 0.;
        };
    }
}


double Voxel::iRate(std::vector<double> &conc_PI, double I0, double z, int node) const {
    if (_material_type[node] == 1 && _timer < _uvt){
        // material is ugap
        // return (-_phi * _eps * I0 * conc_PI[node] * exp( -_eps*conc_PI[node]*z) / 2);
        return (-_phi * _eps * I0 * conc_PI[node] * exp( -_eps*conc_PI[node]*z) / 2);
    }
    
    else if (_material_type[node] == 2 && _timer < _uvt){
        // material is interfacial
        return (-_phi * _eps * I0 * conc_PI[node] * exp( -_eps*conc_PI[node]*z) / 2);
    }
    else{
        // material is particle or exposure time is past
        return 0.;
    }

}


double Voxel::piDotRate(std::vector<double> &conc_PIdot,
                                 std::vector<double> &conc_PI,
                                 std::vector<double> &conc_M,
                                 double I0, double z, int node){
    /*          
    
        equation 2
        d[PI]/_dt = _phi*_eps*[PI]I - k[PIdot][M]

    */
    if (_material_type[node] == 1){
        // material is ugap
        double diffuse, Dm_avg;
        double diffusivity[6]; 

        // compute average diffusivity values for each first order derivative
        diffusivity[0] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node + 1])             + exp(-_Am / _f_free_volume[node]));
        diffusivity[1] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node - 1])             + exp(-_Am / _f_free_volume[node]));
        diffusivity[2] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node + _nodes])         + exp(-_Am / _f_free_volume[node]));
        diffusivity[3] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node - _nodes])         + exp(-_Am / _f_free_volume[node]));
        diffusivity[4] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node + _n_plane_nodes]) + exp(-_Am / _f_free_volume[node]));
        diffusivity[5] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node - _n_plane_nodes]) + exp(-_Am / _f_free_volume[node]));

       // compute chemical diffusion as a function of variable diffusivity
        double denom = 1 / _h / _h; 

        diffuse = (  diffusivity[0] * (conc_PI[node+1]               - conc_PI[node])
                   - diffusivity[1] * (conc_PI[node]                 - conc_PI[node - 1])
                   + diffusivity[2] * (conc_PI[node + _nodes]         - conc_PI[node])
                   - diffusivity[3] * (conc_PI[node]                 - conc_PI[node - _nodes])
                   + diffusivity[4] * (conc_PI[node + _n_plane_nodes] - conc_PI[node])
                   - diffusivity[5] * (conc_PI[node]                 - conc_PI[node - _n_plane_nodes])
                   ) * denom;


        _diff_pdot[node] = diffuse; 

        return (_phi*_eps*I0*conc_PI[node]*exp(-_eps*conc_PI[node]*z) - _k_i[node]*conc_PIdot[node]*conc_M[node] + diffuse);
    }
    
    else if (_material_type[node] == 2){
        // material is interface
        return (_phi*_eps*I0*conc_PI[node]*exp(-_eps*conc_PI[node]*z) - _k_i[node]*conc_PIdot[node]*conc_M[node]);
    }

    else{
        // materials is particle
        return 0.;
    }

}


// equation 3: d[Mdot]/_dt = k[PIdot][M] + ∇(k∇_x [Mdot]) - kt [Mdot]^2
double Voxel::mDotRate(std::vector<double> &conc_Mdot,
                              std::vector<double> &conc_PIdot,
                              std::vector<double> &conc_M,
                              int node) {
    // if material is ugap resin
    if (_material_type[node] == 1){
        // material is resin
        double term1, term2, term3, Dm_avg;
        double diffusivity[6]; 
        term1 = _k_i[node]*_c_PIdot[node]*_c_M[node];
        term2 = _k_t[node]*conc_Mdot[node]*conc_Mdot[node];

        // compute average diffusivity values for each first order derivative
        diffusivity[0] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node + 1])             + exp(-_Am / _f_free_volume[node]));
        diffusivity[1] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node - 1])             + exp(-_Am / _f_free_volume[node]));
        diffusivity[2] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node + _nodes])         + exp(-_Am / _f_free_volume[node]));
        diffusivity[3] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node - _nodes])         + exp(-_Am / _f_free_volume[node]));
        diffusivity[4] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node + _n_plane_nodes]) + exp(-_Am / _f_free_volume[node]));
        diffusivity[5] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node - _n_plane_nodes]) + exp(-_Am / _f_free_volume[node]));

        // compute chemical diffusion as a function of variable diffusivity
        double denom = 1 / _h / _h; 
        term3 = (   diffusivity[0] * (conc_Mdot[node+1]       - conc_Mdot[node])
                  - diffusivity[1] * (conc_Mdot[node]         - conc_Mdot[node - 1])
                  + diffusivity[2] * (conc_Mdot[node + _nodes] - conc_Mdot[node])
                  - diffusivity[3] * (conc_Mdot[node] - conc_Mdot[node - _nodes])
                  + diffusivity[4] * (conc_Mdot[node + _n_plane_nodes] - conc_Mdot[node])
                  - diffusivity[5] * (conc_Mdot[node] - conc_Mdot[node - _n_plane_nodes])
                  ) * denom; 

        _diff_mdot[node] = term3;
        return (term1 - term2 + term3);
    } 

    // material is interface 
    else if (_material_type[node] == 2){
        return _k_i[node]*_c_PIdot[node]*_c_M[node] - _k_t[node]*conc_Mdot[node]*conc_Mdot[node]; 
    }
    
    // material is a particle
    else{
        return 0.;
    }
}


// equation 4: d[M]/_dt = ∇(k∇_x [M]) - k[PI][M] - k[M][Mdot]   
double Voxel::mRate(std::vector<double> &conc_M,
                                  std::vector<double> &conc_Mdot,
                                  std::vector<double> &conc_PIdot,
                                  int node){

    // if material is ugap resin 
    if (_material_type[node] == 1){
        // material is resin
        double diffuse, consume, Dm_avg;
        double diffusivity[6]; 

        consume =   (_k_p[node]*conc_Mdot[node]*conc_M[node])
                  + (_k_i[node]*conc_PIdot[node]*conc_M[node]);

        // compute average diffusivity values for each first order derivative
        diffusivity[0] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node + 1])             + exp(-_Am / _f_free_volume[node]));
        diffusivity[1] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node - 1])             + exp(-_Am / _f_free_volume[node]));
        diffusivity[2] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node + _nodes])         + exp(-_Am / _f_free_volume[node]));
        diffusivity[3] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node - _nodes])         + exp(-_Am / _f_free_volume[node]));
        diffusivity[4] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node + _n_plane_nodes]) + exp(-_Am / _f_free_volume[node]));
        diffusivity[5] = 0.5 * _Dm0 * (exp(-_Am / _f_free_volume[node - _n_plane_nodes]) + exp(-_Am / _f_free_volume[node]));

        // compute chemical diffusion taking into account the average diffusivity values
        double denom = 1 / _h / _h; 
        diffuse = (  diffusivity[0]*(conc_M[node+1]-conc_M[node]) 
                   - diffusivity[1]*(conc_M[node]-conc_M[node-1])
                   + diffusivity[2]*(conc_M[node+_nodes]-conc_M[node])
                   - diffusivity[3]*(conc_M[node]-conc_M[node-_nodes])
                   + diffusivity[4]*(conc_M[node+_n_plane_nodes]-conc_M[node])
                   - diffusivity[5]*(conc_M[node]-conc_M[node-_n_plane_nodes])
                   ) * denom;
        
        // average diffusion coefficient
        Dm_avg = _Dm0 * (  exp(-_Am / _f_free_volume[node - _n_plane_nodes])
                        + exp(-_Am / _f_free_volume[node - _nodes])
                        + exp(-_Am / _f_free_volume[node - 1])
                        + 6 * exp(-_Am / _f_free_volume[node])
                        + exp(-_Am / _f_free_volume[node + _n_plane_nodes])
                        + exp(-_Am / _f_free_volume[node + _nodes])
                        + exp(-_Am / _f_free_volume[node + 1])
                        ) / 12.; 
        diffuse = Dm_avg
                  * (            conc_M[node - _n_plane_nodes]
                           +     conc_M[node - _nodes]
                           +     conc_M[node - 1]
                           - 6 * conc_M[node]
                           +     conc_M[node + _n_plane_nodes]
                           +     conc_M[node + _nodes]
                           +     conc_M[node + 1]
                  ) / _h / _h;
        _diff_m[node] = diffuse;
        return (diffuse - consume);
    }

    // material is interface - no diffusion, return only consumptions
    else if (_material_type[node] == 2){
        return -(_k_p[node]*conc_Mdot[node]*conc_M[node]) - (_k_i[node]*conc_PIdot[node]*conc_M[node]);
    }

    // material is particle - no reacton
    else{
        return 0.;
    }
}

// equation 5: dθ/_dt = ( ∇_x·(K ∇_x θ) + _k_p [M] [M_n·] ΔH + ε [PI] I ) / ρ / C 
double Voxel::tempRate(std::vector<double> &temperature,
                       std::vector<double> &conc_M,
                       std::vector<double> &conc_Mdot,
                       std::vector<double> &conc_PI,
                       std::vector<double> &conc_PIdot,
                       double intensity, int node){

    double heat_diffuse, heat_rxn, heat_uv;
    double therm_cond_avg[6]; 

    // compute average thermal conductivity values for each first order derivative
    therm_cond_avg[0] = 0.5 * (_therm_cond[node+1]             + _therm_cond[node]);
    therm_cond_avg[1] = 0.5 * (_therm_cond[node-1]             + _therm_cond[node]);
    therm_cond_avg[2] = 0.5 * (_therm_cond[node+_nodes]         + _therm_cond[node]);
    therm_cond_avg[3] = 0.5 * (_therm_cond[node-_nodes]         + _therm_cond[node]);
    therm_cond_avg[4] = 0.5 * (_therm_cond[node+_n_plane_nodes] + _therm_cond[node]);
    therm_cond_avg[5] = 0.5 * (_therm_cond[node-_n_plane_nodes] + _therm_cond[node]);

    // compute thermal diffusion taking into account the average thermal properties
    double denom = 1 / _h / _h / _density[node] / _heat_capacity[node];
    heat_diffuse = (  therm_cond_avg[0]*(temperature[node+1]-temperature[node]) 
                    - therm_cond_avg[1]*(temperature[node]-temperature[node-1])
                    + therm_cond_avg[2]*(temperature[node+_nodes]-temperature[node])
                    - therm_cond_avg[3]*(temperature[node]-temperature[node-_nodes])
                    + therm_cond_avg[4]*(temperature[node+_n_plane_nodes]-temperature[node])
                    - therm_cond_avg[5]*(temperature[node]-temperature[node-_n_plane_nodes])
                    ) * denom; 
    
    _diff_theta[node] = heat_diffuse;

    // compute the heat release by all exothermic (bond formation) reactions
    heat_rxn = (  _k_i[node] * conc_PIdot[node] * conc_M[node]
                + _k_p[node] * conc_Mdot[node]  * conc_M[node]
                + _k_t[node] * conc_Mdot[node]  * conc_Mdot[node]
                ) * _dHp;

    // material is resin
    if (_material_type[node] == 1 && _timer < _uvt){
        heat_uv = _eps
                * intensity
                * conc_PI[node]
                * exp(  -_eps*conc_PI[node]*(_len_block-_current_coords[2]*_coord_map_const)  );
    }

    // material is interface
    else if (_material_type[node] == 2 && _timer < _uvt){
        heat_uv = (   _eps
                    * intensity
                    * conc_PI[node]
                    * exp(  -_eps*conc_PI[node]*(_len_block-_current_coords[2]*_coord_map_const)  )
                    + _eps_nacl
                    * intensity
                    * exp(  -_eps_nacl*(_len_block-_current_coords[2]*_coord_map_const)  )
                    ) / 2;
    }

    // material is particle
    else if (_material_type[node] == 0 && _timer < _uvt){
        heat_uv = _eps_nacl
                * intensity
                * exp(  -_eps_nacl*(_len_block-_current_coords[2]*_coord_map_const)  );
    }
    
    // exposure time is up
    else{
        heat_uv = 0;
    }

    return heat_diffuse + (heat_rxn + heat_uv) / _density[node] / _heat_capacity[node];
};


void Voxel::solveSystem(std::vector<double> &c_PI_next,
                        std::vector<double> &c_PIdot_next,
                        std::vector<double> &c_Mdot_next,
                        std::vector<double> &c_M_next,
                        std::vector<double> &theta_next,
                        double I0, double _dt, int method){
    double heat_diffuse, heat_rxn, heat_uv;
    float psi; 
    int node;

     // set trapezoidal hyper parameter: 1-FEuler, 0-BEuler, 0.5-Trap
    if (method == 0){
        psi = 1.; 
    }else if (method == 1){
        psi = 0.;
    }else if (method == 2){
        psi = 0.5;
    }else{
        throw std::invalid_argument("ERROR: enter valid method 1-FEuler, 0-BEuler, 0.5-Trap  ---"); 
    }

    // declare implicit vectors
    std::vector<double> c_PI_0(_n_vol_nodes),
                        c_PI_1(_n_vol_nodes),
                        c_PIdot_0(_n_vol_nodes),
                        c_PIdot_1(_n_vol_nodes),
                        c_Mdot_0(_n_vol_nodes),
                        c_Mdot_1(_n_vol_nodes),
                        c_M_0(_n_vol_nodes),
                        c_M_1(_n_vol_nodes),
                        theta_0(_n_vol_nodes),
                        theta_1(_n_vol_nodes);

    // initialize implicit vectors
    for (int node = 0; node < _n_vol_nodes; node++){
        c_PI_0[node]    = _c_PI[node];
        c_PI_1[node]    = _c_PI[node];
        c_PIdot_0[node] = _c_PIdot[node];
        c_PIdot_1[node] = _c_PIdot[node];
        c_Mdot_0[node]  = _c_Mdot[node];
        c_Mdot_1[node]  = _c_Mdot[node];
        c_M_0[node]     = _c_M[node];
        c_M_1[node]     = _c_M[node];
        theta_0[node]   = _theta[node];
        theta_1[node]   = _theta[node];
    }

    int count    = 0;
    double error = 100;
    double depth = 0;
    double err_step;

    // fixed point iteration
    while (error > _tol){

        // cap iterations
        if (count > _thresh){
            throw std::invalid_argument("--- SolveSystem (trapezoidal) did not converge ---");
        }

        err_step = 0;

        for (int node = 0; node < _n_vol_nodes; node++){

            // map node to coordinates
            node2Coord(node, _current_coords); 
            int i = _current_coords[0]; 
            int j = _current_coords[1]; 
            int k = _current_coords[2]; 

            depth = _len_block-_current_coords[2]*_coord_map_const;

            // internal nodes
            if (   i != 0 && i != (_nodes-1)
                && j != 0 && j != (_nodes-1)
                && k != 0 && k != (_nodes-1)){

                // solve equation 1
                c_PI_1[node] =   _c_PI[node] + _dt * (  (1-psi) * iRate(c_PI_0, I0, depth, node)
                                                        +   psi   * iRate(_c_PI, I0, depth, node));

                err_step += squaredDiff(c_PI_0[node], c_PI_1[node]);


                // solve equation 2
                c_PIdot_1[node] = _c_PIdot[node] + _dt * (   (1-psi) * piDotRate(c_PIdot_0, c_PI_0, c_M_0, I0, depth, node)
                                                            +   psi   * piDotRate(_c_PIdot, _c_PI, _c_M, I0, depth, node));

                err_step += squaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                // solve equation 3
                c_Mdot_1[node] = _c_Mdot[node] + _dt * (   (1-psi) * mDotRate(c_Mdot_0, c_PIdot_0, c_M_0, node)
                                                        +   psi   * mDotRate(_c_Mdot, _c_PIdot, _c_M, node));

                err_step += squaredDiff(c_Mdot_0[node], c_Mdot_1[node]);

                // solve equation 4
                c_M_1[node] = _c_M[node] + _dt * (   (1-psi) * mRate(c_M_0, c_Mdot_0, c_PIdot_0, node)
                                                    +   psi   * mRate(_c_M, _c_Mdot, _c_PIdot, node));

                err_step += squaredDiff(c_M_0[node], c_M_1[node]);

                // solve equation 5
                theta_1[node] = _theta[node] + _dt* (   (1-psi) * tempRate(theta_0, c_M_0, c_Mdot_0, c_PI_0, _c_PIdot, I0, node)
                                                    +   psi   * tempRate(_theta, _c_M, _c_Mdot, _c_PI, _c_PIdot, I0, node));

                err_step += squaredDiff(theta_0[node], theta_1[node]);

            }

            // BOUNDARY NODES
            else if (      i == 0 || i == (_nodes-1)
                        || j == 0 || j == (_nodes-1)
                        || k == 0 || k == (_nodes-1)){

                // solve non-spatially dependent equations: equation 1
                c_PI_1[node] = _c_PI[node]
                                + _dt*(  (1-psi) * iRate(c_PI_0, I0, depth, node)
                                        +   psi   * iRate(_c_PI, I0, depth, node));
                err_step +=   squaredDiff(c_PI_0[node], c_PI_1[node]);
                
                // check material type is interfacial, skip spatial dependencies for reactions
                if (_material_type[node] == 2){
                    // solve equation 2
                    c_PIdot_1[node] = _c_PIdot[node] + _dt * (   (1-psi) * piDotRate(c_PIdot_0, c_PI_0, c_M_0, I0, depth, node)
                                                            +   psi   * piDotRate(_c_PIdot, _c_PI, _c_M, I0, depth, node));
                    err_step +=   squaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                    // solve equation 3
                    c_Mdot_1[node] = _c_Mdot[node] + _dt * (   (1-psi) * mDotRate(c_Mdot_0, c_PIdot_0, c_M_0, node)
                                                            +   psi   * mDotRate(_c_Mdot, _c_PIdot, _c_M, node));
                    err_step +=   squaredDiff(c_Mdot_0[node], c_Mdot_1[node]);

                    // solve equation 4
                    c_M_1[node] = _c_M[node] + _dt * (   (1-psi) * mRate(c_M_0, c_Mdot_0, c_PIdot_0, node)
                                                    +   psi   * mRate(_c_M, _c_Mdot, _c_PIdot, node));
                    err_step += squaredDiff(c_M_0[node], c_M_1[node]);

                }

                // ENFORCE NEUMANN BOUNDARY CONDITIONS
                // bottom face
                double pre_1 = 4 / 3, pre_2 = 1 / 3; 
                if (i == 0){
                    // if material is UGAP resin
                    if (_material_type[node] == 1){
                        // apply neumann bc using a second order approximation
                        c_PIdot_1[node] = pre_1 * c_PIdot_1[node+_n_plane_nodes] - pre_2 * c_PIdot_1[node+_n_plane_nodes+_n_plane_nodes]; 
                        err_step       += squaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                        c_Mdot_1[node] = pre_1 * c_Mdot_1[node+_n_plane_nodes] - pre_2 * c_Mdot_1[node+_n_plane_nodes+_n_plane_nodes]; // second order approximation
                        err_step      += squaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 
                        
                        c_M_1[node] = pre_1 * c_M_1[node+_n_plane_nodes] - pre_2 * c_M_1[node+_n_plane_nodes+_n_plane_nodes]; // second order approximation
                        err_step +=   squaredDiff(c_M_0[node], c_M_1[node]);
                    }
                }
                // top face
                else if (i == (_nodes-1)){
                    if (_material_type[node] == 1){
                        // apply neumann bc using a second order approximation
                        c_PIdot_1[node] = pre_1 * c_PIdot_1[node-_n_plane_nodes] - pre_2 * c_PIdot_1[node-_n_plane_nodes-_n_plane_nodes]; 
                        err_step +=   squaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                        c_Mdot_1[node] = pre_1 * c_Mdot_1[node-_n_plane_nodes] - pre_2 * c_Mdot_1[node-_n_plane_nodes-_n_plane_nodes]; 
                        err_step +=   squaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                        c_M_1[node] = pre_1 * c_M_1[node-_n_plane_nodes] - pre_2 * c_M_1[node-_n_plane_nodes-_n_plane_nodes]; 
                        err_step +=   squaredDiff(c_M_0[node], c_M_1[node]);
                    }
                }

                // wall 1: front wall
                else if (j == 0){
                    if (_material_type[node] == 1){
                        // apply neumann bc using a second order approximation
                        c_PIdot_1[node] = pre_1 * c_PIdot_1[node+_nodes] - pre_2 * c_PIdot_1[node+_nodes+_nodes]; 
                        err_step +=   squaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                        c_Mdot_1[node] = pre_1 * c_Mdot_1[node+_nodes] - pre_2 * c_Mdot_1[node+_nodes+_nodes]; 
                        err_step +=   squaredDiff(c_Mdot_0[node], c_Mdot_1[node]);

                        c_M_1[node] = pre_1 * c_M_1[node+_nodes] - pre_2 * c_M_1[node+_nodes+_nodes]; 
                        err_step +=   squaredDiff(c_M_0[node], c_M_1[node]);
                    }
                }

                // wall 3: back wall
                else if (j == (_nodes-1)){
                    if (_material_type[node] == 1){
                        // apply neumann bc using a second order approximation
                        c_PIdot_1[node] = pre_1 * c_PIdot_1[node-_nodes] - pre_2 * c_PIdot_1[node-_nodes-_nodes];
                        err_step +=   squaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                        c_Mdot_1[node] = pre_1 * c_Mdot_1[node-_nodes] - pre_2 * c_Mdot_1[node-_nodes-_nodes];
                        err_step +=   squaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                        c_M_1[node] = pre_1 * c_M_1[node-_nodes] - pre_2 * c_M_1[node-_nodes-_nodes]; 
                        err_step +=   squaredDiff(c_M_0[node], c_M_1[node]);
                    }
                }

                // wall 2: left wall
                else if (k == 0){
                    if (_material_type[node] == 1){
                        // apply neumann bc using a second order approximation
                        c_PIdot_1[node] = pre_1 * c_PIdot_1[node+1] - pre_2 * c_PIdot_1[node+1+1]; 
                        err_step +=   squaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                        c_Mdot_1[node] = pre_1 * c_Mdot_1[node+1] - pre_2 * c_Mdot_1[node+1+1];
                        err_step +=   squaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                        c_M_1[node] = pre_1 * c_M_1[node+1] - pre_2 * c_M_1[node+1+1];
                        err_step += squaredDiff(c_M_0[node], c_M_1[node]);
                    }
                }

                // wall 4: right wall
                else if (k == (_nodes-1)){
                    if (_material_type[node] == 1){
                        // apply neumann bc using a second order approximation
                        c_PIdot_1[node] = pre_1 * c_PIdot_1[node-1] - pre_2 * c_PIdot_1[node-1-1]; 
                        err_step +=   squaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                        c_Mdot_1[node] = pre_1 * c_Mdot_1[node-1] - pre_2 * c_Mdot_1[node-1-1]; 
                        err_step +=   squaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                        c_M_1[node] = pre_1 * c_M_1[node-1] - pre_2 * c_M_1[node-1-1];
                        err_step +=   squaredDiff(c_M_0[node], c_M_1[node]);
                    }
                }

                else{
                    throw std::invalid_argument("--- trapezoidal: boundary condition error ---");
                }
            }else {
                // something is wrong..
                throw std::invalid_argument("--- FIXED POINT BOUNDARY NODE ISSUE: trapezoidal ---");
            }
        }


        error = sqrt(err_step);
        count++;

        if (error > _tol){
            c_PI_0 = c_PI_1;
            c_PIdot_0 = c_PIdot_1;
            c_Mdot_0 = c_Mdot_1;
            c_M_0 = c_M_1;
            theta_0 = theta_1;
        }
        else{
            c_PI_next = c_PI_1;
            c_PIdot_next = c_PIdot_1;
            c_Mdot_next = c_Mdot_1;
            c_M_next = c_M_1;
            theta_next = theta_1;
        }
    }
}


void Voxel::config2File(double dt){
    
    // write to file
    _print_sim_config.open(_file_path + "sim_config/sim_config.txt");
    _print_sim_config << "Simulation Configuration\n" << std::endl; 

    _print_sim_config << "==================================" << std::endl;

    _print_sim_config << "Initial concentrations (mol/m^3): " << std::endl;
    _print_sim_config << "_c_M0: " << _c_M0 << std::endl;
    _print_sim_config << "_c_PI0: " << _c_PI0 << std::endl;
    _print_sim_config << "_theta0 " << _theta0 << std::endl;
    _print_sim_config << "==================================\n" << std::endl;

    _print_sim_config << "Numerical parameters" << std::endl;
    _print_sim_config << "h: " << _h << std::endl;
    _print_sim_config << "dt: " << dt << std::endl;
    _print_sim_config << "Diffusion CFL: " << _Dm0 * dt / _h / _h << std::endl;
    _print_sim_config << "Thermal CFL: " << dt / _h / _rho_UGAP / _Cp_nacl << std::endl;
    

    _print_sim_config << "\n==================================" << std::endl;
    _print_sim_config << "solids loading: " << (1.0 * _particles_ind.size() / _n_vol_nodes) << std::endl;
    _print_sim_config << "_particles_ind.size(): " << _particles_ind.size() << std::endl;
    _print_sim_config << "\n==================================" << std::endl;

    _print_sim_config.close(); 
}


void Voxel::density2File(){
    // write to file
    _print_density.open(_file_path + "_density/_density.vtk");

    _print_density << "# vtk DataFile Version 2.0" << std::endl;
    _print_density << "voxel _density ugap with particles" << std::endl;
    _print_density << "ASCII" << std::endl;
    _print_density << "DATASET RECTILINEAR_GRID" << std::endl;
    _print_density << "DIMENSIONS " << _nodes << " " <<  _nodes << " " << _nodes << std::endl;

    _print_density << "X_COORDINATES " << _nodes << " float" << std::endl;
    double dx = 0.;
    int counter = 1;
    for (int i = 0; i < _nodes; i++){
        _print_density << dx * 1000 << " ";
        if (counter % 6 == 0){
            _print_density << std::endl;
        }
        dx += _h;
        counter++;
    }
    _print_density << std::endl;

    _print_density << "Y_COORDINATES " << _nodes << " float" << std::endl;
    dx = 0.;
    counter = 1;
    for (int i = 0; i < _nodes; i++){
        _print_density << dx * 1000 << " ";
        if (counter % 6 == 0){
            _print_density << std::endl;
        }
        dx += _h;
        counter++;
    }
    _print_density << std::endl;

    _print_density << "Z_COORDINATES " << _nodes << " float" << std::endl;
    dx = 0.;
    counter = 1;
    for (int i = 0; i < _nodes; i++){

        _print_density << dx * 1000 << " ";
        if (counter % 6 == 0){
            _print_density << std::endl;
        }
        dx += _h;
        counter++;
    }
    _print_density << std::endl;

    _print_density << "POINT_DATA " << _n_vol_nodes << std::endl;
    _print_density << "SCALARS _density float" << std::endl;
    _print_density << "LOOKUP_TABLE default" << std::endl;
    counter = 1;
    for (int i = 0; i < _n_vol_nodes; i++){
        _print_density << _density[i] << " ";
        if (counter % 6 == 0){
            _print_density << std::endl;
        }
        counter++;
    }
    _print_density.close();

}

void Voxel::avgConcentrations2File(int counter, 
                                   std::vector<double> &c_PI_next,
                                   std::vector<double> &c_PIdot_next,
                                   std::vector<double> &c_Mdot_next,
                                   std::vector<double> &c_M_next,
                                   std::vector<double> &theta_next,
                                   double time){
    
    // compute average top concentration
    double avg_top_cPI = 0,        avg_tot_cPI = 0,    avg_bot_cPI = 0; 
    double avg_top_cPIdot = 0,     avg_tot_cPIdot = 0, avg_bot_cPIdot = 0;
    double avg_top_cMdot = 0,      avg_tot_cMdot = 0,  avg_bot_cMdot = 0;
    double avg_top_cM = 0,         avg_tot_cM = 0,     avg_bot_cM = 0;
    double avg_top_theta = 0,      avg_tot_theta = 0,  avg_bot_theta = 0;
    double avg_free_volume = 0,    avg_k_p = 0,        avg_k_t = 0;
    double avg_diff_pdot_top = 0,  avg_diff_pdot = 0,  avg_diff_pdot_bot = 0; 
    double avg_diff_mdot_top = 0,  avg_diff_mdot = 0,  avg_diff_mdot_bot = 0; 
    double avg_diff_m_top = 0,     avg_diff_m = 0,     avg_diff_m_bot = 0; 
    double avg_diff_theta_top = 0, avg_diff_theta = 0, avg_diff_theta_bot = 0; 
    
    // compute average top, total and bottom concentration, and then write to file
    int nodes_resin = 0; 
    int nodes_top_resin = 0, nodes_bot_resin = 0; 
    for (int node = 0; node < _n_vol_nodes; node++){

        // compute average temperature related values over all nodes
        avg_diff_theta  += _diff_theta[node];
        avg_tot_theta   += theta_next[node];

        // compute average diffusivity of temperature top and bottom nodes
        if (_n_plane_nodes < node && node < 2 * _n_plane_nodes){
            avg_diff_theta_bot += _diff_theta[node];
            
        }
        if (_n_vol_nodes - _n_plane_nodes > node && node > _n_vol_nodes - 2 * _n_plane_nodes){
            avg_diff_theta_top += _diff_theta[node];
            
        }

        // compute average reaction related values over resin nodes
        if (_material_type[node] == 1){
            // compute average total concentration
            avg_tot_cPI     += c_PI_next[node];
            avg_tot_cPIdot  += c_PIdot_next[node];
            avg_tot_cMdot   += c_Mdot_next[node];
            avg_tot_cM      += c_M_next[node];
            
            avg_free_volume += _f_free_volume[node];
            avg_k_p         += _k_p[node];
            avg_k_t         += _k_t[node];
            avg_diff_pdot   += _diff_pdot[node];
            avg_diff_mdot   += _diff_mdot[node];
            avg_diff_m      += _diff_m[node];
            


            // compute average bottom concentration
            if (node < _n_plane_nodes){
                avg_bot_cPI     += c_PI_next[node];
                avg_bot_cPIdot  += c_PIdot_next[node];
                avg_bot_cMdot   += c_Mdot_next[node];
                avg_bot_cM      += c_M_next[node];
                // avg_bot_theta += theta_next[node];
                
                nodes_bot_resin++; 
            }

            if (node < 2 * _n_plane_nodes && node > _n_plane_nodes){
                // diffusive terms
                avg_diff_pdot_bot  += _diff_pdot[node];
                avg_diff_mdot_bot  += _diff_mdot[node];
                avg_diff_m_bot     += _diff_m[node];
            }

            // compute average top concentration
            if (node > _n_vol_nodes - _n_plane_nodes){
                avg_top_cPI     += c_PI_next[node];
                avg_top_cPIdot  += c_PIdot_next[node];
                avg_top_cMdot   += c_Mdot_next[node];
                avg_top_cM      += c_M_next[node];
                // avg_top_theta += theta_next[node];
                nodes_top_resin++;
            }

            if (node < _n_vol_nodes - _n_plane_nodes && node > _n_vol_nodes - 2 * _n_plane_nodes){
                // diffusive terms
                avg_diff_pdot_top  += _diff_pdot[node];
                avg_diff_mdot_top  += _diff_mdot[node];
                avg_diff_m_top     += _diff_m[node];
            }
            
            nodes_resin++;
        }   
    }

    avg_tot_cPI     /= nodes_resin;
    avg_bot_cPI     /= nodes_bot_resin;
    avg_top_cPI     /= nodes_top_resin;

    avg_tot_cPIdot  /= nodes_resin;
    avg_bot_cPIdot  /= nodes_bot_resin;
    avg_top_cPIdot  /= nodes_top_resin;

    avg_tot_cMdot   /= nodes_resin;
    avg_bot_cMdot   /= nodes_bot_resin;
    avg_top_cMdot   /= nodes_top_resin;

    avg_tot_cM      /= nodes_resin;
    avg_bot_cM      /= nodes_bot_resin;
    avg_top_cM      /= nodes_top_resin;

    avg_tot_theta   /= _n_vol_nodes;
    avg_free_volume /= nodes_resin;
    avg_k_p         /= nodes_resin;
    avg_k_t         /= nodes_resin;

    avg_diff_pdot      /= nodes_resin;
    avg_diff_pdot_top  /= nodes_top_resin;
    avg_diff_pdot_bot  /= nodes_bot_resin;
    
    avg_diff_mdot      /= nodes_resin;
    avg_diff_mdot_top  /= nodes_top_resin;
    avg_diff_mdot_bot  /= nodes_bot_resin;    

    avg_diff_m         /= nodes_resin;
    avg_diff_m_top     /= nodes_top_resin;
    avg_diff_m_bot     /= nodes_bot_resin;

    avg_diff_theta     /= _n_vol_nodes;
    avg_diff_theta_top /= _n_vol_nodes;
    avg_diff_theta_bot /= _n_vol_nodes;


    // open file
    if (counter == 0){std::string file_avg_concentrations = _file_path + "python_plotting/avg_concentration_simID_" + std::to_string(_sim_id) + ".csv";
        _print_avg_concentrations.open(file_avg_concentrations);
        _print_avg_concentrations <<       "time, avg_top_cPI, avg_tot_cPI, avg_bot_cPI, ";
        _print_avg_concentrations <<       "avg_top_cPIdot, avg_tot_cPIdot, avg_bot_cPIdot, "; 
        _print_avg_concentrations <<       "avg_top_cMdot, avg_tot_cMdot, avg_bot_cMdot, ";
        _print_avg_concentrations <<       "avg_top_cM, avg_tot_cM, avg_bot_cM, "; 
        _print_avg_concentrations <<       "avg_tot_theta, "; 
        _print_avg_concentrations <<       "avg_free_volume, avg_k_p, avg_k_t, ";
        _print_avg_concentrations <<       "avg_diff_pdot_top, avg_diff_pdot, avg_diff_pdot_bot, ";
        _print_avg_concentrations <<       "avg_diff_mdot_top, avg_diff_mdot, avg_diff_mdot_bot, ";
        _print_avg_concentrations <<       "avg_diff_m_top, avg_diff_m, avg_diff_m_bot, ";
        _print_avg_concentrations <<       "avg_diff_theta_top, avg_diff_theta, avg_diff_theta_bot" << std::endl;
        // _print_avg_concentrations <<       "avg_diff_pdot, avg_diff_mdot, avg_diff_m, avg_diff_theta" << std::endl;

    }

    _print_avg_concentrations << time << ", ";
    _print_avg_concentrations << avg_top_cPI        << ", " << avg_tot_cPI     << ", " << avg_bot_cPI        << ", ";
    _print_avg_concentrations << avg_top_cPIdot     << ", " << avg_tot_cPIdot  << ", " << avg_bot_cPIdot     << ", ";
    _print_avg_concentrations << avg_top_cMdot      << ", " << avg_tot_cMdot   << ", " << avg_bot_cMdot      << ", ";
    _print_avg_concentrations << avg_top_cM         << ", " << avg_tot_cM      << ", " << avg_bot_cM         << ", ";
    _print_avg_concentrations << avg_tot_theta      << ", " << avg_free_volume << ", " << avg_k_t            << ", "; 
    _print_avg_concentrations << avg_k_p                                                                     << ", "; 
    _print_avg_concentrations << avg_diff_pdot_top  << ", " << avg_diff_pdot   << ", " << avg_diff_pdot_bot  << ", ";
    _print_avg_concentrations << avg_diff_mdot_top  << ", " << avg_diff_mdot   << ", " << avg_diff_mdot_bot  << ", ";
    _print_avg_concentrations << avg_diff_m_top     << ", " << avg_diff_m      << ", " << avg_diff_m_bot     << ", ";
    _print_avg_concentrations << avg_diff_theta_top << ", " << avg_diff_theta  << ", " << avg_diff_theta_bot << std::endl;

    // track average quantities for objective function
    _c_PI_avg.push_back(avg_tot_cPI);
    _c_PIdot_avg.push_back(avg_tot_cPIdot);
    _c_Mdot_avg.push_back(avg_tot_cMdot);
    _c_M_avg.push_back(avg_tot_cM);

    // _print_avg_concentrations << avg_diff_pdot << ", " << avg_diff_mdot << ", " << avg_diff_m << ", " << avg_diff_theta << std::endl;
    
    if (time == 30.0){
        std::cout << "--- COMPLETE ---" << std::endl;
        _print_avg_concentrations.close();
    }

}

void Voxel::concentrations2File(int counter,
                                std::vector<double> &c_PI_next,
                                std::vector<double> &c_PIdot_next,
                                std::vector<double> &c_Mdot_next,
                                std::vector<double> &c_M_next,
                                std::vector<double> &theta_next,
                                double time){


    // write to file
    std::string file_name = _file_path + "concentrations_t" + std::to_string(counter) + ".vtk";
    _print_concentrations.open(file_name);
    
    _print_concentrations << "# vtk DataFile Version 2.0"              << std::endl;
    _print_concentrations << "voxel concentration ugap with particles" << std::endl;
    _print_concentrations << "ASCII"                                   << std::endl;
    _print_concentrations << "DATASET RECTILINEAR_GRID"                << std::endl;
    _print_concentrations << "DIMENSIONS " << _nodes << " " <<  _nodes << " " << _nodes << std::endl;

    _print_concentrations << "X_COORDINATES " << _nodes << " float" << std::endl;
    double dx = 0.;
    int cnt = 1;
    for (int i = 0; i < _nodes; i++){
        _print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        dx += _h;
        cnt++;
    }
    _print_concentrations << std::endl;

    _print_concentrations << "Y_COORDINATES " << _nodes << " float" << std::endl;
    dx = 0.;
    cnt = 1;
    for (int i = 0; i < _nodes; i++){
        _print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        dx += _h;
        cnt++;
    }
    _print_concentrations << std::endl;

    _print_concentrations << "Z_COORDINATES " << _nodes << " float" << std::endl;
    dx = 0.;
    cnt = 1;
    for (int i = 0; i < _nodes; i++){

        _print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        dx += _h;
        cnt++;
    }
    _print_concentrations << std::endl;

    _print_concentrations << "POINT_DATA " << _n_vol_nodes << std::endl;
    _print_concentrations << "SCALARS _c_PI float" << std::endl;
    _print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int i = 0; i < _n_vol_nodes; i++){
        _print_concentrations << c_PI_next[i] << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        cnt++;
    }
    _print_concentrations << std::endl;

    _print_concentrations << "SCALARS _c_PIdot float" << std::endl;
    _print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int i = 0; i < _n_vol_nodes; i++){
        _print_concentrations << c_PIdot_next[i] << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        cnt++;
    }
    _print_concentrations << std::endl;

    _print_concentrations << "SCALARS _c_Mdot float" << std::endl;
    _print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int i = 0; i < _n_vol_nodes; i++){
        _print_concentrations << c_Mdot_next[i] << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        cnt++;
    }
    _print_concentrations << std::endl;

    _print_concentrations << "SCALARS _c_M float" << std::endl;
    _print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int i = 0; i < _n_vol_nodes; i++){
        _print_concentrations << c_M_next[i] << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        cnt++;
    }
    _print_concentrations << std::endl;

    _print_concentrations << "SCALARS _theta float" << std::endl;
    _print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int i = 0; i < _n_vol_nodes; i++){
        _print_concentrations << _theta[i] << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        cnt++;
    }
    _print_concentrations << std::endl;

    _print_concentrations.close();
}


void Voxel::nonBoundaries2File( int counter, 
                                std::vector<double> &c_PI_next,
                                std::vector<double> &c_PIdot_next,
                                std::vector<double> &c_Mdot_next,
                                std::vector<double> &c_M_next,
                                std::vector<double> &theta_next,
                                double time, 
                                int (&coords)[3]){

    // writes only non-boundary nodes to file

    // WRITE TO FILE
    std::string file_name = _file_path + "concentrations_t" + std::to_string(counter) + ".vtk";
    _print_concentrations.open(file_name);
    _print_concentrations << "# vtk DataFile Version 2.0" << std::endl;
    _print_concentrations << "voxel concentration ugap with particles" << std::endl;
    _print_concentrations << "ASCII" << std::endl;
    _print_concentrations << "DATASET RECTILINEAR_GRID" << std::endl;
    _print_concentrations << "DIMENSIONS " << (_nodes-2) << " " <<  (_nodes-2) << " " << (_nodes-2) << std::endl;

    // OUTPUT COORDS
    _print_concentrations << "X_COORDINATES " << (_nodes-2) << " float" << std::endl;
    double dx = _h;
    int cnt = 1;
    for (int i = 1; i < _nodes-1; i++){
        _print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        dx += _h;
        cnt++;
    }
    _print_concentrations << std::endl;

    _print_concentrations << "Y_COORDINATES " << (_nodes-2) << " float" << std::endl;
    dx = _h;
    cnt = 1;
    for (int i = 1; i < _nodes-1; i++){
        _print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        dx += _h;
        cnt++;
    }
    _print_concentrations << std::endl;

    _print_concentrations << "Z_COORDINATES " << (_nodes-2) << " float" << std::endl;
    dx = _h;
    cnt = 1;
    for (int i = 1; i < _nodes-1; i++){

        _print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            _print_concentrations << std::endl;
        }
        dx += _h;
        cnt++;
    }
    _print_concentrations << std::endl;

    // OUTPUT C_PI
    _print_concentrations << "POINT_DATA " << (_nodes-2) * (_nodes-2) * (_nodes-2) << std::endl;
    _print_concentrations << "SCALARS _c_PI float" << std::endl;
    _print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;

    // loop through all nodes
    for (int node = 0; node < _n_vol_nodes; node++){
        
        // only write internal nodes
        node2Coord(node, _current_coords); 
        if (   _current_coords[0] != 0 && _current_coords[0] != (_nodes-1) 
            && _current_coords[1] != 0 && _current_coords[1] != (_nodes-1) 
            && _current_coords[2] != 0 && _current_coords[2] != (_nodes-1)){
            _print_concentrations << c_PI_next[node] << " ";
            if (cnt % 6 == 0){
                _print_concentrations << std::endl;
            }
            cnt++;
        }
    }
    
    _print_concentrations << std::endl;

    // OUTPUT C_PIDOT
    _print_concentrations << "SCALARS _c_PIdot float" << std::endl;
    _print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;

    // loop throug all nodes
    for (int node = 0; node < _n_vol_nodes; node++){

        // only write internal nodes
        node2Coord(node, _current_coords);
        if (   _current_coords[0] != 0 && _current_coords[0] != (_nodes-1)
            && _current_coords[1] != 0 && _current_coords[1] != (_nodes-1)
            && _current_coords[2] != 0 && _current_coords[2] != (_nodes-1)){
            _print_concentrations << c_PIdot_next[node] << " ";
            if (cnt % 6 == 0){
                _print_concentrations << std::endl;
            }
            cnt++;
        }
    }
    _print_concentrations << std::endl;

    // OUTPUT C_MDOT
    _print_concentrations << "SCALARS _c_Mdot float" << std::endl;
    _print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;

    // loop through all nodes
    for (int node = 0; node < _n_vol_nodes; node++){
        
        // only write internal nodes
        node2Coord(node, _current_coords);
        if (   _current_coords[0] != 0 && _current_coords[0] != (_nodes-1)
            && _current_coords[1] != 0 && _current_coords[1] != (_nodes-1)
            && _current_coords[2] != 0 && _current_coords[2] != (_nodes-1)){
            
            _print_concentrations << c_Mdot_next[node] << " ";
            if (cnt % 6 == 0){
                _print_concentrations << std::endl;
            }
        }
    }
    _print_concentrations << std::endl;

    // OUTPUT C_M
    _print_concentrations << "SCALARS _c_M float" << std::endl;
    _print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int node = 0; node < _n_vol_nodes; node++){
        
        // only write internal nodes
        node2Coord(node, _current_coords);
        if (   _current_coords[0] != 0 && _current_coords[0] != (_nodes-1)
            && _current_coords[1] != 0 && _current_coords[1] != (_nodes-1)
            && _current_coords[2] != 0 && _current_coords[2] != (_nodes-1)){
            
            _print_concentrations << c_M_next[node] << " ";
            if (cnt % 6 == 0){
                _print_concentrations << std::endl;
            }
        }
    }
    _print_concentrations << std::endl;

    // OUTPUT THETA
    _print_concentrations << "SCALARS _theta float" << std::endl;
    _print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int node = 0; node < _n_vol_nodes; node++){

        // only write internal nodes
        node2Coord(node, _current_coords);
        if (   _current_coords[0] != 0 && _current_coords[0] != (_nodes-1)
            && _current_coords[1] != 0 && _current_coords[1] != (_nodes-1)
            && _current_coords[2] != 0 && _current_coords[2] != (_nodes-1)){
            
            _print_concentrations << _theta[node] << " ";
            if (cnt % 6 == 0){
                _print_concentrations << std::endl;
            }
        }
    }
    _print_concentrations << std::endl;

    _print_concentrations.close();
}


void Voxel::simulate(int method, int save_voxel, int obj_fn, double w[4]){

    // time discretization -> [0., dt, 2*dt, ..., T]
    int N_TIME_STEPS = _t_final / _dt;
    int print_iter   = N_TIME_STEPS / 600; 
    std::vector<double> total_time(N_TIME_STEPS, 0);

    if (!_multi_thread){
        std::cout << "=================================="                                     << std::endl;
        std::cout << "Simulation parameters"                                                  << std::endl;
        std::cout << "_sim_id: "               << _sim_id                                       << std::endl;
        std::cout << "Total time: "           << _t_final                                      << std::endl;
        std::cout << "Number of time steps: " << N_TIME_STEPS                                 << std::endl;
        std::cout << "Print iteration: "      << print_iter                                   << std::endl;
        std::cout << "=================================="                                     << std::endl;
        std::cout << "Numerical parameters"                                                   << std::endl;
        std::cout << "h: "   << _h                                                             << std::endl;
        std::cout << "_dt: "  << _dt                                                            << std::endl;
        std::cout << "Diffusion CFL: "   << _Dm0 * _dt / _h / _h                                  << std::endl;
        std::cout << "Thermal CFL: "     << _dt * _K_thermal_shanna/ _h / _h / _rho_UGAP / _Cp_nacl << std::endl;
        std::cout << "\n=================================="                                   << std::endl;
    }

    config2File(_dt); 


    // initialize next time step values
    std::vector<double> c_PI_next(_n_vol_nodes,    _c_PI0);
    std::vector<double> c_PIdot_next(_n_vol_nodes, 0.);
    std::vector<double> c_Mdot_next(_n_vol_nodes,  0.);
    std::vector<double> c_M_next(_n_vol_nodes,     _c_M0);
    std::vector<double> theta_next(_n_vol_nodes,   _theta0);

    // compute initial reaction rate constants
    computeRxnRateConstants();

    // write initial values to files
    if (save_voxel == 1){
        concentrations2File(0,
                            _c_PI,
                            _c_PIdot,
                            _c_Mdot,
                            _c_M,
                            _theta,
                            0);
    }
    
    avgConcentrations2File(0,
                           _c_PI,
                           _c_PIdot,
                           _c_Mdot,
                           _c_M,
                           _theta,
                           0);

    // begin time stepping
    _timer = 0.;
    int file_counter = 1;
    double uv_light = I0;
    for (int t = 0; t < N_TIME_STEPS; t++) {

        total_time[t] = _timer;
        if (_timer <= _uvt){
            uv_light = I0; 
        }else{
            uv_light = 0.;
        }

        // compute energy intensity and reaction rate constants
        computeRxnRateConstants();

        // solve system of equations
        solveSystem(c_PI_next, c_PIdot_next, c_Mdot_next, c_M_next, theta_next, uv_light, _dt, method);

        _c_PI    = c_PI_next;
        _c_PIdot = c_PIdot_next;
        _c_Mdot  = c_Mdot_next;
        _c_M     = c_M_next;
        _theta   = theta_next;

        // display time
        _timer += _dt;
        if ((t + 1) % 100 == 0 && !_multi_thread){
            std::cout << "time: "      << _timer << " / " << _t_final                       << std::endl;
            std::cout << "iteration: " << t + 1 << " / " << N_TIME_STEPS + 1 << std::endl << std::endl;
        }

        // store solution results (every 100 steps) including last time step

        if (std::abs(std::floor(_timer * 2) / 2 - _timer) < _dt || t == N_TIME_STEPS - 1){    
            if (save_voxel == 1){
                concentrations2File(file_counter,
                                    _c_PI,
                                    _c_PIdot,
                                    _c_Mdot,
                                    _c_M,
                                    _theta,
                                    _timer);
            }

            avgConcentrations2File(file_counter,
                                   c_PI_next,
                                   c_PIdot_next,
                                   c_Mdot_next,
                                   c_M_next,
                                   theta_next,
                                   _timer);

            file_counter++;
        }
    }

    double max_PI    = *std::max_element(_c_PI_avg.begin(), _c_PI_avg.end());
    // double min_PI    = std::max(*std::min_element(_c_PI_avg.begin(), _c_PI_avg.end()), 0.0);
    double max_PIdot = *std::max_element(_c_PIdot_avg.begin(), _c_PIdot_avg.end());
    // double min_PIdot = std::max(*std::min_element(_c_PIdot_avg.begin(), _c_PIdot_avg.end()), 0.0);
    double max_Mdot  = *std::max_element(_c_Mdot_avg.begin(), _c_Mdot_avg.end());
    // double min_Mdot  = std::max(*std::min_element(_c_Mdot_avg.begin(), _c_Mdot_avg.end()), 0.0);
    double max_M     = *std::max_element(_c_M_avg.begin(), _c_M_avg.end());
    // double min_M     = std::max(*std::min_element(_c_M_avg.begin(), _c_M_avg.end()), 0.0);

    double average_PI_final    = _c_PI_avg[_c_PI_avg.size()-1];
    double average_PIdot_final = _c_PIdot_avg[_c_PIdot_avg.size()-1]; 
    double average_Mdot_final  = _c_Mdot_avg[_c_Mdot_avg.size()-1];  
    double average_M_final     = _c_M_avg[_c_M_avg.size()-1];

    // compute weighted multi objective function
    _obj_PI      = std::max(average_PI_final    / max_PI   , 0.0);
    _obj_PIdot   = std::max(average_PIdot_final / max_PIdot, 0.0);
    _obj_Mdot    = std::max(average_Mdot_final  / max_Mdot , 0.0);
    _obj_M       = std::max(average_M_final     / max_M    , 0.0);
    _obj_default = w[0]*_obj_PI + w[1]*_obj_PIdot + w[2]*_obj_Mdot + w[3]*_obj_M;
    
    // switch _obj to match input obj_fn
    switch (obj_fn){
        case 1:
            _obj = _obj_PI;
            break;
        case 2:
            _obj = _obj_PIdot;
            break;
        case 3:
            _obj = _obj_Mdot;
            break;
        case 4:
            _obj = _obj_M;
            break;
        default:
            _obj = _obj_default;
            break;
    }

    if (!_multi_thread){
        std::cout << "==================================" << std::endl;
        std::cout << "Simulation complete"                << std::endl;
        std::cout << "==================================" << std::endl;
        std::cout << "obj = " << _obj                     << std::endl;
        std::cout << "temp: " << _theta0                  << std::endl; 
        std::cout << "uvt:  " << _uvt                     << std::endl; 
        std::cout << "I0:   " << I0                       << std::endl;
        std::cout << "_rp:  " << _rp                      << std::endl;
        std::cout << "_vp:  " << _vp                      << std::endl;
        std::cout << "==================================" << std::endl;
    }
}

double Voxel::getObjPI() {
    return _obj_PI;
}

double Voxel::getObjPIDot() {
    return _obj_PIdot;
}

double Voxel::getObjMDot() {
    return _obj_Mdot;
}

double Voxel::getObjM() {
    return _obj_M;
}

double Voxel::getObjective() {
    return _obj;
}