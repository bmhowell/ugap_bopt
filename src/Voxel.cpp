#include "Voxel.h"

// tfinal, dt, node, idsim, temp, rp, vp, uvi, uvt

Voxel::Voxel(float TF, double DT, int N, int IDSIM, double temp, float UVI, float UVT){
    std::cout << "Initializing parameters: " << std::endl;

    // MEMBER VARIABLES
    // representative volume element RVE simulation parameters
    t_final = TF;                                               // |    s    |  final time
    dt      = DT;                                               // |    s    |  time step
    nodes   = N;                                                // | unitless|  total number of nodes
    sim_id  = IDSIM;                                            // |   ---   |  simulation id
    theta0  = temp;                                             // |    K    | initial and ambient temperature
    I0      = UVI;                                              // |  W/m^2  |  UV intensity
    uvt     = uvt;                                              // |    s    | uv exposure time

    // set file path
    file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/materials_opt/output";   // MACBOOK PRO
    // file_path = "/home/brian/Documents/berkeley/materials_opt/output/";         // LINUX CENTRAL COMPUTING

    interfacial_nodes = 1;                                      // |   ---   |  interfacial thickness parameter
    len_block         = 0.00084;                                // |    m    |  sample length
    h                 = (double) len_block / (nodes - 1);       // |    m    |  spatial discretization
    N_VOL_NODES       = nodes * nodes * nodes;                  // | unitless|  total number of nodes in RVE
    N_PLANE_NODES     = nodes * nodes;                          // | unitless|  total number of nodes in plane of RVE
    coord_map_const   = len_block / nodes; 

    // formulation - wt. percent
    percent_PI    = 0.0333;                                     // |   wt.%  | weight percent of photo initiator
    percent_PEA   = 0.15;                                       // |   wt.%  | weight percent of PEA
    percent_HDDA  = 0.0168;                                     // |   wt.%  | weight percent of HDDA
    percent_8025D = 0.4084;                                     // |   wt.%  | weight percent of 8025D
    percent_8025E = 0.0408;                                     // |   wt.%  | weight percent of 8025E
    percent_E4396 = 0.3507;                                     // |   wt.%  | weight percent of HDDA
    percent_M = percent_PEA + percent_HDDA;                     // |   wt.%  | weight percent of monomer

    // physical properties
    // densities and molecular weights
    rho_PEA = 1020;                                             // | kg/m^3  | density of PEA (estimated)
    rho_HDDA = 1020;                                            // | kg/m^3  | density of HDDA (estimated)
    rho_E4396 = 1100;                                           // | kg/m^3  | density of EBECRYL 4396
    rho_M = 0.899 * rho_PEA + 0.101 * rho_HDDA;                 // | kg/m^3  | weighted average density of monomer
    rho_P = 0.03 * rho_HDDA + 0.29 * rho_PEA + 0.68 * rho_E4396;// | kg/m^3  | weighted average density of polymer
    rho_UGAP = 1840;                                            // | kg/m^3  | estimated density of UGAP
    rho_nacl = 2170;                                            // | kg/m^3  | estimated density of NaCl

    mw_PEA = 0.19221;                                           // |  kg/mol | molecular weight of PEA
    mw_HDDA = 0.226;                                            // |  kg/mol | molecular weight of HDDA
    mw_M = 0.899 * mw_PEA + 0.101 * mw_HDDA;                    // |  kg/mol | weighted average molecular weight of monomer
    mw_PI = 0.4185;                                             // |  kg/mol | molecular weight of photo initiator

    basis_wt = 0.5;                                             // |   kg    | arbitrary starting ink weight
    basis_vol = basis_wt / rho_UGAP;                            // |   m^3   | arbitrary starting ink volume
    mol_PI = basis_wt * percent_PI / mw_PI;                     // |   mol   | required PI for basis weight
    mol_M = basis_wt * percent_M / mw_M;                        // |   mol   | required monomer for basis weight
    c_M0 = mol_M / basis_vol;                                   // | mol/m^3 | initial concentration of monomer
    c_PI0 = mol_PI / basis_vol;                                 // | mol/m^3 | initial concentration of photoinitiator
    c_NaCl = 37241.4;                                           // | mol/m^3 | concentration of NaCl

    // diffusion properties
    Dm0 = 2.36e-6;                                              // |  m^2/s  | diffusion c pre-exponential, monomer (shanna lit.)
    Am = 0.66;                                                  // | unitless| diffusion constant parameter, monomer (bowman lit.)

    // bowman reaction parameters
    Rg       = 8.3145;                                          // | J/mol K | universal gas constant
    alpha_P  = 0.000075;                                        // |   1/K   | coefficent of thermal expansion, polymerization (taki + bowman lit.)
    alpha_M  = 0.0005;                                          // |   1/K   | coefficent of thermal expansion, monomer (taki + bowman lit.)
    theta_gP = 236.75;                                          // |    K    | glass transition temperature, polymer UGAP (DMA MEASURED)
    theta_gM = 213.0;                                           // |    K    | glass transition temperature, monomer (bowman)

    k_P0 = 1.6e3;                                               // |m^3/mol s| true kinetic constant, polymerization (bowman lit) / 
    E_P  = 18.23e3;                                             // |  J/mol  | activation energy, polymerization (bowman lit. 1) /
    A_Dp = 0.66;                                                // | unitless| diffusion parameter, polymerization (bowman lit.)
    f_cp = 4.2e-2;                                              // | unitless| critical free volume, polymerization (SHANNA lit.)
    
    k_T0 = 3.6e3;                                               // |m^3/mol s| true kinetic constant, termination (shanna lit.)/ 
    E_T  = 2.94e3;                                              // |  J/mol  | activation energy, termination (bowman lit.)
    A_Dt = 1.2;                                                 // | unitless| activation energy, termination (bowman lit. MODIFIED?)
    f_ct = 6.0e-2;                                              // | unitless| critical free volume, termination (bowman lit.) / 
    R_rd = 11.;                                                 // |m^3/mol  | reaction diffusion parameter (taki lit.)

    k_I0 = 5.1e-4;                                              // |m^3/mol s| primary radical rate constant (TRIAL/ERROR METHOD) / 
    E_I  = 18.23e3;                                             // |  J/mol  | activation energy, initiation (bowman lit.)
    A_I  = 0.66;                                                // | unitless| diffusion parameter, initiation (bowman lit.)
    f_ci = 0.042;                                               // | unitless| critical free volume, initiation (bowman lit.)

    // thermal properties
    dHp            = 5.48e4;                                    // |  J/mol  | heat of polymerization of acrylate monomers (bowman lit.)
    Cp_nacl        = 880;                                       // | J/kg/K  | heat capacity of NaCl
    Cp_pea         = 180.3;                                     // | J/mol/K | heat capacity of PEA @ 298K - https://polymerdatabase.com/polymer%20physics/Cp%20Table.html
    Cp_pea        /= mw_PEA;                                    // | J/kg/K  | convert units
    Cp_hdda        = 202.9;                                     // | J/mol/K | solid heat capacity of HDDA - https://webbook.nist.gov/cgi/cbook.cgi?ID=C629118&Units=SI&Mask=1F
    Cp_hdda       /= mw_HDDA;                                   // | J/kg/K  | convert units
    K_thermal_nacl = 0.069;                                     // | W/m/K   | thermal conductivity

    Cp_shanna        = 1700;                                    // | J/kg/K  | shanna's heat capacity
    K_thermal_shanna = 0.2;                                     // | W/m/K   | shanna's thermal conductivity

    // photo initiator properties
    eps      = 9.66e-1;                                         // |m^3/mol m| initiator absorbtivity
    eps_nacl = 7e-4;                                            // |   1/m   | NaCl absorbtivity
    phi      = 0.6;                                             // | unitless| quantum yield inititation

    // numerical method parameters: backward euler
    tol = 5e-2;
    thresh = 50;

    // spatial discretization -> [0., h, 2*h, ..., L]
    double z_increment = 0.0;
    for (int i=0; i<nodes; i++){
        z_space.push_back(z_increment);
        z_increment += h;
    }



    /* initialize voxel values */

    // UV gradient
    std::fill_n(std::back_inserter(uv_values),     N_VOL_NODES, 0.);

    // material properties
    std::fill_n(std::back_inserter(density),       N_VOL_NODES, rho_UGAP);
    std::fill_n(std::back_inserter(heat_capacity), N_VOL_NODES, Cp_shanna);
    std::fill_n(std::back_inserter(therm_cond),    N_VOL_NODES, K_thermal_shanna);
    std::fill_n(std::back_inserter(material_type), N_VOL_NODES, 1);                // UGAP=1, particle=0
    std::fill_n(std::back_inserter(f_free_volume), N_VOL_NODES, 0.);

    // concentrations
    std::fill_n(std::back_inserter(c_PI),          N_VOL_NODES, c_PI0);
    std::fill_n(std::back_inserter(c_M),           N_VOL_NODES, c_M0);
    std::fill_n(std::back_inserter(c_PIdot),       N_VOL_NODES, 0.);
    std::fill_n(std::back_inserter(c_Mdot),        N_VOL_NODES, 0.);
    
    // diffusion values
    std::fill_n(std::back_inserter(diff_pdot),     N_VOL_NODES, 0.);
    std::fill_n(std::back_inserter(diff_mdot),     N_VOL_NODES, 0.);
    std::fill_n(std::back_inserter(diff_m)   ,     N_VOL_NODES, 0.);
    std::fill_n(std::back_inserter(diff_theta),    N_VOL_NODES, 0.);

    // temperature
    std::fill_n(std::back_inserter(theta),         N_VOL_NODES, theta0);

    // rate constants
    std::fill_n(std::back_inserter(k_t),           N_VOL_NODES, k_T0);
    std::fill_n(std::back_inserter(k_p),           N_VOL_NODES, k_P0);
    std::fill_n(std::back_inserter(k_i),           N_VOL_NODES, k_I0);

    std::cout << "==================================" << std::endl;

    std::cout << "Initial concentrations (mol/m^3): " << std::endl;
    std::cout << "c_M0: "  << c_M0   << std::endl;
    std::cout << "c_PI0: " << c_PI0  << std::endl;
    std::cout << "theta0 " << theta0 << std::endl;
    std::cout << "==================================\n" << std::endl;

    // initialize system
    current_coords[0] = 0;
    current_coords[1] = 0;
    current_coords[2] = 0;

}

// Destructor
Voxel::~Voxel() {
    std::cout << "Voxel destructor called" << std::endl;
}

// helper functions
double Voxel::SquaredDiff(double val_1, double val_2){
    return (val_1 - val_2) * (val_1 - val_2);
}


void Voxel::UniqueVec(std::vector<int>& vec) {
    std::sort(vec.begin(), vec.end());
    auto last = std::unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());
}


void Voxel::Node2Coord(int node, int (&coord)[3]){
    /*
        Given a 3D cube of size m by n by p, and integer coordinates (i, j, k)
        where 0 <= i < m, 0 <= j < n, and 0 <= k < p, the grid number can be 
        calculated as follows:
    */
    coord[2] = node % nodes;    // compute k
    int temp = node / nodes; 
    coord[1] = temp % nodes;    // compute j
    coord[0] = temp / nodes;    // compute i
    
}


int Voxel::Coord2Node(int (&coord)[3]){
    /*
        Given a 3D cube of size m by n by p, and integer coordinates (i, j, k)
        where 0 <= i < m, 0 <= j < n, and 0 <= k < p, the grid number can be 
        calculated as follows:
    */
    return coord[0] * nodes * nodes + coord[1] * nodes + coord[2];
}


void Voxel::ComputeParticles(double radius_1, double solids_loading) {
    
    // total number of host nodes for particles
    double n_particle_nodes = std::round(N_VOL_NODES * solids_loading);
    double particle_distance, node_particle, rand_loc;
    int node;

    std::cout << "\n--------- ------- ---------" << std::endl;
    std::cout << "--- GENERATING PARTICLES ---" << std::endl;
    std::cout << "--------- ------- ---------" << std::endl;
    int counter1 = 0;
    
    int particle_coords[3] = {0, 0, 0}; 
    int tot_part_nodes = 0; 
    while ((tot_part_nodes < n_particle_nodes) and (counter1 < 10000000)){
        // step 1: choose random node as particle
        // https://stackoverflow.com/questions/19665818/generate-random-numbers-using-c11-random-library
        std::random_device rd{};
        std::mt19937  gen{rd()};


        std::uniform_real_distribution<> dist{0, 1};
        rand_loc = dist(gen);

        // step 2: generate random seed location for particle
        node_particle = ceil(N_VOL_NODES * rand_loc) - 1;
        Node2Coord(node_particle, particle_coords);
        
        // step 3: find nodes that are within the distance of the seed location
        for (int node = 0; node < N_VOL_NODES; node++){
            Node2Coord(node, current_coords);
            particle_distance = sqrt(   SquaredDiff(current_coords[0] * coord_map_const, particle_coords[0] * coord_map_const)
                                      + SquaredDiff(current_coords[1] * coord_map_const, particle_coords[1] * coord_map_const)
                                      + SquaredDiff(current_coords[2] * coord_map_const, particle_coords[2] * coord_map_const) );

            // check if node is the particle radius range
            if (particle_distance <= radius_1){
                particles_ind.push_back(node);
            }

            // check if node is in interfacial region (between resin and particle)
            else if (interfacial_thick != 0 and particle_distance <= radius_1 + interfacial_thick * h){
                particle_interfacial_nodes.push_back(node);
            }

            else{
                continue;
            }

        }

        // ensure vectors contain no duplicated nodes
        UniqueVec(particles_ind);
        UniqueVec(particle_interfacial_nodes);

        // assign interfacial material properties
        for (int i = 0; i < particle_interfacial_nodes.size(); i++){
            material_type[particle_interfacial_nodes[i]]    = 2;

            // thermal properties
            density[particle_interfacial_nodes[i]]          = (rho_nacl + rho_UGAP) / 2;
            heat_capacity[particle_interfacial_nodes[i]]    = (Cp_nacl + Cp_shanna) / 2;
            therm_cond[particle_interfacial_nodes[i]]       = (K_thermal_nacl + K_thermal_shanna) / 2;

            // reaction properties
            k_t[particle_interfacial_nodes[i]]              = k_T0;
            k_p[particle_interfacial_nodes[i]]              = k_P0;
            c_PI[particle_interfacial_nodes[i]]             = c_PI0 / 2;
            c_PIdot[particle_interfacial_nodes[i]]          = 0.;
            c_Mdot[particle_interfacial_nodes[i]]           = 0.;
            c_M[particle_interfacial_nodes[i]]              = c_M0 / 2;
        }

        // assign particle material properties
        for (int i = 0; i < particles_ind.size(); i++){

            // assign type particle
            material_type[particles_ind[i]]                 = 0;

            // thermal properties
            density[particles_ind[i]]                       = rho_nacl;
            heat_capacity[particles_ind[i]]                 = Cp_nacl;
            therm_cond[particles_ind[i]]                    = K_thermal_nacl;

            // reaction properties
            k_t[particles_ind[i]]                           = 0.;
            k_p[particles_ind[i]]                           = 0.;
            c_PI[particles_ind[i]]                          = 0.;
            c_PIdot[particles_ind[i]]                       = 0.;
            c_Mdot[particles_ind[i]]                        = 0.;
            c_M[particles_ind[i]]                           = 0.;
        }


        // update total number of particle nodes
        tot_part_nodes = particles_ind.size() + particle_interfacial_nodes.size() / 2;  

        counter1++;
        if (counter1 >= 10000000){
            std::cout << "--- PARTICLE ITERATION THRESHOLD ---" << std::endl;
        }
        if (tot_part_nodes >= n_particle_nodes){
            std::cout << "N_VOL_NODES: "    << N_VOL_NODES                          << std::endl;
            std::cout << "tot_part_nodes: " << tot_part_nodes                       << std::endl;
            std::cout << "solids loading: " << (1.0 * tot_part_nodes / N_VOL_NODES) << std::endl;
        }
    }

    std::cout << "-------------------------------------------"                          << std::endl;
    std::cout << "number of particles generated: "  << counter1                         << std::endl;
    std::cout << "solids loading: "                 << n_particle_nodes / N_VOL_NODES   << std::endl;
    std::cout << "n_particle_nodes: "               << n_particle_nodes                 << std::endl;
    std::cout << "particles_ind.size(): "           << particles_ind.size()             << std::endl;
    std::cout << "-------------------------------------------"                          << std::endl;
}


void Voxel::ComputeRxnRateConstants() {
    /*
     *   @updateVec - updates the reaction rate constants:
     *                          - f_free_volume
     *                          - k_p
     *                          - k_t
     *                for every voxel.
     */

    // initialize intermediate values
    double vT, phi_M, phi_P, k_tr, denom;

    // loop through all voxels
    for (int i=0; i<N_VOL_NODES; i++){

        // compute reaction rate constants for resin
        if (material_type[i] != 0){

            // bowman (1) equation 20
            vT = c_M[i]*mw_M/rho_M + (c_M0-c_M[i])*mw_M/rho_P;

            // bowman (1) equation 22
            phi_M = c_M[i]*mw_M/rho_M/vT;

            // bowman (1) equation 23
            phi_P = (c_M0-c_M[i])*mw_M/rho_P/vT;

            // bowman (1) equation 24
            f_free_volume[i] = 0.025 + alpha_M*phi_M*(theta[i]-theta_gM) + alpha_P*phi_P*(theta[i]-theta_gP);

            // compute temperature dependent rate constants
            // bowman (1) equation 17
            k_p[i] = k_P0*exp(-E_P / Rg / theta[i]) / (1 + exp(A_Dp * (1/f_free_volume[i] - 1/f_cp)));
            // k_p[i] = k_P0 / (1 + exp(A_Dp * (1/f_free_volume[i] - 1/f_cp)));
            // k_i[i] = k_I0*exp(-E_I / Rg / theta[i]) / (1 + exp(A_I  * (1/f_free_volume[i] - 1/f_ci)));
            k_i[i] = k_I0; 
            // k_i[i] = k_I0; // / (1 + exp(A_I  * (1/f_free_volume[i] - 1/f_ci))); // taki method 

            // bowman (1) equation 18

            k_tr   = R_rd * k_p[i] * c_M[i];
            denom  = (k_tr / (k_T0*exp(-E_T/Rg/theta[i])) + exp(-A_Dt*(1/f_free_volume[i] - 1/f_ct)));
            k_t[i] = k_T0*exp(-E_T/Rg/theta[i]) / (1+1/denom);
            // k_t[i] = k_T0 / (1 + 1 / (R_rd * k_p[i] * c_M[i] / (k_T0) + exp(-A_Dt*(1/f_free_volume[i] - 1/f_ct))));
            

        }else{
            // compute reaction rate constants for particles
            f_free_volume[i] = 0.;
            k_t[i]           = 0.;
            k_p[i]           = 0.;
            k_i[i]           = 0.;
        };
    }
}


double Voxel::IRate(std::vector<double> &conc_PI, double I0, double z, int node) const {
    if (material_type[node] == 1){
        // material is ugap
        // return (-phi * eps * I0 * conc_PI[node] * exp( -eps*conc_PI[node]*z) / 2);
        return (-phi * eps * I0 * conc_PI[node] * exp( -eps*conc_PI[node]*z) / 2);
    }
    
    else if (material_type[node] == 2){
        // material is interfacial
        return (-phi * eps * I0 * conc_PI[node] * exp( -eps*conc_PI[node]*z) / 2);
    }
    else{
        // material is particle
        return 0.;
    }

}


double Voxel::PIdotRate(std::vector<double> &conc_PIdot,
                                 std::vector<double> &conc_PI,
                                 std::vector<double> &conc_M,
                                 double I0, double z, int node){
    /*          
    
        equation 2
        d[PI]/dt = phi*eps*[PI]I - k[PIdot][M]

    */
    if (material_type[node] == 1){
        // material is ugap
        double diffuse, Dm_avg;
        double diffusivity[6]; 

        // compute average diffusivity values for each first order derivative
        diffusivity[0] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node + 1])             + exp(-Am / f_free_volume[node]));
        diffusivity[1] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node - 1])             + exp(-Am / f_free_volume[node]));
        diffusivity[2] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node + nodes])         + exp(-Am / f_free_volume[node]));
        diffusivity[3] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node - nodes])         + exp(-Am / f_free_volume[node]));
        diffusivity[4] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node + N_PLANE_NODES]) + exp(-Am / f_free_volume[node]));
        diffusivity[5] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node - N_PLANE_NODES]) + exp(-Am / f_free_volume[node]));

       // compute chemical diffusion as a function of variable diffusivity
        double denom = 1 / h / h; 

        diffuse = (  diffusivity[0] * (conc_PI[node+1]               - conc_PI[node])
                   - diffusivity[1] * (conc_PI[node]                 - conc_PI[node - 1])
                   + diffusivity[2] * (conc_PI[node + nodes]         - conc_PI[node])
                   - diffusivity[3] * (conc_PI[node]                 - conc_PI[node - nodes])
                   + diffusivity[4] * (conc_PI[node + N_PLANE_NODES] - conc_PI[node])
                   - diffusivity[5] * (conc_PI[node]                 - conc_PI[node - N_PLANE_NODES])
                   ) * denom;


        diff_pdot[node] = diffuse; 

        return (phi*eps*I0*conc_PI[node]*exp(-eps*conc_PI[node]*z) - k_i[node]*conc_PIdot[node]*conc_M[node] + diffuse);
    }
    
    else if (material_type[node] == 2){
        // material is interface
        return (phi*eps*I0*conc_PI[node]*exp(-eps*conc_PI[node]*z) - k_i[node]*conc_PIdot[node]*conc_M[node]);
    }

    else{
        // materials is particle
        return 0.;
    }

}


// equation 3: d[Mdot]/dt = k[PIdot][M] + ∇(k∇_x [Mdot]) - kt [Mdot]^2
double Voxel::MdotRate(std::vector<double> &conc_Mdot,
                              std::vector<double> &conc_PIdot,
                              std::vector<double> &conc_M,
                              int node) {
    // if material is ugap resin
    if (material_type[node] == 1){
        // material is resin
        double term1, term2, term3, Dm_avg;
        double diffusivity[6]; 
        term1 = k_i[node]*c_PIdot[node]*c_M[node];
        term2 = k_t[node]*conc_Mdot[node]*conc_Mdot[node];

        // compute average diffusivity values for each first order derivative
        diffusivity[0] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node + 1])             + exp(-Am / f_free_volume[node]));
        diffusivity[1] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node - 1])             + exp(-Am / f_free_volume[node]));
        diffusivity[2] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node + nodes])         + exp(-Am / f_free_volume[node]));
        diffusivity[3] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node - nodes])         + exp(-Am / f_free_volume[node]));
        diffusivity[4] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node + N_PLANE_NODES]) + exp(-Am / f_free_volume[node]));
        diffusivity[5] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node - N_PLANE_NODES]) + exp(-Am / f_free_volume[node]));

        // compute chemical diffusion as a function of variable diffusivity
        double denom = 1 / h / h; 
        term3 = (   diffusivity[0] * (conc_Mdot[node+1]       - conc_Mdot[node])
                  - diffusivity[1] * (conc_Mdot[node]         - conc_Mdot[node - 1])
                  + diffusivity[2] * (conc_Mdot[node + nodes] - conc_Mdot[node])
                  - diffusivity[3] * (conc_Mdot[node] - conc_Mdot[node - nodes])
                  + diffusivity[4] * (conc_Mdot[node + N_PLANE_NODES] - conc_Mdot[node])
                  - diffusivity[5] * (conc_Mdot[node] - conc_Mdot[node - N_PLANE_NODES])
                  ) * denom; 

        diff_mdot[node] = term3;
        return (term1 - term2 + term3);
    } 

    // material is interface 
    else if (material_type[node] == 2){
        return k_i[node]*c_PIdot[node]*c_M[node] - k_t[node]*conc_Mdot[node]*conc_Mdot[node]; 
    }
    
    // material is a particle
    else{
        return 0.;
    }
}


// equation 4: d[M]/dt = ∇(k∇_x [M]) - k[PI][M] - k[M][Mdot]   
double Voxel::MRate(std::vector<double> &conc_M,
                                  std::vector<double> &conc_Mdot,
                                  std::vector<double> &conc_PIdot,
                                  int node){

    // if material is ugap resin 
    if (material_type[node] == 1){
        // material is resin
        double diffuse, consume, Dm_avg;
        double diffusivity[6]; 

        consume =   (k_p[node]*conc_Mdot[node]*conc_M[node])
                  + (k_i[node]*conc_PIdot[node]*conc_M[node]);

        // compute average diffusivity values for each first order derivative
        diffusivity[0] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node + 1])             + exp(-Am / f_free_volume[node]));
        diffusivity[1] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node - 1])             + exp(-Am / f_free_volume[node]));
        diffusivity[2] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node + nodes])         + exp(-Am / f_free_volume[node]));
        diffusivity[3] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node - nodes])         + exp(-Am / f_free_volume[node]));
        diffusivity[4] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node + N_PLANE_NODES]) + exp(-Am / f_free_volume[node]));
        diffusivity[5] = 0.5 * Dm0 * (exp(-Am / f_free_volume[node - N_PLANE_NODES]) + exp(-Am / f_free_volume[node]));

        // compute chemical diffusion taking into account the average diffusivity values
        double denom = 1 / h / h; 
        diffuse = (  diffusivity[0]*(conc_M[node+1]-conc_M[node]) 
                   - diffusivity[1]*(conc_M[node]-conc_M[node-1])
                   + diffusivity[2]*(conc_M[node+nodes]-conc_M[node])
                   - diffusivity[3]*(conc_M[node]-conc_M[node-nodes])
                   + diffusivity[4]*(conc_M[node+N_PLANE_NODES]-conc_M[node])
                   - diffusivity[5]*(conc_M[node]-conc_M[node-N_PLANE_NODES])
                   ) * denom;
        
        // average diffusion coefficient
        Dm_avg = Dm0 * (  exp(-Am / f_free_volume[node - N_PLANE_NODES])
                        + exp(-Am / f_free_volume[node - nodes])
                        + exp(-Am / f_free_volume[node - 1])
                        + 6 * exp(-Am / f_free_volume[node])
                        + exp(-Am / f_free_volume[node + N_PLANE_NODES])
                        + exp(-Am / f_free_volume[node + nodes])
                        + exp(-Am / f_free_volume[node + 1])
                        ) / 12.; 
        diffuse = Dm_avg
                  * (            conc_M[node - N_PLANE_NODES]
                           +     conc_M[node - nodes]
                           +     conc_M[node - 1]
                           - 6 * conc_M[node]
                           +     conc_M[node + N_PLANE_NODES]
                           +     conc_M[node + nodes]
                           +     conc_M[node + 1]
                  ) / h / h;
        diff_m[node] = diffuse;
        return (diffuse - consume);
    }

    // material is interface - no diffusion, return only consumptions
    else if (material_type[node] == 2){
        return -(k_p[node]*conc_Mdot[node]*conc_M[node]) - (k_i[node]*conc_PIdot[node]*conc_M[node]);
    }

    // material is particle - no reacton
    else{
        return 0.;
    }
}

// equation 5: dθ/dt = ( ∇_x·(K ∇_x θ) + k_p [M] [M_n·] ΔH + ε [PI] I ) / ρ / C 
double Voxel::TempRate(std::vector<double> &temperature,
                       std::vector<double> &conc_M,
                       std::vector<double> &conc_Mdot,
                       std::vector<double> &conc_PI,
                       std::vector<double> &conc_PIdot,
                       double intensity, int node){

    double heat_diffuse, heat_rxn, heat_uv;
    double therm_cond_avg[6]; 

    // compute average thermal conductivity values for each first order derivative
    therm_cond_avg[0] = 0.5 * (therm_cond[node+1]             + therm_cond[node]);
    therm_cond_avg[1] = 0.5 * (therm_cond[node-1]             + therm_cond[node]);
    therm_cond_avg[2] = 0.5 * (therm_cond[node+nodes]         + therm_cond[node]);
    therm_cond_avg[3] = 0.5 * (therm_cond[node-nodes]         + therm_cond[node]);
    therm_cond_avg[4] = 0.5 * (therm_cond[node+N_PLANE_NODES] + therm_cond[node]);
    therm_cond_avg[5] = 0.5 * (therm_cond[node-N_PLANE_NODES] + therm_cond[node]);

    // compute thermal diffusion taking into account the average thermal properties
    double denom = 1 / h / h / density[node] / heat_capacity[node];
    heat_diffuse = (  therm_cond_avg[0]*(temperature[node+1]-temperature[node]) 
                    - therm_cond_avg[1]*(temperature[node]-temperature[node-1])
                    + therm_cond_avg[2]*(temperature[node+nodes]-temperature[node])
                    - therm_cond_avg[3]*(temperature[node]-temperature[node-nodes])
                    + therm_cond_avg[4]*(temperature[node+N_PLANE_NODES]-temperature[node])
                    - therm_cond_avg[5]*(temperature[node]-temperature[node-N_PLANE_NODES])
                    ) * denom; 

    // compute the heat release by all exothermic (bond formation) reactions
    heat_rxn = (  k_i[node] * conc_PIdot[node] * conc_M[node]
                + k_p[node] * conc_Mdot[node]  * conc_M[node]
                + k_t[node] * conc_Mdot[node]  * conc_Mdot[node]
                ) * dHp;

    // material is resin
    if (material_type[node] == 1){
        heat_uv = eps
                * intensity
                * conc_PI[node]
                * exp(  -eps*conc_PI[node]*(len_block-current_coords[2]*coord_map_const)  );
    }

    // material is interface
    else if (material_type[node] == 2){
        heat_uv = (   eps
                    * intensity
                    * conc_PI[node]
                    * exp(  -eps*conc_PI[node]*(len_block-current_coords[2]*coord_map_const)  )
                    + eps_nacl
                    * intensity
                    * exp(  -eps_nacl*(len_block-current_coords[2]*coord_map_const)  )
                    ) / 2;
    }
    else{
        // material is particle
        heat_uv = eps_nacl
                * intensity
                * exp(  -eps_nacl*(len_block-current_coords[2]*coord_map_const)  );
    }
    // heat_uv = eps
    //             * intensity
    //             * conc_PI[node]
    //             * exp(  -eps*conc_PI[node]*(len_block-current_coords[2]*coord_map_const)  );

    diff_theta[node] = heat_diffuse;
    return heat_diffuse + (heat_rxn + heat_uv) / density[node] / heat_capacity[node];
};


void Voxel::SolveSystem(std::vector<double> &c_PI_next,
                        std::vector<double> &c_PIdot_next,
                        std::vector<double> &c_Mdot_next,
                        std::vector<double> &c_M_next,
                        std::vector<double> &theta_next,
                        double I0, double dt, int method){
    double heat_diffuse, heat_rxn, heat_uv;
    int node;

    // FORWARD EULER
    if (method == 0){
        double depth = 0;
        for (int node = 0; node < N_VOL_NODES; node++){
            Node2Coord(node, current_coords); 
            int i = current_coords[0]; 
            int j = current_coords[1]; 
            int k = current_coords[2];

            depth = len_block-current_coords[2]*coord_map_const;

            // INTERNAL NODES
            if (    current_coords[0] != 0 && current_coords[0] != (nodes-1)
                 && current_coords[1] != 0 && current_coords[1] != (nodes-1)
                 && current_coords[2] != 0 && current_coords[2] != (nodes-1)){
                // std::cout << "\nINTERNAL NODE: " << std::endl; 

                // solve equations 1-5
                c_PI_next[node]    = c_PI[node] + dt*IRate(c_PI, I0, depth, node); 
                c_PIdot_next[node] = c_PIdot[node] + dt*PIdotRate(c_PIdot, c_PI, c_M, I0, depth, node); 
                c_Mdot_next[node]  = c_Mdot[node] + dt*MdotRate(c_Mdot, c_PIdot, c_M, node); 
                c_M_next[node]     = c_Mdot[node] + dt*MRate(c_M, c_Mdot, c_PIdot, node); 
                theta_next[node]   = theta[node] + dt*TempRate(theta, c_M, c_Mdot, c_PI, c_PIdot, I0, node);
                    }
            // BOUNDARY NODES
            else if (   current_coords[0] == 0 or current_coords[0] == (nodes-1)
                     or current_coords[1] == 0 or current_coords[1] == (nodes-1)
                     or current_coords[2] == 0 or current_coords[2] == (nodes-1)){
                // std::cout << "\nBOUNDARY NODE: " << std::endl; 
                

                // solve non-spatially dependent equations
                c_PI_next[node] = c_PI[node] + dt*IRate(c_PI, I0, depth, node); 

                // check material type is interfacial, skip spatial dependencies for reactions
                if (material_type[node] == 2){
                    c_PIdot_next[node] = c_PIdot[node] + dt*PIdotRate(c_PIdot, c_PI, c_M, I0, depth, node); 
                    c_Mdot_next[node]  = c_Mdot[node] + dt*MdotRate(c_Mdot, c_PIdot, c_M, node); 
                    c_M_next[node]     = c_Mdot[node] + dt*MRate(c_M, c_Mdot, c_PIdot, node); 
                }

                // ENFORCE NEUMANN BOUNDARY CONDITIONS

                // bottom face
                if (i == 0){
                    if (material_type[node] == 1){
                        // ugap resin
                        c_PIdot[node] = c_PIdot[node+N_PLANE_NODES]; 
                        c_Mdot[node]  = c_Mdot[node+N_PLANE_NODES]; 
                        c_M[node]     = c_M[node+N_PLANE_NODES];  
                    }
                }

                // top face
                else if (i == (nodes-1)){
                    // ugap resin
                    if (material_type[node] == 1){
                        c_PIdot[node] = c_PIdot[node-N_PLANE_NODES]; 
                        c_Mdot[node]  = c_Mdot[node-N_PLANE_NODES];
                        c_M[node]     = c_M[node-N_PLANE_NODES]; 
                        }
                } 
                
                // wall 1: front wall 
                else if (j == 0){
                    // ugap resin
                    if (material_type[node] == 1){
                        c_PIdot[node] = c_PIdot[node+nodes]; 
                        c_Mdot[node]  = c_Mdot[node+nodes]; 
                        c_M[node]     = c_M[node+nodes];    
                    }
                }
                
                // wall 3: back wall
                else if (j == (nodes-1)){
                    // ugap resin 
                    if (material_type[node] == 1){
                        c_PIdot[node] = c_PIdot[node-nodes];
                        c_Mdot[node]  = c_Mdot[node-nodes]; 
                        c_M[node]     = c_M[node-nodes];
                    }
                }

                // wall 2: left wall
                else if (k == 0){
                    if (material_type[node] == 1){
                        // ugap resin
                        c_PIdot[node] = c_PIdot[node+1]; 
                        c_Mdot[node]  = c_Mdot[node+1]; 
                        c_M[node]     = c_M[node+1]; 
                    }
                }

                // wall 4: right wall
                else if (k == (nodes-1)){
                    if (material_type[node] == 1){
                        // ugap resin
                        c_PIdot[node] = c_PIdot[node-1]; 
                        c_Mdot[node]  = c_Mdot[node-1]; 
                        c_M[node]     = c_M[node-1]; 
                    }
                }else{
                    throw std::invalid_argument("--- forward euler: boundary condition error ---");
                }
            }else{
                std::cout << "\n--- error location ---" << std::endl;
                std::cout << "i : " << i << " | j: " << j << " | k: " << k << std::endl; 
                throw std::invalid_argument("--- forward euler: node error ---");
            }
        }
    }

    // backward euler
    else if (method == 1){

        std::vector<double> c_PI_0(N_VOL_NODES),
                            c_PI_1(N_VOL_NODES),
                            c_PIdot_0(N_VOL_NODES),
                            c_PIdot_1(N_VOL_NODES),
                            c_Mdot_0(N_VOL_NODES),
                            c_Mdot_1(N_VOL_NODES),
                            c_M_0(N_VOL_NODES),
                            c_M_1(N_VOL_NODES),
                            theta_0(N_VOL_NODES),
                            theta_1(N_VOL_NODES);

        for (int node = 0; node < N_VOL_NODES; node++){
            c_PI_0[node] = c_PI[node];
            c_PI_1[node] = c_PI[node];
            c_PIdot_0[node] = c_PIdot[node];
            c_PIdot_1[node] = c_PIdot[node];
            c_Mdot_0[node] = c_Mdot[node];
            c_Mdot_1[node] = c_Mdot[node];
            c_M_0[node] = c_M[node];
            c_M_1[node] = c_M[node];
            theta_0[node] = theta[node];
            theta_1[node] = theta[node];
        }

        int count = 0;
        double error = 100;
        double depth = 0; 

        // fixed point iteration

        while (error > tol){

            // cap iterations
            if (count > thresh){
                std::cout << "error: " << error << std::endl;
                throw std::invalid_argument("--- SolveSystem (backward euler) did not converge ---");
            }

            double err_step = 0;
            for (int node = 0; node < N_VOL_NODES; node++){

                // get node coordinates
                Node2Coord(node, current_coords);
                int i = current_coords[0]; 
                int j = current_coords[1]; 
                int k = current_coords[2];

                // internal nodes
                if (    i != 0 && i != (nodes-1)
                     && j != 0 && j != (nodes-1)
                     && k != 0 && k != (nodes-1)){

                    // solve equation 1
                    c_PI_1[node] = c_PI[node] + dt* IRate(c_PI_0,
                                                            I0,
                                                            len_block-current_coords[2]*coord_map_const,
                                                            node);

                    err_step +=   SquaredDiff(c_PI_0[node], c_PI_1[node]);

                    // solve equation 2
                    c_PIdot_1[node] = c_PIdot[node] + dt* PIdotRate(c_PIdot_0,
                                                                    c_PI_0,
                                                                    c_M_0,
                                                                    I0,
                                                                    len_block-current_coords[2]*coord_map_const,
                                                                    node);
                    err_step += SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                    // solve equation 3
                    c_Mdot_1[node] = c_Mdot[node] + dt* MdotRate(c_Mdot_0,
                                                                    c_PIdot_0,
                                                                    c_M_0, 
                                                                    node);

                    err_step += SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]);


                    // solve equation 4
                    c_M_1[node] = c_M[node] + dt* MRate(c_M_0,
                                                        c_Mdot_0,
                                                        c_PIdot_0,
                                                        node);
                    err_step += SquaredDiff(c_M_0[node], c_M_1[node]);

                    // solve equation 5
                    theta_1[node] = theta[node] + dt*TempRate(theta_0,
                                                                c_M_0,
                                                                c_Mdot_0,
                                                                c_PI_0,
                                                                c_PIdot, 
                                                                I0, node);
                    err_step += SquaredDiff(theta_0[node], theta_1[node]);
                }
                // BOUNDARY NODES
                else if (   i == 0 || i == (nodes-1)
                         || j == 0 || j == (nodes-1)
                         || k == 0 || k == (nodes-1)){

                    // solve non-spatially dependent equations: equation 1
                    c_PI_1[node] = c_PI[node] + dt*IRate(c_PI_0,
                                                            I0,
                                                            len_block-current_coords[2]*coord_map_const,
                                                            node);
                    err_step += SquaredDiff(c_PI_0[node], c_PI_1[node]);

                    // check material type is interfacial, skip spatial dependencies for reactions
                    if (material_type[node] == 2){
                        // solve equation 2
                        c_PIdot_1[node] = c_PIdot[node] + dt*PIdotRate(c_PIdot_0,
                                                                c_PI_0,
                                                                c_M_0,
                                                                I0,
                                                                len_block-current_coords[2]*coord_map_const,
                                                                node);
                        err_step += SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                        // solve equation 3
                        c_Mdot_1[node] = c_Mdot[node] + dt*MdotRate(c_Mdot_0,
                                                                    c_PIdot_0,
                                                                    c_M_0,
                                                                    node);
                        err_step += SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]);

                        // solve equation 4
                        c_M_1[node] = c_M[node] + dt*MRate(c_M_0, c_Mdot_0, c_PIdot_0, node);
                        err_step += SquaredDiff(c_M_0[node], c_M_1[node]);

                    }

                    // ENFORCE NEUMANN BOUNDARY CONDITIONS
                    // bottom face
                    if (i == 0){
                        if (material_type[node] == 1){
                            // ugap resin
                            
                            c_PIdot_1[node] = c_PIdot_0[node+N_PLANE_NODES]; // first order approximation
                            err_step +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node] = c_Mdot_0[node+N_PLANE_NODES];
                            err_step +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                            c_M_1[node] = c_M_0[node+N_PLANE_NODES];
                            err_step +=   SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }

                    // top face
                    else if (i == (nodes-1)){
                        // ugap resin
                        if (material_type[node] == 1){
                            c_PIdot_1[node] =  c_PIdot_0[node-N_PLANE_NODES];   // first order approximation
                            err_step +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node] = c_Mdot_0[node-N_PLANE_NODES]; 
                            err_step +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                            c_M_1[node] = c_M_0[node-N_PLANE_NODES];
                            err_step +=   SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }

                    // wall 1: front wall
                    else if (j == 0){
                        if (material_type[node] == 1){
                            c_PIdot_1[node] =  c_PIdot_0[node+nodes]; // first order approximation
                            err_step +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node] = c_Mdot_0[node+nodes]; 
                            err_step +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]);

                            c_M_1[node] = c_M_0[node+nodes];
                            err_step +=   SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }

                    // wall 3: back wall
                    else if (j == (nodes-1)){
                        if (material_type[node] == 1){

                            c_PIdot_1[node] =  c_PIdot_0[node-nodes]; 
                            err_step       +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node]  = c_Mdot_0[node-nodes]; 
                            err_step       +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                            c_M_1[node]     = c_M_0[node-nodes];
                            err_step       +=   SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }

                    // wall 2: left wall
                    else if (k == 0){
                        if (material_type[node] == 1){
                            c_PIdot_1[node] =  c_PIdot_0[node+1]; 
                            err_step       +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node]  = c_Mdot_0[node+1]; 
                            err_step       +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                            c_M_1[node]     = c_M_0[node+1];
                            err_step       += SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }

                    // wall 4: right wall
                    else if (k == (nodes-1)){
                        if (material_type[node] == 1){
                            c_PIdot_1[node] =  c_PIdot_0[node-1]; 
                            err_step +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node] = c_Mdot_0[node-1]; 
                            err_step +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                            c_M_1[node] = c_M_0[node-1];
                            err_step +=   SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }else{
                        throw std::invalid_argument("--- Backward Euler: boundary condition error ---");
                    }
                }else{
                    // something is wrong..
                    throw std::invalid_argument("--- FIXED POINT NODE BOUNDARY ISSUE ---");
                }
            }

            error = sqrt(err_step);
            count++;

            if (error > tol){
                c_PI_0 = c_PI_1;
                c_PIdot_0 = c_PIdot_1;
                c_Mdot_0 = c_Mdot_1;
                c_M_0 = c_M_1;
                theta_0 = theta_1;
            }else{
                c_PI_next = c_PI_1;
                c_PIdot_next = c_PIdot_1;
                c_Mdot_next = c_Mdot_1;
                c_M_next = c_M_1;
                theta_next = theta_1;
            }
        }
    }

    // trapezoidal
    else if (method == 2){
        
        // declare implicit vectors
        std::vector<double> c_PI_0(N_VOL_NODES),
                            c_PI_1(N_VOL_NODES),
                            c_PIdot_0(N_VOL_NODES),
                            c_PIdot_1(N_VOL_NODES),
                            c_Mdot_0(N_VOL_NODES),
                            c_Mdot_1(N_VOL_NODES),
                            c_M_0(N_VOL_NODES),
                            c_M_1(N_VOL_NODES),
                            theta_0(N_VOL_NODES),
                            theta_1(N_VOL_NODES);

        // initialize implicit vectors
        for (int node = 0; node < N_VOL_NODES; node++){
            c_PI_0[node]    = c_PI[node];
            c_PI_1[node]    = c_PI[node];
            c_PIdot_0[node] = c_PIdot[node];
            c_PIdot_1[node] = c_PIdot[node];
            c_Mdot_0[node]  = c_Mdot[node];
            c_Mdot_1[node]  = c_Mdot[node];
            c_M_0[node]     = c_M[node];
            c_M_1[node]     = c_M[node];
            theta_0[node]   = theta[node];
            theta_1[node]   = theta[node];
        }

        // set trapezoidal hyper parameter: 1-FEuler, 0-BEuler, 0.5-Trap
        float psi = 0.5;
        int count = 0;
        double error = 100;
        double err_step;
        double depth = 0;

        // fixed point iteration
        while (error > tol){

            // cap iterations
            if (count > thresh){
                throw std::invalid_argument("--- SolveSystem (trapezoidal) did not converge ---");
            }

            err_step = 0;

            for (int node = 0; node < N_VOL_NODES; node++){

                // map node to coordinates
                Node2Coord(node, current_coords); 
                int i = current_coords[0]; 
                int j = current_coords[1]; 
                int k = current_coords[2]; 

                depth = len_block-current_coords[2]*coord_map_const;

                // internal nodes
                if (   i != 0 && i != (nodes-1)
                    && j != 0 && j != (nodes-1)
                    && k != 0 && k != (nodes-1)){

                    // solve equation 1
                    c_PI_1[node] =   c_PI[node] + dt * (  (1-psi) * IRate(c_PI_0, I0, depth, node)
                                                            +   psi   * IRate(c_PI, I0, depth, node));

                    err_step += SquaredDiff(c_PI_0[node], c_PI_1[node]);


                    // solve equation 2
                    c_PIdot_1[node] = c_PIdot[node] + dt * (   (1-psi) * PIdotRate(c_PIdot_0, c_PI_0, c_M_0, I0, depth, node)
                                                                +   psi   * PIdotRate(c_PIdot, c_PI, c_M, I0, depth, node));

                    err_step += SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                    // solve equation 3
                    c_Mdot_1[node] = c_Mdot[node] + dt * (   (1-psi) * MdotRate(c_Mdot_0, c_PIdot_0, c_M_0, node)
                                                            +   psi   * MdotRate(c_Mdot, c_PIdot, c_M, node));

                    err_step += SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]);

                    // solve equation 4
                    c_M_1[node] = c_M[node] + dt * (   (1-psi) * MRate(c_M_0, c_Mdot_0, c_PIdot_0, node)
                                                        +   psi   * MRate(c_M, c_Mdot, c_PIdot, node));

                    err_step += SquaredDiff(c_M_0[node], c_M_1[node]);

                    // solve equation 5
                    theta_1[node] = theta[node] + dt* (   (1-psi) * TempRate(theta_0, c_M_0, c_Mdot_0, c_PI_0, c_PIdot, I0, node)
                                                        +   psi   * TempRate(theta, c_M, c_Mdot, c_PI, c_PIdot, I0, node));

                    err_step += SquaredDiff(theta_0[node], theta_1[node]);

                }

                // BOUNDARY NODES
                else if (   i == 0 || i == (nodes-1)
                         || j == 0 || j == (nodes-1)
                         || k == 0 || k == (nodes-1)){

                    // solve non-spatially dependent equations: equation 1
                    c_PI_1[node] = c_PI[node]
                                    + dt*(  (1-psi) * IRate(c_PI_0, I0, depth, node)
                                            +   psi   * IRate(c_PI, I0, depth, node));
                    err_step +=   SquaredDiff(c_PI_0[node], c_PI_1[node]);
                    
                    // check material type is interfacial, skip spatial dependencies for reactions
                    if (material_type[node] == 2){
                        // solve equation 2
                        c_PIdot_1[node] = c_PIdot[node] + dt * (   (1-psi) * PIdotRate(c_PIdot_0, c_PI_0, c_M_0, I0, depth, node)
                                                                +   psi   * PIdotRate(c_PIdot, c_PI, c_M, I0, depth, node));
                        err_step +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                        // solve equation 3
                        c_Mdot_1[node] = c_Mdot[node] + dt * (   (1-psi) * MdotRate(c_Mdot_0, c_PIdot_0, c_M_0, node)
                                                                +   psi   * MdotRate(c_Mdot, c_PIdot, c_M, node));
                        err_step +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]);

                        // solve equation 4
                        c_M_1[node] = c_M[node] + dt * (   (1-psi) * MRate(c_M_0, c_Mdot_0, c_PIdot_0, node)
                                                        +   psi   * MRate(c_M, c_Mdot, c_PIdot, node));
                        err_step += SquaredDiff(c_M_0[node], c_M_1[node]);

                    }

                    // ENFORCE NEUMANN BOUNDARY CONDITIONS
                    // bottom face
                    double pre_1 = 4 / 3, pre_2 = 1 / 3; 
                    if (i == 0){
                        // if material is UGAP resin
                        if (material_type[node] == 1){
                            // apply neumann bc using a second order approximation
                            c_PIdot_1[node] = pre_1 * c_PIdot_1[node+N_PLANE_NODES] - pre_2 * c_PIdot_1[node+N_PLANE_NODES+N_PLANE_NODES]; 
                            err_step       += SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node] = pre_1 * c_Mdot_1[node+N_PLANE_NODES] - pre_2 * c_Mdot_1[node+N_PLANE_NODES+N_PLANE_NODES]; // second order approximation
                            err_step      += SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 
                            
                            c_M_1[node] = pre_1 * c_M_1[node+N_PLANE_NODES] - pre_2 * c_M_1[node+N_PLANE_NODES+N_PLANE_NODES]; // second order approximation
                            err_step +=   SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }
                    // top face
                    else if (i == (nodes-1)){
                        if (material_type[node] == 1){
                            // apply neumann bc using a second order approximation
                            c_PIdot_1[node] = pre_1 * c_PIdot_1[node-N_PLANE_NODES] - pre_2 * c_PIdot_1[node-N_PLANE_NODES-N_PLANE_NODES]; 
                            err_step +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node] = pre_1 * c_Mdot_1[node-N_PLANE_NODES] - pre_2 * c_Mdot_1[node-N_PLANE_NODES-N_PLANE_NODES]; 
                            err_step +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                            c_M_1[node] = pre_1 * c_M_1[node-N_PLANE_NODES] - pre_2 * c_M_1[node-N_PLANE_NODES-N_PLANE_NODES]; 
                            err_step +=   SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }

                    // wall 1: front wall
                    else if (j == 0){
                        if (material_type[node] == 1){
                            // apply neumann bc using a second order approximation
                            c_PIdot_1[node] = pre_1 * c_PIdot_1[node+nodes] - pre_2 * c_PIdot_1[node+nodes+nodes]; 
                            err_step +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node] = pre_1 * c_Mdot_1[node+nodes] - pre_2 * c_Mdot_1[node+nodes+nodes]; 
                            err_step +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]);

                            c_M_1[node] = pre_1 * c_M_1[node+nodes] - pre_2 * c_M_1[node+nodes+nodes]; 
                            err_step +=   SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }

                    // wall 3: back wall
                    else if (j == (nodes-1)){
                        if (material_type[node] == 1){
                            // apply neumann bc using a second order approximation
                            c_PIdot_1[node] = pre_1 * c_PIdot_1[node-nodes] - pre_2 * c_PIdot_1[node-nodes-nodes];
                            err_step +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node] = pre_1 * c_Mdot_1[node-nodes] - pre_2 * c_Mdot_1[node-nodes-nodes];
                            err_step +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                            c_M_1[node] = pre_1 * c_M_1[node-nodes] - pre_2 * c_M_1[node-nodes-nodes]; 
                            err_step +=   SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }

                    // wall 2: left wall
                    else if (k == 0){
                        if (material_type[node] == 1){
                            // apply neumann bc using a second order approximation
                            c_PIdot_1[node] = pre_1 * c_PIdot_1[node+1] - pre_2 * c_PIdot_1[node+1+1]; 
                            err_step +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node] = pre_1 * c_Mdot_1[node+1] - pre_2 * c_Mdot_1[node+1+1];
                            err_step +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                            c_M_1[node] = pre_1 * c_M_1[node+1] - pre_2 * c_M_1[node+1+1];
                            err_step += SquaredDiff(c_M_0[node], c_M_1[node]);
                        }
                    }

                    // wall 4: right wall
                    else if (k == (nodes-1)){
                        if (material_type[node] == 1){
                            // apply neumann bc using a second order approximation
                            c_PIdot_1[node] = pre_1 * c_PIdot_1[node-1] - pre_2 * c_PIdot_1[node-1-1]; 
                            err_step +=   SquaredDiff(c_PIdot_0[node], c_PIdot_1[node]);

                            c_Mdot_1[node] = pre_1 * c_Mdot_1[node-1] - pre_2 * c_Mdot_1[node-1-1]; 
                            err_step +=   SquaredDiff(c_Mdot_0[node], c_Mdot_1[node]); 

                            c_M_1[node] = pre_1 * c_M_1[node-1] - pre_2 * c_M_1[node-1-1];
                            err_step +=   SquaredDiff(c_M_0[node], c_M_1[node]);
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

            if (error > tol){
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
}


void Voxel::Config2File(double dt){
    
    // write to file
    print_sim_config.open(file_path + "sim_config/sim_config.txt");
    print_sim_config << "Simulation Configuration\n" << std::endl; 

    print_sim_config << "==================================" << std::endl;

    print_sim_config << "Initial concentrations (mol/m^3): " << std::endl;
    print_sim_config << "c_M0: " << c_M0 << std::endl;
    print_sim_config << "c_PI0: " << c_PI0 << std::endl;
    print_sim_config << "theta0 " << theta0 << std::endl;
    print_sim_config << "==================================\n" << std::endl;

    print_sim_config << "Numerical parameters" << std::endl;
    print_sim_config << "h: " << h << std::endl;
    print_sim_config << "dt: " << dt << std::endl;
    print_sim_config << "Diffusion CFL: " << Dm0 * dt / h / h << std::endl;
    print_sim_config << "Thermal CFL: " << dt / h / rho_UGAP / Cp_nacl << std::endl;
    

    print_sim_config << "\n==================================" << std::endl;
    print_sim_config << "solids loading: " << (1.0 * particles_ind.size() / N_VOL_NODES) << std::endl;
    print_sim_config << "particles_ind.size(): " << particles_ind.size() << std::endl;
    print_sim_config << "\n==================================" << std::endl;

    print_sim_config.close(); 
}


void Voxel::Density2File(){
    // write to file
    print_density.open(file_path + "density/density.vtk");

    print_density << "# vtk DataFile Version 2.0" << std::endl;
    print_density << "voxel density ugap with particles" << std::endl;
    print_density << "ASCII" << std::endl;
    print_density << "DATASET RECTILINEAR_GRID" << std::endl;
    print_density << "DIMENSIONS " << nodes << " " <<  nodes << " " << nodes << std::endl;

    print_density << "X_COORDINATES " << nodes << " float" << std::endl;
    double dx = 0.;
    int counter = 1;
    for (int i = 0; i < nodes; i++){
        print_density << dx * 1000 << " ";
        if (counter % 6 == 0){
            print_density << std::endl;
        }
        dx += h;
        counter++;
    }
    print_density << std::endl;

    print_density << "Y_COORDINATES " << nodes << " float" << std::endl;
    dx = 0.;
    counter = 1;
    for (int i = 0; i < nodes; i++){
        print_density << dx * 1000 << " ";
        if (counter % 6 == 0){
            print_density << std::endl;
        }
        dx += h;
        counter++;
    }
    print_density << std::endl;

    print_density << "Z_COORDINATES " << nodes << " float" << std::endl;
    dx = 0.;
    counter = 1;
    for (int i = 0; i < nodes; i++){

        print_density << dx * 1000 << " ";
        if (counter % 6 == 0){
            print_density << std::endl;
        }
        dx += h;
        counter++;
    }
    print_density << std::endl;

    print_density << "POINT_DATA " << N_VOL_NODES << std::endl;
    print_density << "SCALARS density float" << std::endl;
    print_density << "LOOKUP_TABLE default" << std::endl;
    counter = 1;
    for (int i = 0; i < N_VOL_NODES; i++){
        print_density << density[i] << " ";
        if (counter % 6 == 0){
            print_density << std::endl;
        }
        counter++;
    }
    print_density.close();

}

void Voxel::AvgConcentrations2File(int counter, 
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
    for (int node = 0; node < N_VOL_NODES; node++){

        // compute average temperature related values over all nodes
        avg_diff_theta  += diff_theta[node];
        avg_tot_theta   += theta_next[node];

        // compute average diffusivity of temperature top and bottom nodes
        if (N_PLANE_NODES < node && node < 2 * N_PLANE_NODES){
            avg_diff_theta_bot += diff_theta[node];
            
        }
        if (N_VOL_NODES - N_PLANE_NODES > node && node > N_VOL_NODES - 2 * N_PLANE_NODES){
            avg_diff_theta_top += diff_theta[node];
            
        }

        // compute average reaction related values over resin nodes
        if (material_type[node] == 1){
            // compute average total concentration
            avg_tot_cPI     += c_PI_next[node];
            avg_tot_cPIdot  += c_PIdot_next[node];
            avg_tot_cMdot   += c_Mdot_next[node];
            avg_tot_cM      += c_M_next[node];
            
            avg_free_volume += f_free_volume[node];
            avg_k_p         += k_p[node];
            avg_k_t         += k_t[node];
            avg_diff_pdot   += diff_pdot[node];
            avg_diff_mdot   += diff_mdot[node];
            avg_diff_m      += diff_m[node];
            


            // compute average bottom concentration
            if (node < N_PLANE_NODES){
                avg_bot_cPI     += c_PI_next[node];
                avg_bot_cPIdot  += c_PIdot_next[node];
                avg_bot_cMdot   += c_Mdot_next[node];
                avg_bot_cM      += c_M_next[node];
                // avg_bot_theta += theta_next[node];
                
                
                
                nodes_bot_resin++; 
                
            }

            if (node < 2 * N_PLANE_NODES && node > N_PLANE_NODES){
                // diffusive terms
                avg_diff_pdot_bot  += diff_pdot[node];
                avg_diff_mdot_bot  += diff_mdot[node];
                avg_diff_m_bot     += diff_m[node];
                
            }

            // compute average top concentration
            if (node > N_VOL_NODES - N_PLANE_NODES){
                avg_top_cPI     += c_PI_next[node];
                avg_top_cPIdot  += c_PIdot_next[node];
                avg_top_cMdot   += c_Mdot_next[node];
                avg_top_cM      += c_M_next[node];
                // avg_top_theta += theta_next[node];
                nodes_top_resin++;
            }

            if (node < N_VOL_NODES - N_PLANE_NODES && node > N_VOL_NODES - 2 * N_PLANE_NODES){
                // diffusive terms
                avg_diff_pdot_top  += diff_pdot[node];
                avg_diff_mdot_top  += diff_mdot[node];
                avg_diff_m_top     += diff_m[node];
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

    avg_tot_theta   /= N_VOL_NODES;
    avg_free_volume /= nodes_resin;
    avg_k_p         /= nodes_resin;
    avg_k_t         /= nodes_resin;

    avg_diff_pdot   /= nodes_resin;
    avg_diff_pdot_top  /= nodes_top_resin;
    avg_diff_pdot_bot  /= nodes_bot_resin;
    
    avg_diff_mdot   /= nodes_resin;
    avg_diff_mdot_top  /= nodes_top_resin;
    avg_diff_mdot_bot  /= nodes_bot_resin;    

    avg_diff_m      /= nodes_resin;
    avg_diff_m_top     /= nodes_top_resin;
    avg_diff_m_bot     /= nodes_bot_resin;

    avg_diff_theta  /= N_VOL_NODES;
    avg_diff_theta_top /= N_VOL_NODES;
    avg_diff_theta_bot /= N_VOL_NODES;


    // open file
    if (counter == 0){std::string file_avg_concentrations = file_path + "python_plotting/avg_concentration_simID_" + std::to_string(sim_id) + ".csv";
        print_avg_concentrations.open(file_avg_concentrations);
        print_avg_concentrations <<       "time, avg_top_cPI, avg_tot_cPI, avg_bot_cPI, ";
        print_avg_concentrations <<       "avg_top_cPIdot, avg_tot_cPIdot, avg_bot_cPIdot, "; 
        print_avg_concentrations <<       "avg_top_cMdot, avg_tot_cMdot, avg_bot_cMdot, ";
        print_avg_concentrations <<       "avg_top_cM, avg_tot_cM, avg_bot_cM, "; 
        print_avg_concentrations <<       "avg_tot_theta, "; 
        print_avg_concentrations <<       "avg_free_volume, avg_k_p, avg_k_t, ";
        print_avg_concentrations <<       "avg_diff_pdot_top, avg_diff_pdot, avg_diff_pdot_bot, ";
        print_avg_concentrations <<       "avg_diff_mdot_top, avg_diff_mdot, avg_diff_mdot_bot, ";
        print_avg_concentrations <<       "avg_diff_m_top, avg_diff_m, avg_diff_m_bot, ";
        print_avg_concentrations <<       "avg_diff_theta_top, avg_diff_theta, avg_diff_theta_bot" << std::endl;
        // print_avg_concentrations <<       "avg_diff_pdot, avg_diff_mdot, avg_diff_m, avg_diff_theta" << std::endl;

    }

    print_avg_concentrations << time << ", ";
    print_avg_concentrations << avg_top_cPI        << ", " << avg_tot_cPI     << ", " << avg_bot_cPI        << ", ";
    print_avg_concentrations << avg_top_cPIdot     << ", " << avg_tot_cPIdot  << ", " << avg_bot_cPIdot     << ", ";
    print_avg_concentrations << avg_top_cMdot      << ", " << avg_tot_cMdot   << ", " << avg_bot_cMdot      << ", ";
    print_avg_concentrations << avg_top_cM         << ", " << avg_tot_cM      << ", " << avg_bot_cM         << ", ";
    print_avg_concentrations << avg_tot_theta      << ", " << avg_free_volume << ", " << avg_k_t            << ", "; 
    print_avg_concentrations << avg_k_p                                                                     << ", "; 
    print_avg_concentrations << avg_diff_pdot_top  << ", " << avg_diff_pdot   << ", " << avg_diff_pdot_bot  << ", ";
    print_avg_concentrations << avg_diff_mdot_top  << ", " << avg_diff_mdot   << ", " << avg_diff_mdot_bot  << ", ";
    print_avg_concentrations << avg_diff_m_top     << ", " << avg_diff_m      << ", " << avg_diff_m_bot     << ", ";
    print_avg_concentrations << avg_diff_theta_top << ", " << avg_diff_theta  << ", " << avg_diff_theta_bot << std::endl;

    // print_avg_concentrations << avg_diff_pdot << ", " << avg_diff_mdot << ", " << avg_diff_m << ", " << avg_diff_theta << std::endl;
    
    if (time == 30.0){
        std::cout << "--- COMPLETE ---" << std::endl;
        print_avg_concentrations.close();
    }

}

void Voxel::Concentrations2File(int counter,
                                std::vector<double> &c_PI_next,
                                std::vector<double> &c_PIdot_next,
                                std::vector<double> &c_Mdot_next,
                                std::vector<double> &c_M_next,
                                std::vector<double> &theta_next,
                                double time){


    // write to file
    std::string file_name = file_path + "concentrations_t" + std::to_string(counter) + ".vtk";
    print_concentrations.open(file_name);
    print_concentrations << "# vtk DataFile Version 2.0" << std::endl;
    print_concentrations << "voxel concentration ugap with particles" << std::endl;
    print_concentrations << "ASCII" << std::endl;
    print_concentrations << "DATASET RECTILINEAR_GRID" << std::endl;
    print_concentrations << "DIMENSIONS " << nodes << " " <<  nodes << " " << nodes << std::endl;

    print_concentrations << "X_COORDINATES " << nodes << " float" << std::endl;
    double dx = 0.;
    int cnt = 1;
    for (int i = 0; i < nodes; i++){
        print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        dx += h;
        cnt++;
    }
    print_concentrations << std::endl;

    print_concentrations << "Y_COORDINATES " << nodes << " float" << std::endl;
    dx = 0.;
    cnt = 1;
    for (int i = 0; i < nodes; i++){
        print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        dx += h;
        cnt++;
    }
    print_concentrations << std::endl;

    print_concentrations << "Z_COORDINATES " << nodes << " float" << std::endl;
    dx = 0.;
    cnt = 1;
    for (int i = 0; i < nodes; i++){

        print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        dx += h;
        cnt++;
    }
    print_concentrations << std::endl;

    print_concentrations << "POINT_DATA " << N_VOL_NODES << std::endl;
    print_concentrations << "SCALARS c_PI float" << std::endl;
    print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int i = 0; i < N_VOL_NODES; i++){
        print_concentrations << c_PI_next[i] << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        cnt++;
    }
    print_concentrations << std::endl;

    print_concentrations << "SCALARS c_PIdot float" << std::endl;
    print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int i = 0; i < N_VOL_NODES; i++){
        print_concentrations << c_PIdot_next[i] << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        cnt++;
    }
    print_concentrations << std::endl;

    print_concentrations << "SCALARS c_Mdot float" << std::endl;
    print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int i = 0; i < N_VOL_NODES; i++){
        print_concentrations << c_Mdot_next[i] << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        cnt++;
    }
    print_concentrations << std::endl;

    print_concentrations << "SCALARS c_M float" << std::endl;
    print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int i = 0; i < N_VOL_NODES; i++){
        print_concentrations << c_M_next[i] << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        cnt++;
    }
    print_concentrations << std::endl;

    print_concentrations << "SCALARS theta float" << std::endl;
    print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int i = 0; i < N_VOL_NODES; i++){
        print_concentrations << theta[i] << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        cnt++;
    }
    print_concentrations << std::endl;

    print_concentrations.close();
}


void Voxel::NonBoundaries2File( int counter, 
                                std::vector<double> &c_PI_next,
                                std::vector<double> &c_PIdot_next,
                                std::vector<double> &c_Mdot_next,
                                std::vector<double> &c_M_next,
                                std::vector<double> &theta_next,
                                double time, 
                                int (&coords)[3]){

    // writes only non-boundary nodes to file

    // WRITE TO FILE
    std::string file_name = file_path + "concentrations_t" + std::to_string(counter) + ".vtk";
    print_concentrations.open(file_name);
    print_concentrations << "# vtk DataFile Version 2.0" << std::endl;
    print_concentrations << "voxel concentration ugap with particles" << std::endl;
    print_concentrations << "ASCII" << std::endl;
    print_concentrations << "DATASET RECTILINEAR_GRID" << std::endl;
    print_concentrations << "DIMENSIONS " << (nodes-2) << " " <<  (nodes-2) << " " << (nodes-2) << std::endl;

    // OUTPUT COORDS
    print_concentrations << "X_COORDINATES " << (nodes-2) << " float" << std::endl;
    double dx = h;
    int cnt = 1;
    for (int i = 1; i < nodes-1; i++){
        print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        dx += h;
        cnt++;
    }
    print_concentrations << std::endl;

    print_concentrations << "Y_COORDINATES " << (nodes-2) << " float" << std::endl;
    dx = h;
    cnt = 1;
    for (int i = 1; i < nodes-1; i++){
        print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        dx += h;
        cnt++;
    }
    print_concentrations << std::endl;

    print_concentrations << "Z_COORDINATES " << (nodes-2) << " float" << std::endl;
    dx = h;
    cnt = 1;
    for (int i = 1; i < nodes-1; i++){

        print_concentrations << dx * 1000 << " ";
        if (cnt % 6 == 0){
            print_concentrations << std::endl;
        }
        dx += h;
        cnt++;
    }
    print_concentrations << std::endl;

    // OUTPUT C_PI
    print_concentrations << "POINT_DATA " << (nodes-2) * (nodes-2) * (nodes-2) << std::endl;
    print_concentrations << "SCALARS c_PI float" << std::endl;
    print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;

    // loop through all nodes
    for (int node = 0; node < N_VOL_NODES; node++){
        
        // only write internal nodes
        Node2Coord(node, current_coords); 
        if (   current_coords[0] != 0 && current_coords[0] != (nodes-1) 
            && current_coords[1] != 0 && current_coords[1] != (nodes-1) 
            && current_coords[2] != 0 && current_coords[2] != (nodes-1)){
            print_concentrations << c_PI_next[node] << " ";
            if (cnt % 6 == 0){
                print_concentrations << std::endl;
            }
            cnt++;
        }
    }
    
    print_concentrations << std::endl;

    // OUTPUT C_PIDOT
    print_concentrations << "SCALARS c_PIdot float" << std::endl;
    print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;

    // loop throug all nodes
    for (int node = 0; node < N_VOL_NODES; node++){

        // only write internal nodes
        Node2Coord(node, current_coords);
        if (   current_coords[0] != 0 && current_coords[0] != (nodes-1)
            && current_coords[1] != 0 && current_coords[1] != (nodes-1)
            && current_coords[2] != 0 && current_coords[2] != (nodes-1)){
            print_concentrations << c_PIdot_next[node] << " ";
            if (cnt % 6 == 0){
                print_concentrations << std::endl;
            }
            cnt++;
        }
    }
    print_concentrations << std::endl;

    // OUTPUT C_MDOT
    print_concentrations << "SCALARS c_Mdot float" << std::endl;
    print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;

    // loop through all nodes
    for (int node = 0; node < N_VOL_NODES; node++){
        
        // only write internal nodes
        Node2Coord(node, current_coords);
        if (   current_coords[0] != 0 && current_coords[0] != (nodes-1)
            && current_coords[1] != 0 && current_coords[1] != (nodes-1)
            && current_coords[2] != 0 && current_coords[2] != (nodes-1)){
            
            print_concentrations << c_Mdot_next[node] << " ";
            if (cnt % 6 == 0){
                print_concentrations << std::endl;
            }
        }
    }
    print_concentrations << std::endl;

    // OUTPUT C_M
    print_concentrations << "SCALARS c_M float" << std::endl;
    print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int node = 0; node < N_VOL_NODES; node++){
        
        // only write internal nodes
        Node2Coord(node, current_coords);
        if (   current_coords[0] != 0 && current_coords[0] != (nodes-1)
            && current_coords[1] != 0 && current_coords[1] != (nodes-1)
            && current_coords[2] != 0 && current_coords[2] != (nodes-1)){
            
            print_concentrations << c_M_next[node] << " ";
            if (cnt % 6 == 0){
                print_concentrations << std::endl;
            }
        }
    }
    print_concentrations << std::endl;

    // OUTPUT THETA
    print_concentrations << "SCALARS theta float" << std::endl;
    print_concentrations << "LOOKUP_TABLE default" << std::endl;
    cnt = 1;
    for (int node = 0; node < N_VOL_NODES; node++){

        // only write internal nodes
        Node2Coord(node, current_coords);
        if (   current_coords[0] != 0 && current_coords[0] != (nodes-1)
            && current_coords[1] != 0 && current_coords[1] != (nodes-1)
            && current_coords[2] != 0 && current_coords[2] != (nodes-1)){
            
            print_concentrations << theta[node] << " ";
            if (cnt % 6 == 0){
                print_concentrations << std::endl;
            }
        }
    }
    print_concentrations << std::endl;

    print_concentrations.close();
}


void Voxel::Simulate(int method, int save_voxel){

    // time discretization -> [0., dt, 2*dt, ..., T]
    int N_TIME_STEPS = t_final / dt;
    int print_iter   = N_TIME_STEPS / 600; 
    std::vector<double> total_time(N_TIME_STEPS, 0);

    std::cout << "=================================="                                     << std::endl;
    std::cout << "Simulation parameters"                                                  << std::endl;
    std::cout << "sim_id: "               << sim_id                                       << std::endl;
    std::cout << "Total time: "           << t_final                                      << std::endl;
    std::cout << "Number of time steps: " << N_TIME_STEPS                                 << std::endl;
    std::cout << "Print iteration: "      << print_iter                                   << std::endl;
    std::cout << "=================================="                                     << std::endl;
    std::cout << "Numerical parameters"                                                   << std::endl;
    std::cout << "h: "   << h                                                             << std::endl;
    std::cout << "dt: "  << dt                                                            << std::endl;
    std::cout << "Diffusion CFL: "   << Dm0 * dt / h / h                                  << std::endl;
    std::cout << "Thermal CFL: "     << dt * K_thermal_shanna/ h / h / rho_UGAP / Cp_nacl << std::endl;
    std::cout << "\n=================================="                                   << std::endl;

    Config2File(dt); 


    // initialize next time step values
    std::vector<double> c_PI_next(N_VOL_NODES,    c_PI0);
    std::vector<double> c_PIdot_next(N_VOL_NODES, 0.);
    std::vector<double> c_Mdot_next(N_VOL_NODES,  0.);
    std::vector<double> c_M_next(N_VOL_NODES,     c_M0);
    std::vector<double> theta_next(N_VOL_NODES,   theta0);

    // compute initial reaction rate constants
    ComputeRxnRateConstants();

    // write initial values to files
    if (save_voxel == 1){
        Concentrations2File(0,
                            c_PI,
                            c_PIdot,
                            c_Mdot,
                            c_M,
                            theta,
                            0);
    }
    
    AvgConcentrations2File(0,
                           c_PI,
                           c_PIdot,
                           c_Mdot,
                           c_M,
                           theta,
                           0);

    // begin time stepping
    double timer = 0.;
    int file_counter = 1;
    for (int t = 0; t < N_TIME_STEPS; t++) {

        total_time[t] = timer;

        // compute energy intensity and reaction rate constants
        ComputeRxnRateConstants();

        // solve system of equations
        SolveSystem(c_PI_next, c_PIdot_next, c_Mdot_next, c_M_next, theta_next, I0, dt, method);

        c_PI    = c_PI_next;
        c_PIdot = c_PIdot_next;
        c_Mdot  = c_Mdot_next;
        c_M     = c_M_next;
        theta   = theta_next;

        // display time
        timer += dt;
        if ((t + 1) % 100 == 0){
            std::cout << "time: " << timer << " / " << t_final << std::endl;
            std::cout << "iteration: " << t + 1 << " / " << N_TIME_STEPS + 1 << std::endl << std::endl;
        }

        // store solution results (every 100 steps) including last time step

        if (std::abs(std::floor(timer * 2) / 2 - timer) < dt || t == N_TIME_STEPS - 1){    
            if (save_voxel == 1){
                Concentrations2File(0,
                                    c_PI,
                                    c_PIdot,
                                    c_Mdot,
                                    c_M,
                                    theta,
                                    0);
            }

            AvgConcentrations2File(file_counter,
                                   c_PI_next,
                                   c_PIdot_next,
                                   c_Mdot_next,
                                   c_M_next,
                                   theta_next,
                                   timer);

            file_counter++;
        }
    }
}
