// Copyright 2023 Brian Howell
// MIT License
// Project: BayesOpt

#include "common.h"
#include "GaussianProcess.h"
#include "Voxel.h"
#include "helper_functions.h"  // Include the header file


// Generate data
obj_fns gen_data(float tfinal,
                double dt,
                int node,
                int idsim,
                bopt &bopti,
                sim &simi,
                std::string file_path,
                bool multi_thread, 
                int obj_fn) {
    // print info (if not multi-threading)

    if (!multi_thread) {
        std::cout << "============ begin simulation ============" << std::endl;
        std::cout << "id sim: " << idsim       << std::endl;
        std::cout << "temp: "   << bopti.temp  << std::endl;
        std::cout << "rp: "     << bopti.rp    << std::endl;
        std::cout << "vp: "     << bopti.vp    << std::endl;
        std::cout << "uvi: "    << bopti.uvi   << std::endl;
        std::cout << "uvt: "    << bopti.uvt   << std::endl;
        std::cout                              << std::endl;
    }

    // run simulation
    auto start = std::chrono::high_resolution_clock::now();
    Voxel VoxelSystem1(tfinal,
                       dt,
                       node,
                       idsim,
                       bopti.temp,
                       bopti.uvi,
                       bopti.uvt,
                       file_path,
                       multi_thread);

    VoxelSystem1.computeParticles(bopti.rp, bopti.vp);
    if (simi.save_density == 1) {
        VoxelSystem1.density2File();
    }

    double default_weights[4] = {0.1, 0.2, 0.2, 0.5};
    double pareto_weights[4]  = {3.56574286e-09, 2.42560512e-03, 2.80839829e-01, 7.14916061e-01};

    VoxelSystem1.simulate(simi.method, simi.save_voxel, obj_fn, pareto_weights);

    auto end = std::chrono::high_resolution_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (!multi_thread) {
        std::cout << " --- Simulation time: "
                  << t.count() * 1e-6 / 60
                  << "min ---"
                  << std::endl;

        std::cout << "testing obj: "
                  << VoxelSystem1.getObjective()
                  << std::endl;
    } else {
        std::cout << "---sim " << idsim << " complete ----" << std::endl;
    }

    obj_fns objectives;
    objectives.obj_pi    = VoxelSystem1.getObjPI();
    objectives.obj_pidot = VoxelSystem1.getObjPIDot();
    objectives.obj_mdot  = VoxelSystem1.getObjMDot();
    objectives.obj_m     = VoxelSystem1.getObjM();
    objectives.obj       = VoxelSystem1.getObjective();

    return objectives;
}

// initialize input variables
void bootstrap(sim &sim_settings,
               constraints &c,
               std::vector<bopt> &bopti,
               int num_sims,
               std::string file_path,
               bool multi_thread, 
               int obj_fn) {
    // generate random values

    if (multi_thread) {
        // Create an array of generators
        std::vector<std::mt19937> gens(num_sims);

        // Create an array of distributions
        std::vector<std::uniform_real_distribution<double>> dis(num_sims);

        // Seed each generator
        for (int id = 0; id < num_sims; ++id) {
            std::random_device rd;
            gens[id].seed(rd());
            dis[id] = std::uniform_real_distribution<double>(0.0, 1.0);
        }

        #pragma omp parallel for
        for (int id = 0; id < num_sims; ++id) {
            bopt b;
            b.temp = (c.max_temp - c.min_temp) * dis[id](gens[id])
                   +  c.min_temp;
            b.rp   = (c.max_rp   - c.min_rp)   * dis[id](gens[id])
                   +  c.min_rp;
            b.vp   = (c.max_vp   - c.min_vp)   * dis[id](gens[id])
                   +  c.min_vp;
            b.uvi  = (c.max_uvi  - c.min_uvi)  * dis[id](gens[id])
                   +  c.min_uvi;
            b.uvt  = (c.max_uvt  - c.min_uvt)  * dis[id](gens[id])
                   +  c.min_uvt;

            // peform simulation with randomly generatored values
            obj_fns objectives;
            objectives = gen_data(sim_settings.tfinal,
                             sim_settings.dt,
                             sim_settings.node,
                             id,
                             b,
                             sim_settings,
                             file_path,
                             true, 
                             obj_fn);
            
            // store objective values in bopt
            b.obj_pi    = objectives.obj_pi;
            b.obj_pidot = objectives.obj_pidot;
            b.obj_mdot  = objectives.obj_mdot;
            b.obj_m     = objectives.obj_m;
            b.obj       = objectives.obj;

            // // write individual data to file (prevent accidental loss of data)
            // write_to_file(b, sim_settings, id, file_path);
            #pragma omp critical
            {   
                int thread_id = omp_get_thread_num();
                if (!std::isnan(b.obj)) {
                    bopti.push_back(b);
                    std::cout << "Thread " << thread_id << std::endl;
                    std::cout << " | b.obj:         "  << b.obj        << std::endl
                            << " | b.obj_pi:      "  << b.obj_pi     << std::endl
                            << " | b.obj_pidot:   "  << b.obj_m      << std::endl
                            << " | b.obj_mdot:    "  << b.obj_mdot   << std::endl
                            << " | b.obj_m:       "  << b.obj_m      << std::endl
                            << " | b.temp:        "  << b.temp       << std::endl
                            << " | b.rp:          "  << b.rp         << std::endl
                            << " | b.vp:          "  << b.vp         << std::endl
                            << " | b.uvi:         "  << b.uvi        << std::endl
                            << " | b.uvt:         "  << b.uvt        << std::endl;
                    std::cout << "-------------------\n" << std::endl; 
                } else {
                    std::cout << "\nWARNING: NaN value detected" << std::endl;
                    std::cout << "thread: " << thread_id << std::endl;
                    
                }
            }
        }
    } else {
        // initialize rnd variables
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (int id = 0; id < num_sims; ++id) {
            bopt b;
            b.temp = (c.max_temp - c.min_temp) * distribution(gen)
                   +  c.min_temp;
            b.rp   = (c.max_rp   - c.min_rp)   * distribution(gen)
                   +  c.min_rp;
            b.vp   = (c.max_vp   - c.min_vp)   * distribution(gen)
                   +  c.min_vp;
            b.uvi  = (c.max_uvi  - c.min_uvi)  * distribution(gen)
                   +  c.min_uvi;
            b.uvt  = (c.max_uvt  - c.min_uvt)  * distribution(gen)
                   +  c.min_uvt;

            // peform simulation with randomly generatored values
            obj_fns objectives;
            objectives = gen_data(sim_settings.tfinal,
                             sim_settings.dt,
                             sim_settings.node,
                             id,
                             b,
                             sim_settings,
                             file_path,
                             false, 
                             obj_fn);
            
            // store objective values in bopt
            b.obj_pi    = objectives.obj_pi;
            b.obj_pidot = objectives.obj_pidot;
            b.obj_mdot  = objectives.obj_mdot;
            b.obj_m     = objectives.obj_m;
            b.obj       = objectives.obj;

            std::cout << "-------------------\n" << std::endl; 
            std::cout << " | b.obj:         "  << b.obj 
                        << " | b.obj_pi:      "  << b.obj_pi
                        << " | b.obj_pidot:   "  << b.obj_m
                        << " | b.obj_mdot:    "  << b.obj_mdot
                        << " | b.obj_m:       "  << b.obj_m
                        << " | b.temp:        "  << b.temp
                        << " | b.rp:          "  << b.rp
                        << " | b.vp:          "  << b.vp
                        << " | b.uvi:         "  << b.uvi
                        << " | b.uvt:         "  << b.uvt << std::endl;
            std::cout << "-------------------\n" << std::endl; 

            // // write individual data to file (prevent accidental loss of data)
            // write_to_file(b, sim_settings, id, file_path);

            bopti.push_back(b);
        }
    }
}

void write_to_file(bopt &b, sim &sim_set, int id, std::string file_path) {
    std::ofstream myfile;
    myfile.open(file_path + "/sim_" + std::to_string(id) + ".dat");
    myfile << "temp,rp,vp,uvi,uvt,obj,tn" << std::endl;
    myfile << b.temp << ","
           << b.rp << ","
           << b.vp << ","
           << b.uvi << ","
           << b.uvt << ","
           << b.obj << ","
           << sim_set.time_stepping
           << std::endl;
    myfile.close();
}

void store_tot_data(std::vector<bopt> &bopti,
                    sim &sim_set,
                    int num_sims,
                    std::string file_path) {
    std::cout << "--- storing data ---\n" << std::endl;
    std::ofstream myfile;
    myfile.open(file_path + "/tot_bopt.dat");
    myfile << "temp,rp,vp,uvi,uvt,obj,tn" << std::endl;
    for (int id = 0; id < num_sims; ++id) {
        myfile << bopti[id].temp << ","
               << bopti[id].rp   << ","
               << bopti[id].vp   << ","
               << bopti[id].uvi  << ","
               << bopti[id].uvt  << ","
               << bopti[id].obj  << ","
               << sim_set.time_stepping << std::endl;
    }
    myfile.close();
}

int  read_data(std::vector<bopt> &bopti,
               std::string file_path) {
    std::ifstream file(file_path + "/tot_bopt.txt");

    std::string line;
    std::getline(file, line);  // skip first line
    int id = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;

        // create new bopt struct for data point
        bopt b;

        // Parse the comma-separated values in the line
        std::getline(iss, token, ',');
        b.temp = std::stof(token);

        std::getline(iss, token, ',');
        b.rp = std::stof(token);

        std::getline(iss, token, ',');
        b.vp = std::stof(token);

        std::getline(iss, token, ',');
        b.uvi = std::stof(token);

        std::getline(iss, token, ',');
        b.uvt = std::stof(token);

        std::getline(iss, token, ',');
        b.obj = std::stof(token);

        bopti.push_back(b);

        id++;
    }

    // return number of data points
    return id;
}
