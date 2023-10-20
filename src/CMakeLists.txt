cmake_minimum_required(VERSION 3.20)
project(ugap_opt)


set(CMAKE_CXX_STANDARD 14)


add_executable(${PROJECT_NAME}
              main.cpp
              common.h
              helper_functions.cpp
              helper_functions.h
              GaussianProcess.cpp
              GaussianProcess.h
              BayesianOpt.cpp
              BayesianOpt.h
              Voxel.cpp
              Voxel.h
       )


# set run option to release mode
set(CMAKE_BUILD_TYPE Release)


# add EIGEN library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


# Find OpenMP library
find_package(OpenMP REQUIRED)

# set(BLAS_LIBRARIES "/usr/lib/x86_64-linux-gnu/libblas.so")
# set(LAPACK_LIBRARIES "/usr/lib/x86_64-linux-gnu/lapack/liblapack.so")
# set(Eigen3_DIR "/usr/share/eigen3/cmake")


find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)


target_link_libraries(${PROJECT_NAME} 
                      Eigen3::Eigen ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
                      OpenMP::OpenMP_CXX)

