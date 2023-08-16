cmake_minimum_required(VERSION 3.20)
project(opt_ugap)

set(CMAKE_CXX_STANDARD 14)

add_executable(opt_ugap
               main.cpp
               common.h
               helper_functions.cpp
               helper_functions.h
               GaussianProcess.cpp
               GaussianProcess.h
               Voxel.cpp
               Voxel.h
        )


# add EIGEN library
# https://stackoverflow.com/questions/49125383/how-do-i-configure-the-way-a-project-is-built-in-clion
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${EIGEN_LIBRARIES})