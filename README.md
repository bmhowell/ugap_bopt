# Bayesian Optimization for UV-dependent Mateierals

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

Simuluation and optimization of Urethane-Grafated-Acrylate Polymer (UGAP) using a Bayesian Optimizaiton framework. 

The simulation is modeled using reaction-diffusions and heat equations of the general 3D forms: 

$$ \frac{\partial [A]_i]}{\partial t} = \nabla_x \cdot (\kappa \nabla_x [A]_i_i]) + \sum_{j=1}^{N} k^{consume}_j [A]_i + \sum_{j=1}^{N} k^{generate}_j [A]_i$$

for reaction-diffusion and:

$$ \rho C \frac{\partial \theta]}{\partial t} = \nabla_x \cdot (K \nabla_x \theta) + \Delta H_{rxn} + I_{abs}$$. 

for heat.

The model is solved spatially and temporally using the Crank-Nicolson method. 

Bayesian Optimization is implemented using the Squared Exponential, Rational Quadratic, and Local-Periodic Kernels for the surrogate modeling, and an Upper Confidence Interval optimizer for the exploration/exploitation tradeoff.


## Table of Contents

- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. **Install Eigen:**

    Download and install Eigen from the [Eigen website](http://eigen.tuxfamily.org/dox/GettingStarted.html).

3. **Install BLAS/LAPACK (if not installed):**

    Depending on your operating system, you may need to install BLAS/LAPACK. For example, on Ubuntu:

    ```bash
    sudo apt-get install libblas-dev liblapack-dev
    ```

    On macOS, BLAS/LAPACK is usually provided by the Accelerate framework.

4. **Build and run the project:**

    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ./your_executable
    ```


## Contributing

If you want to contribute to this project, follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes and open a pull request
4. Ensure your code passes any existing tests
5. Update the README if needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
