# Multi-Fidelity Gaussian Process Model with Vecchia Approximation

This repository introduces a novel **multi-fidelity Gaussian process (GP) model** designed for spatio-temporal data fusion. By leveraging the **Vecchia approximation**, our approach significantly reduces computational complexity while maintaining flexibility and scalability.

## Key Features
- **Independent Vecchia Approximation**: Our framework separates the low-fidelity GP from the discrepancy process, enabling the Vecchia approximation to be applied independently to each component.
- **Non-Stationary Integration**: The model supports non-stationary integration of different fidelity levels, a feature that is notoriously challenging in non-approximated models due to the dense matrices involved in standard computations.
- **Scalable and Efficient**: By addressing the computational challenges of dense covariance matrices, our model offers a practical solution for large spatio-temporal datasets.

##  Repository structure

The repository contains illustrative examples on how to run the model and comparison with the non-approximated version.
- **Computation time example** : Contains a png of an example of computation time and the script used for producing the figure. This is based on an old functions not currently used.
- **Illustrative example** : Contains two fully functioning examples in 1-D (Time only) and 3-D (Space-Time) of the model on simulated data, with a comparison with non-approximated version of the model. Note that the function for data simulation can be internally adjusted for generating large dataset.

## Current Status
The project is currently under development. Contributions, feedback, and discussions are welcome!
