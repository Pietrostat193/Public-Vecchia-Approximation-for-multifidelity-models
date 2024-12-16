# Multi-Fidelity Gaussian Process Model with Vecchia Approximation

This repository introduces a novel **multi-fidelity Gaussian process (GP) model** designed for spatio-temporal data fusion. By leveraging the **Vecchia approximation**, our approach significantly reduces computational complexity while maintaining flexibility and scalability.

## Key Features
- **Independent Vecchia Approximation**: Our framework separates the low-fidelity GP from the discrepancy process, enabling the Vecchia approximation to be applied independently to each component.
- **Non-Stationary Integration**: The model supports non-stationary integration of different fidelity levels, a feature that is notoriously challenging in non-approximated models due to the dense matrices involved in standard computations.
- **Scalable and Efficient**: By addressing the computational challenges of dense covariance matrices, our model offers a practical solution for large spatio-temporal datasets.

## Current Status
The project is currently under development. Contributions, feedback, and discussions are welcome!
