# Reproducing Paper Results

This guide provides the necessary scripts to reproduce the tables presented in the paper. All scripts are designed to be run within the **MATLAB** environment.

## 📊 Summary Table of Scripts

| Result | Script Path | Description |
| :--- | :--- | :--- |
| **Table 1** | `SyntheticDataExperiment/reviewer_decomp_vecchia_experiment_v3.m` | Uncertainty propopagation analysis for subcomponents. |
| **Table 2** | `SyntheticDataExperiment/Main_syntheticDataSimulation_V4.m` | Main synthetic data simulation suite. |
| **Table 4** | `RealDataExperiment/RealDataExperiment_main2.m` | Main experiment using real-world datasets. |
| **Table B.2**| `Ordering Comparison/sim_vecchia_ordering_experiment_20runs.m` | Vecchia ordering experiment (20 runs). |

---

## 🚀 How to Run
1. **Open MATLAB** and set the repository root as your current working directory.
2. Ensure the `Utilities/` folder is added to your path:
   ```matlab
   addpath(genpath('.'));







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
