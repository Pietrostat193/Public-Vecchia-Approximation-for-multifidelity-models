# Multi-Fidelity Gaussian Process Model with Vecchia Approximation

This repository accompanies the paper:

 **A new framework for non-stationary
spatio-temporal data fusion of multi-fidelity
models**

We propose a scalable multi-fidelity Gaussian process (GP) model for spatio-temporal data fusion. The method leverages a Vecchia approximation to reduce computational complexity while maintaining flexibility in modeling non-stationary relationships across fidelity levels.

The key contribution is a decomposition of the multi-fidelity structure into independent components, allowing separate Vecchia approximations for the low-fidelity process and the discrepancy process. This enables scalable inference on large spatio-temporal datasets that would otherwise be computationally prohibitive under standard GP formulations.

---

## Key Features

- Independent Vecchia approximation applied separately to fidelity components  
- Support for non-stationary multi-fidelity integration  
- Scalable inference for large spatio-temporal datasets  
- Fully reproducible MATLAB pipeline for all reported results  

---

## Repository Structure

The repository is organized as follows:

- **Illustrative_example/**: Simple 1D (time) and 3D (spatio-temporal) examples demonstrating the model and comparison against the exact non-approximated GP formulation.

- **SyntheticDataExperiment/**: Synthetic experiments used to generate Tables 1 and 2 of the paper. See its internal README for details.

- **RealDataExperiment/**: Real-data LOSO experiments on South Lombardy weather stations used for Table 4:
  - `RealDataExperiment_main2.m`: MFGP results  
  - `GP_realDataExperiment.m`: GP-3D baseline  

- **Ordering Comparison/**: Vecchia ordering ablation study used for Table B.2.

- **Datasets/**: Pre-processed MATLAB datasets used across experiments, including the South Lombardy dataset.

- **Utilities/**: Shared helper functions (kernels, likelihoods, simulators, predictors). Includes the vendored WMFGP toolbox under `Utilities/WMFGP/`.

- **Computation times/**: Legacy scripts illustrating scalability differences between Vecchia and non-Vecchia methods. Included for completeness but not part of the main pipeline.

- **paper/**: Placeholder for LaTeX source (maintained externally on Overleaf).

---


# Reproducing Paper Results

This guide provides the necessary scripts to reproduce the tables presented in the paper. All scripts are designed to be run within the **MATLAB** environment.

## 🧰 Requirements

- **MATLAB R2023a** (developed and tested). Minimum supported: **R2020a**
  (required for `groupsummary`, used by the aggregation block in
  `RealDataExperiment_main2.m`).
- **Optimization Toolbox** — used by `fminunc` / `optimoptions('fminunc',…)`
  in all MFGP hyper-parameter fits
  (`RealDataExperiment_main2.m`,
  `SyntheticDataExperiment/reviewer_decomp_vecchia_experiment_v3.m`,
  `SyntheticDataExperiment/run_demo.m`,
  `Ordering Comparison/predict_oos_best4_only.m`,
  `Computation times/ComputationTime.m`).
- **Statistics and Machine Learning Toolbox** — used by `fitrgp` for the
  GP-3D baseline (`RealDataExperiment/GP_realDataExperiment.m`).

No other toolboxes are required (no Parallel Computing, no GPU, no Symbolic,
no Curve Fitting, no third-party packages — the WMFGP code is vendored under
`Utilities/WMFGP/`).

Quick check from the MATLAB Command Window:
```matlab
license('test','Optimization_Toolbox')      % must return 1
license('test','Statistics_Toolbox')        % must return 1
```

## 📊 Summary Table of Scripts

Each paper table has a dedicated thin-wrapper script at the repository root
(`repro_table*.m`). Running the wrapper sets up the MATLAB path and dispatches
to the underlying implementation. The 1:1 mapping is:

| Result | Entry-point (run this) | Underlying script |
| :--- | :--- | :--- |
| **Table 1** | `repro_table1.m` | `SyntheticDataExperiment/reviewer_decomp_vecchia_experiment_v3_20runs.m` (20-replication wrapper; single-run variant: `reviewer_decomp_vecchia_experiment_v3.m`) |
| **Table 2** | `repro_table2.m` | `SyntheticDataExperiment/Main_syntheticDataSimulation_V4.m` |
| **Table 4 (MFGP rows)** | `repro_table4_MFGP.m` | `RealDataExperiment/RealDataExperiment_main2.m` |
| **Table 4 (GP-3D row)** | `repro_table4_GP3D.m` | `RealDataExperiment/GP_realDataExperiment.m` |
| **Table B.2** | `repro_tableB2.m` | `Ordering Comparison/sim_vecchia_ordering_experiment_20runs.m` |

> All real-data scripts use `capN = 100` (number of time points per station
> retained in train/test). This is the canonical setting for this repository
> and produces the Table 4 numbers reported in
> [RealDataExperiment/README.md](RealDataExperiment/README.md).

Per-folder details (paper-table → config mapping, output files, run commands)
are in the local READMEs:

- [SyntheticDataExperiment/README.md](SyntheticDataExperiment/README.md)
- [RealDataExperiment/README.md](RealDataExperiment/README.md)

---

## 🚀 How to Run
1. **Open the repository in MATLAB** at the repo root.
2. Run the corresponding `repro_table*.m` wrapper for the result you want to
   reproduce, e.g. `>> repro_table2` in the Command Window. The wrapper sets
   up the MATLAB path and dispatches to the underlying implementation script.

The main reproduction scripts now add the repository folders to the MATLAB path
automatically (`addpath(genpath(repoRoot))` is done inside each entry-point
script), so users do not need to manually call `addpath(genpath('.'))` first.
All paths to datasets and outputs are resolved relative to the repository
root, so the scripts run unchanged on any machine after cloning.

Outputs (`.mat`, `.csv`, diary logs) are written under
`<Experiment>/outputs/<script-name>/` rather than next to the source scripts.

### Vendored dependency: WMFGP toolbox

The warped MFGP configurations (`Const_W_RhoC`, `Adap_W_RhoC` in
`RealDataExperiment_main2.m`) rely on the WMFGP toolbox (KCDF-based
warping). A copy is vendored in `Utilities/WMFGP/` and is picked up
automatically by `addpath(genpath(repoRoot))`, so no extra setup is
needed.






