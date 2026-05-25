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







# Multi-Fidelity Gaussian Process Model with Vecchia Approximation

This repository introduces a novel **multi-fidelity Gaussian process (GP) model** designed for spatio-temporal data fusion. By leveraging the **Vecchia approximation**, our approach significantly reduces computational complexity while maintaining flexibility and scalability.

## Key Features
- **Independent Vecchia Approximation**: Our framework separates the low-fidelity GP from the discrepancy process, enabling the Vecchia approximation to be applied independently to each component.
- **Non-Stationary Integration**: The model supports non-stationary integration of different fidelity levels, a feature that is notoriously challenging in non-approximated models due to the dense matrices involved in standard computations.
- **Scalable and Efficient**: By addressing the computational challenges of dense covariance matrices, our model offers a practical solution for large spatio-temporal datasets.

##  Repository structure

The repository contains illustrative examples on how to run the model and
comparison with the non-approximated version, plus the full experimental
pipelines used in the paper.

- **Illustrative_example/**: fully working examples in 1-D (Time only) and 3-D
  (Space-Time) of the model on simulated data, with a comparison against the
  non-approximated version. The data-simulation routine can be tuned to
  generate larger datasets.
- **SyntheticDataExperiment/**: synthetic-data experiments used to produce
  Tables 1 and 2 of the paper. See its [README](SyntheticDataExperiment/README.md).
- **RealDataExperiment/**: real-data LOSO experiments over South Lombardy
  weather stations used to produce Table 4 (MFGP rows via
  `RealDataExperiment_main2.m`, GP-3D row via `GP_realDataExperiment.m`).
  See its [README](RealDataExperiment/README.md).
- **Ordering Comparison/**: Vecchia ordering ablation used to produce
  Table B.2.
- **Datasets/**: pre-processed `.mat` files used by the experiments
  (`South_Lombardy_sorted_data.mat` for the real-data experiments) and the
  synthetic-dataset generation scripts.
- **Utilities/**: shared helpers (likelihoods, kernels, predictors,
  data-simulation routines) used by all experiments. Includes the vendored
  WMFGP toolbox under `Utilities/WMFGP/` for the warped MFGP configurations.
- **Computation times/**: legacy computation-time figure and the script that
  produced it, based on older functions no longer used by the main pipeline.
  Its purpose is illustrative only — to show that the Vecchia-approximated
  algorithm scales better than the non-approximated version as the dataset
  grows. It is not intended as a reference benchmark, since absolute timings
  depend heavily on the specific implementation and hardware.
- **paper/**: placeholder folder for the LaTeX source of the paper
  (currently maintained on Overleaf). See [paper/README.md](paper/README.md).

## Current Status
Under revision!
