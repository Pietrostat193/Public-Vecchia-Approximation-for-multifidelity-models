# Synthetic Data Experiment

This folder contains the main simulation script used to reproduce the synthetic-data results reported in Table 2 of the paper.

## Main script

Run:

```matlab
run('SyntheticDataExperiment/Main_syntheticDataSimulation_V4.m')
```

The script bootstraps the repository path automatically, so it can be launched directly from the MATLAB editor.

## What the script computes

`Main_syntheticDataSimulation_V4.m` runs a Monte Carlo sweep over two discrepancy-noise levels:

- `sigma_d^2 = 2`
- `sigma_d^2 = 4`

For each noise level it repeats the experiment `R = 100` times and evaluates five models:

- `GP3D` = GP using only the 3D HF coordinates
- `GPL` = GP using only the LF value as input
- `GP4D` = GP using LF value + 3D coordinates
- `Classic` = dense exact multi-fidelity GP with GLS mean adjustment
- `Vecchia_v4` = Vecchia multi-fidelity GP

The script computes:

- `MAE`
- `RMSE`
- `MAPE`
- `COV90`
- `COV95`

For each metric it reports both mean and standard deviation across successful runs.

## Settings used for the paper-style summary

The script is configured with:

- simulation condition `is = 12`
- `trainFrac = 0.3`
- `rho = 0.6`
- temporal ordering (`orderingName = "time-major"`)
- neighborhood size `nn_use = 40`
- correlation-based conditioning for the Vecchia model

These settings match the Table 2 description: temporal ordering, neighborhood size 40, and correlation conditioning.

## Output files

When the script finishes, it saves outputs to:

`SyntheticDataExperiment/outputs/Main_syntheticDataSimulation_V4/`

Files produced:

- `sweep_results.mat`
  - MATLAB workspace-style output containing `Rows`, the metric arrays, run settings, and `paperTable`
- `performance_table_long.csv`
  - long-format table with columns:
    - `NoiseLevel`
    - `Metric`
    - `Stat`
    - `GP1`, `GP2`, `GP3`, `Classic`, `Vecchia_v4`
    - `nOK`, `nFail`
- `table2_summary.csv`
  - compact paper-style summary using the paper naming/order:
    - `GP3D`, `GPL`, `GP4D`, `Classic`, `Vecchia_v4`
  - each entry is formatted as `mean (std)`

## Which output corresponds to Table 2

The file `table2_summary.csv` is the closest direct reproduction of Table 2.

Use these rows from that file:

- `MAE`
- `RMSE`
- `COV95`

for the two noise levels:

- `sigma_d^2 = 2`
- `sigma_d^2 = 4`

The long-format file `performance_table_long.csv` contains the same information in numeric form, plus extra metrics (`MAPE`, `COV90`) and run counts.

## Important note on naming

Inside the MATLAB code the internal model order is:

- `GP1`
- `GP2`
- `GP3`
- `Classic`
- `Vecchia_v4`

For comparison with the paper, these map to:

- `GP1 -> GPL`
- `GP2 -> GP4D`
- `GP3 -> GP3D`

The exported file `table2_summary.csv` already uses the paper-facing names and order.

## Reviewer decomposition experiment (20 replications)

The script `reviewer_decomp_vecchia_experiment_v3_20runs.m` is a 20-replication
wrapper around `reviewer_decomp_vecchia_experiment_v3.m`. It reproduces the
reviewer-requested decomposition table that compares the exact multi-fidelity
GP with the Vecchia approximation at fixed hyperparameters, sweeping the
neighborhood size `m` and the conditioning strategy.

Run:

```matlab
run('SyntheticDataExperiment/reviewer_decomp_vecchia_experiment_v3_20runs.m')
```

or, from PowerShell at the repository root:

```powershell
& "C:\Program Files\MATLAB\R2023a\bin\matlab.exe" -batch "run('SyntheticDataExperiment/reviewer_decomp_vecchia_experiment_v3_20runs.m')"
```

### What the script does

For each of `nRep = 20` replications (seeds `12345`, `12346`, ...):

1. Simulates a new dataset via `simulate_data(seed, 0.8)`.
2. Fits the **exact** multi-fidelity GP baseline by minimizing
   `likelihood2Dsp` (quasi-Newton, 100 iterations) with
   `MeanFunction = "GP_res"`, `RhoFunction = "constant"`, RBF kernel,
   multiplicative combination, jitter `1e-6`, and `hyp_init = 0.1 * ones(18,1)`.
   It stores the exact `alpha = K^{-1}(y - m)`, `log|K|`, and `y' K^{-1} y`.
3. Evaluates the Vecchia likelihood `nlml_vecchia_fullMF` at the exact hyperparameters
   for every combination of:
   - neighborhood sizes `m ∈ {10, 20, 30, 40, 60}`;
   - conditioning strategies `{"MinMax", "Corr"}`
     (mapped to `Nearest-Neighbor` and `Corr` in the paper table).
4. Records, per setting, the relative errors of `K^{-1} y`, `log|K|`, and
   `y' K^{-1} y` against the exact baseline, plus the exact-GP test RMSE.

### Output files

Everything is written to:

`SyntheticDataExperiment/outputs/reviewer_decomp_vecchia_experiment_v3_20runs/`

Files produced:

- `reviewer_v3_20runs_raw.csv` — one row per replication × conditioning × `m`.
- `reviewer_v3_20runs_summary.csv` — mean and standard deviation of each
  metric across replications, grouped by `(Conditioning, m)`, together with
  `n_rep` (number of successful replications).
- `reviewer_v3_20runs_paper_table.csv` — compact paper-style table with
  `mean (std)` strings and the `Nearest-Neighbor` / `Corr` labels used in the
  paper.
- `reviewer_v3_20runs_results.mat` — MATLAB workspace with `Tall`, `Summary`,
  `PaperTable`, the per-replication exact RMSE, the chosen settings, and a
  `failedRep` flag.

### Notes

- The Vecchia RMSE column is currently `NaN`: `predictVecchia_CM_calibrated2`
  is not compatible with the diagnostic fields populated by
  `nlml_vecchia_fullMF`, so test-set predictions for the Vecchia model are
  skipped in this wrapper. The relevant comparison here is the decomposition
  of the likelihood error (`relErr` on `K^{-1} y`, `log|K|`, and `y' K^{-1} y`).
- The conditioning strategy `"Corr"` is handled inside
  `Utilities/nlml_vecchia_fullMF.m`: when `ModelInfo.conditioning == "Corr"`,
  the conditioning set for each point is selected by maximum absolute
  correlation (computed with the model kernel, fidelity column included)
  instead of by squared Euclidean distance.
