# Main_sinthetic_simulation3.m — Synthetic Simulation Sweep (2 Noise Levels, 5 Models)

This repository contains a MATLAB script to run a synthetic spatio-temporal multi-fidelity simulation and benchmark **five Gaussian-process-based models** across **two observation-noise settings**. The script performs **R Monte Carlo runs**, computes **MAE / RMSE / MAPE** for each model, and reports **Mean** and **Std Dev** across successful runs.

---

## What this script does

**Script to run:** `Main_sinthetic_simulation3.m`

For a fixed simulation condition (`is = 12`) and a fixed correlation parameter (`rho = 0.6`), the script:

1. Sweeps over two high-fidelity noise values:
   - `sigma_d2 = 2`
   - `sigma_d2 = 4`

2. Repeats the simulation **R = 100** times for each noise value.

3. For each run:
   - generates synthetic multi-fidelity training/test data
   - trains / fits:
     - **GP1**
     - **GP2**
     - **GP3**
     - **Classic (Exact MFGP)**
     - **Vecchia60**
   - predicts on the HF test set
   - computes error metrics:
     - **MAE** (Mean Absolute Error)
     - **RMSE** (Root Mean Squared Error)
     - **MAPE** (Mean Absolute Percentage Error)

4. Aggregates results across successful runs and prints a table containing:
   - Mean + Std for each metric and each model
   - number of successful runs (`nOK`)
   - number of failed runs (`nFail`)

---

## Requirements

- **MATLAB** (tested with versions that support `table` and `optimoptions/fminunc`)
- Optimization Toolbox (for `fminunc`)

---

## Required functions/files (must be on the MATLAB path)

The script calls the following project functions. Make sure they exist in your repository and are discoverable via `addpath(...)` or by placing them in the same folder (or subfolders) on the MATLAB path:

### Simulation / configuration
- `make_sim_conditions()`
- `simulate_data_dynamic(seed, trainFrac, cfg)`

### Classic Exact MFGP (global `ModelInfo` workflow)
- `likelihood2Dsp(hyp)`
- `predict2Dsp(X_test)`

### GP1 / GP2 / GP3 predictors
- `train_and_predict_gpr(ModelInfo2)`

### Vecchia60 likelihood + predictor
- `likelihoodVecchia_nonstat_GLS(hyp)`
- `predictVecchia_CM_calibrated2(X_test)`

> **Important:** The Classic and Vecchia pipelines rely on a global variable named `ModelInfo`.
> The script manages this (via `clear global ModelInfo; global ModelInfo;`) but the called likelihood/predict functions must be implemented consistently with this convention.

---

## Expected structure of `simulate_data_dynamic` output

The script assumes `simulate_data_dynamic(...)` returns a struct `out` with (at minimum):

### High-fidelity training
- `out.HF_train.t`
- `out.HF_train.s1`
- `out.HF_train.s2`
- `out.HF_train.fH`

### High-fidelity testing
- `out.HF_test.t`
- `out.HF_test.s1`
- `out.HF_test.s2`
- `out.HF_test.fH`

### Low-fidelity data
- `out.LF.t`
- `out.LF.s1`
- `out.LF.s2`
- `out.LF.fL`

### Alignment index for GP1/2/3 predictions
- `out.test_row_idx`

`out.test_row_idx` is **required** and must correctly align the prediction vectors returned by `train_and_predict_gpr(...)` with the HF test set (so that `Y1(out.test_row_idx)` corresponds to `out.HF_test`).

---

## How to run

### 1) Open MATLAB and set your working directory
Set the working directory to the folder containing `Main_sinthetic_simulation3.m`.

### 2) Ensure all dependencies are on the path
If your functions are in subfolders, do something like:
```matlab
addpath(genpath(pwd));
