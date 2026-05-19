# Real Data Experiment (South Lombardy wind speed)

This folder contains the script used to reproduce the aggregated MFGP
performance table reported in the paper:

| Model ID    | Count | MAE     | RMSE    | Corr    | PICP_95 | NLML   |
|-------------|-------|---------|---------|---------|---------|--------|
| GP-3D       | 18    | 0.3974  | 0.5245  | 0.77    | 93.84%  | —      |
| MFGP_gc     | 18    | 0.1906  | 0.2429  | 0.9351  | 89.06   | 2790.6 |
| MFGP_ac     | 18    | 0.1935  | 0.2477  | 0.9312  | 58.28   | 2738.5 |
| MFGP_gWc    | 18    | 0.1949  | 0.2487  | 0.9307  | 91.39   | 3323.3 |
| MFGP_aWc    | 18    | 0.2005  | 0.2552  | 0.9304  | 76.94   | 3274.5 |
| MFGP_gGP    | 18    | 0.3203  | 0.4007  | 0.8729  | 83.78   | 2729.3 |
| MFGP_aGP    | 18    | 0.4446  | 0.5598  | 0.8854  | 83.17   | 2786.3 |

The table aggregates leave-one-station-out predictions across the 18
validation stations of the South Lombardy dataset.

## Main script

```matlab
run('RealDataExperiment/RealDataExperiment_main2.m')
```

The script bootstraps the repository path automatically (adds
`Utilities/` and everything under the repo root via `addpath(genpath(...))`),
so it can be launched directly from the MATLAB editor.

## What the script computes

For every station `s` in `unique(data.IDStation)` the script performs a
leave-one-station-out (LOSO) experiment:

1. Splits the data into training (all other stations) and test
   (held-out station), capping the number of time points per station
   to `capN = 100` via `cap_times_per_station`.
2. Builds the multi-fidelity inputs:
   - LF input/output: `(Time, Lat_LF, Lon_LF)` → `Wind_speed`
   - HF input/output: same coordinates → `ws`
3. Precomputes the Vecchia neighborhood indices for the LF and HF sets
   once per station (`vecchia_approx_space_time_corr_fast1` +
   `extract_vecchia_indices`, with `nn_size = 25`, RBF kernel,
   `Corr` conditioning).
4. Loops over six MFGP configurations and, for each one:
   - sets `ModelInfo.GLSType` and `ModelInfo.RhoFunction`;
   - applies the warping step (`KCDF_Estim` / `Kernel_invNS`) when the
     configuration name contains `_W_`;
   - fits the hyperparameters with `fminunc` on
     `likelihoodVecchia_nonstat_GLS_v3` (quasi-Newton, 50 iterations);
   - predicts the held-out station with either
     `predict_calibratedCM3_AdaptiveGLS_v4` (adaptive GLS) or
     `predict_calibratedCM3_fixed` (fixed GLS);
   - undoes the warping when applicable and computes `RMSE`, `MAE`,
     `Corr`, `PICP_95`, and stores the final negative log marginal
     likelihood as `NLML`.

### Mapping between script config names and paper Model IDs

| Script `conf_tag` | `GLSType` | `RhoFunction`         | Warping | Paper Model ID |
|-------------------|-----------|-----------------------|---------|----------------|
| `Const_RhoC`      | fixed     | `constant`            | no      | `MFGP_gc`      |
| `Adap_RhoC`       | adaptive  | `constant`            | no      | `MFGP_ac`      |
| `Const_W_RhoC`    | fixed     | `constant`            | yes     | `MFGP_gWc`     |
| `Adap_W_RhoC`     | adaptive  | `constant`            | yes     | `MFGP_aWc`     |
| `Const_RhoA`      | fixed     | `GP_scaled_empirical` | no      | `MFGP_gGP`     |
| `Adap_RhoA`       | adaptive  | `GP_scaled_empirical` | no      | `MFGP_aGP`     |

The `GP-3D` row in the paper table is **not** produced by this script;
it is a single-fidelity GP baseline that uses only the HF 3D
coordinates `(Time, Lat_HF, Lon_HF)` and is computed by a separate
script: [Utilities/GP_realDataExperiment.m](../Utilities/GP_realDataExperiment.m).

That script runs the same LOSO loop over the 18 validation stations,
fits a single-fidelity sparse GP with `fitrgp` (subset of regressors,
ARD squared-exponential kernel, linear basis), and produces the `RMSE`,
`MAE`, `Corr`, and `PICP_95` numbers reported in the `GP-3D` row.
It does not return an `NLML` (hence the `—` entry in the table above).

> Note: at the time of writing, `GP_realDataExperiment.m` still uses an
> absolute Windows path to load the dataset. If you want to run it on
> another machine, edit its `dataFile` line to
> `fullfile(repoRoot, 'Datasets', 'South_Lombardy_sorted_data.mat')`
> or load `Datasets/South_Lombardy_sorted_data.mat` manually before
> running it.

## Output files

The script writes everything under:

`RealDataExperiment/outputs/RealDataExperiment_main2/`

Files produced:

- `Experiment_Full_Results_744.mat`
  - `ResultsHistory744(station).(config)` with `y_true`, `y_pred`,
    `CI_up`, `CI_low`, `hyp`;
  - `all_metrics744` table with one row per `(Station, Config)` and the
    columns `RMSE`, `MAE`, `Corr`, `PICP_95`, `NLML`.
- `all_metrics744.csv` — CSV export of `all_metrics744`.
- `Experiment_Log.txt` — full diary of the run.

To reproduce the paper-style aggregated table, average the columns of
`all_metrics744` over `Station` for each `Config`. For example, in
MATLAB:

```matlab
T = readtable(fullfile('RealDataExperiment','outputs', ...
    'RealDataExperiment_main2','all_metrics744.csv'));
aggT = groupsummary(T, 'Config', 'mean', ...
    {'MAE','RMSE','Corr','PICP_95','NLML'});
disp(aggT)
```

## Paths and external dependencies

The script is fully relative to the repository: the dataset is loaded
from `Datasets/South_Lombardy_sorted_data.mat` via

```matlab
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);
dataFile  = fullfile(repoRoot, 'Datasets', 'South_Lombardy_sorted_data.mat');
addpath(genpath(repoRoot));
```

so no absolute paths need to be edited on a new machine.

### Warping toolbox (vendored)

The four configurations whose tag does **not** contain `_W_`
(`Const_RhoC`, `Adap_RhoC`, `Const_RhoA`, `Adap_RhoA`) need only the
files already in this repository.

The two warped configurations (`Const_W_RhoC`, `Adap_W_RhoC` →
`MFGP_gWc`, `MFGP_aWc`) rely on the **WMFGP** toolbox functions
`KCDF_Estim`, `Gen_Lookup`, `Kernel_invNS` (and their internal
dependencies). To keep the repository self-contained, the toolbox has
been **vendored** under [Utilities/WMFGP/](../Utilities/WMFGP/) and is
picked up automatically by the `addpath(genpath(repoRoot))` call at
the top of the script, so no extra configuration is required.

If those functions are not on the path at startup, the script prints a
warning and skips the two warped configurations; all other
configurations run normally.
