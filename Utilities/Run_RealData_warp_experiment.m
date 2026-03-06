% ============================================================
% Runner for:
% compare_rho_LOSO_realdata_warped_constant(data, opts)
% ============================================================

clear; clc;

% ---------- 1) Load data ----------
dataPath = ...
 'C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy';

dataFile = fullfile(dataPath, 'South_Lombardy_sorted_data.mat');
S = load(dataFile);

% --- Inspect what was loaded
disp(fieldnames(S));

% --- Assume the table is called "data"
% If not, change this line to the correct variable name
data = S.sorted_data;

% ---------- 2) Sanity check ----------
requiredVars = ["Wind_speed","ws","Lat_LF","Lon_LF", ...
                "Lat_HF","Lon_HF","Time","IDStation"];
assert(all(ismember(requiredVars, string(data.Properties.VariableNames))), ...
    "ERROR: Data table missing required columns.");

fprintf('Loaded data with %d rows and %d stations.\n', ...
        height(data), numel(unique(data.IDStation)));

% ---------- 3) Options ----------
opts = struct();

% --- LOSO for ALL stations
opts.holdout_station = [];
opts.do_all_stations = true;

% --- Multi-start
opts.n_starts = 1;          % <<< as requested
opts.max_iter = 200;
opts.max_fun  = 2000;

% --- Vecchia / model
opts.nn_size      = 20;
opts.cand_mult    = 10;
opts.conditioning = "Corr";
opts.kernel       = "RBF";
opts.cov_type     = "RBF";
opts.combination  = "multiplicative";

% --- Forced inside function anyway
opts.MeanFunction = "zero";
opts.RhoFunction  = "constant";

% --- Reproducibility / size control
opts.seed = 42;
opts.time_cap_per_station = 100;

% --- Warping
opts.warping_kernel  = 'Tria';
opts.warping_addpath = ...
 'C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\5_Warping\WMFGP-main (1)\WMFGP-main';  % <-- adjust if needed

% --- Warm-start OFF (safe default)
opts.prev_results_const = [];
opts.use_prev_hyp = false;
opts.warm_jitter_scale = 0.05;
opts.use_prev_only_first_start = true;

% ---------- 4) Run ----------
results_warp = compare_rho_LOSO_realdata_warped_constant(data, opts);

% ---------- 5) Quick summary ----------
disp(results_warp(:, ...
    {'holdout_station','NLML','RMSE','MAE','nTrainL','nTrainH','nTestH'}));

fprintf('\nOverall mean RMSE: %.6f\n', mean(results_warp.RMSE));
fprintf('Overall mean MAE : %.6f\n', mean(results_warp.MAE));



% ========================================================================
% SINGLE SCRIPT: side-by-side table of RMSE / MAE / CORR for 3 methods
% Requires in workspace:
%   results       (table)  -> contains rho constant + rho GP_scaled_emp
%   results_warp  (table)  -> contains rho constant (warped warmstart)
% ========================================================================

clearvars -except results results_warp
clc

% --------- basic checks
assert(exist('results','var')==1 && istable(results), 'Need table "results" in workspace.');
assert(exist('results_warp','var')==1 && istable(results_warp), 'Need table "results_warp" in workspace.');

% --------- helper: safe correlation for vectors
safe_corr = @(a,b) local_safe_corr(a,b);

% --------- 1) Add CORR column to results (non-warped)
if ~ismember("CORR", string(results.Properties.VariableNames))
    results.CORR = nan(height(results),1);
end

for i = 1:height(results)
    y    = results.y_true{i};
    yhat = results.mu_pred{i};
    results.CORR(i) = safe_corr(y, yhat);
end

% --------- 2) Add CORR column to results_warp
if ~ismember("CORR", string(results_warp.Properties.VariableNames))
    results_warp.CORR = nan(height(results_warp),1);
end

for i = 1:height(results_warp)
    y    = results_warp.y_true{i};
    yhat = results_warp.mu_pred{i};
    results_warp.CORR(i) = safe_corr(y, yhat);
end

% --------- 3) Extract the 3 method tables
% Method A: rho constant (non-warped)
A = results(results.RhoFunction == "constant", ...
    {'holdout_station','RMSE','MAE','CORR'});
A.Properties.VariableNames = {'holdout_station','RMSE_const','MAE_const','CORR_const'};

% Method B: rho GP_scaled_empirical (non-warped)
B = results(results.RhoFunction == "GP_scaled_empirical", ...
    {'holdout_station','RMSE','MAE','CORR'});
B.Properties.VariableNames = {'holdout_station','RMSE_GP','MAE_GP','CORR_GP'};

% Method C: rho constant (warped warmstart)
C = results_warp(:, {'holdout_station','RMSE','MAE','CORR'});
C.Properties.VariableNames = {'holdout_station','RMSE_warp','MAE_warp','CORR_warp'};

% --------- 4) Outer-join on holdout_station (robust if a method is missing a station)
T = outerjoin(A, B, 'Keys','holdout_station', 'MergeKeys',true);
T = outerjoin(T, C, 'Keys','holdout_station', 'MergeKeys',true);

% Sort by station id
T = sortrows(T, 'holdout_station');

% --------- 5) Add mean row (optional)
meanRow = varfun(@mean, T, ...
    'InputVariables', setdiff(T.Properties.VariableNames,"holdout_station"), ...
    'OutputFormat','table');

meanRow.holdout_station = -1; % marker for "MEAN"
meanRow = movevars(meanRow,'holdout_station','Before',1);

T_with_mean = [T; meanRow];

% --------- 6) Display + optionally export
disp('=== Side-by-side metrics (last row is mean; holdout_station=-1) ===');
disp(T_with_mean);

% Optional: write to CSV
% writetable(T_with_mean, 'metrics_3methods_side_by_side.csv');

% ========================================================================
% Local function: safe correlation
% ========================================================================
function r = local_safe_corr(y, yhat)
    r = NaN;
    try
        y = double(y(:));
        yhat = double(yhat(:));
        ok = isfinite(y) & isfinite(yhat);
        y = y(ok); yhat = yhat(ok);
        if numel(y) < 2
            return;
        end
        if std(y) == 0 || std(yhat) == 0
            return;
        end
        C = corrcoef(y, yhat);
        r = C(1,2);
    catch
        r = NaN;
    end
end
