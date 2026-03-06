%% EXPERIMENT (Single station + single model) + Ordering Strategy Comparison
% - Uses ONLY the first station in the file
% - Uses ONLY the first configuration (first row of configs)
% - Compares multiple ORDERING strategies (everything else identical)
% - Prints results (NO saving, NO overwriting)
% - Optionally warmstarts from an existing ResultsHistory in workspace (read-only)

%clear; clc;
global ModelInfo

%% ---------------- USER SETTINGS ----------------
capN     = 100;
dataFile = "C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy\South_Lombardy_sorted_data.mat";

% Vecchia / model settings
nn_size   = 25;
kernelStr = "RBF";
jitterVal = 1e-5;

% Optimizer settings
MaxIter   = 50;
UseWarmStartFromResultsHistory = true; % only reads if ResultsHistory exists in workspace

% Ordering strategies to compare (train ordering only; test stays unchanged)
ORDERINGS = { ...
    "Station-major", ...          % sort by (Lat,Lon,Time)  [paper default]
    "Time-major", ...             % sort by (Time,Lat,Lon)
    "Time-causal+RandSpace", ...  % time blocks, random stations within each time
    "Random" ...                  % fully random
};

% seed for random orderings (reproducible)
seedRandSpace = 111;
seedRandom    = 222;

%% ---------------- PATHS ----------------
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\5_Warping\WMFGP-main (1)\WMFGP-main");
addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\New\3D-Example\3D-Example\Utilities");

%% ---------------- LOAD DATA ----------------
S    = load(dataFile);
data = S.sorted_data;

hold_id_list = unique(data.IDStation);

if isempty(hold_id_list)
    error("No stations found in data.");
end

% ONLY FIRST STATION
hold_id = hold_id_list(1);
s_name  = sprintf('Station_%d', hold_id);

fprintf('\n==========================================================\n');
fprintf(' ORDERING COMPARISON | ONLY FIRST STATION: %d\n', hold_id);
fprintf('==========================================================\n');

%% ---------------- SPLIT TRAIN/TEST (hold-out station) ----------------
is_hold = (data.IDStation == hold_id);
if ~any(is_hold)
    error("Station %d not found.", hold_id);
end

train_tbl = cap_times_per_station(data(~is_hold,:), capN, 42);
test_tbl  = cap_times_per_station(data(is_hold,:),  capN, 42);

X_L_raw = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
y_L_raw = double(train_tbl.Wind_speed);

X_H_raw = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
y_H_raw = double(train_tbl.ws);

Xstar  = [double(test_tbl.Time), double(test_tbl.Lat_LF), double(test_tbl.Lon_LF)];
y_true = double(test_tbl.ws);

% Sort test by time for nicer plots
[~, pT] = sort(Xstar(:,1), 'ascend');
Xstar_s  = Xstar(pT,:);
y_true_s = y_true(pT);

%% ---------------- SINGLE MODEL (FIRST CONFIG ONLY) ----------------
% (matches your configs{1,:})
conf_tag = "Const_RhoC";
GLSType  = "fixed";
RhoFunc  = "constant";
use_warp = false;

fprintf('\nUsing ONLY first model config: %s | GLSType=%s | Rho=%s | warp=%d\n', ...
    conf_tag, GLSType, RhoFunc, use_warp);

%% ---------------- Base ModelInfo (fixed across orderings) ----------------
ModelInfo = struct();
ModelInfo.nn_size      = nn_size;
ModelInfo.kernel       = kernelStr;
ModelInfo.jitter       = jitterVal;
ModelInfo.MeanFunction = "zero";
ModelInfo.conditioning = "Corr";
ModelInfo.cov_type     = kernelStr;

ModelInfo.GLSType      = GLSType;
ModelInfo.RhoFunction  = RhoFunc;
ModelInfo.cand_mult    = 10;
ModelInfo.show_path_diag = false;

%% ---------------- Hyperparameters dimension & start ----------------
n_params = 11 + 3*(ModelInfo.RhoFunction == "GP_scaled_empirical");
hyp0 = rand(n_params,1);

% Optional warmstart (read-only; does NOT overwrite anything)
if UseWarmStartFromResultsHistory && evalin('base','exist(''ResultsHistory'',''var'')==1')
    try
        oldHyp = evalin('base', sprintf("ResultsHistory.%s.%s.hyp", s_name, conf_tag));
        oldHyp = oldHyp(:);
        if numel(oldHyp) == n_params && all(isfinite(oldHyp))
            hyp0 = oldHyp;
            fprintf("Warmstart from existing ResultsHistory in base workspace.\n");
        else
            fprintf("Warmstart found but wrong size / non-finite -> using random hyp0.\n");
        end
    catch
        fprintf("No warmstart found for %s.%s -> using random hyp0.\n", s_name, conf_tag);
    end
end

opts_opt = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','iter', ...
    'MaxIterations', MaxIter);

%% ---------------- Run ordering comparison ----------------
results = table('Size',[0 9], ...
    'VariableTypes',{'string','double','double','double','double','double','double','double','double'}, ...
    'VariableNames',{'Ordering','NLML','RMSE','MAE','Corr','PICP_95','MeanPred','StdPred','TimeSeconds'});

predStore = struct(); % for plotting only (not saved to disk)

for oi = 1:numel(ORDERINGS)
    ordName = string(ORDERINGS{oi});

    % ----- build ordering permutation on TRAIN ONLY -----
    p = build_ordering_perm(X_L_raw, ordName, seedRandSpace, seedRandom);

    X_L = X_L_raw(p,:); y_L = y_L_raw(p);
    X_H = X_H_raw(p,:); y_H = y_H_raw(p);

    % Put into ModelInfo
    ModelInfo.X_L = X_L; ModelInfo.y_L = y_L;
    ModelInfo.X_H = X_H; ModelInfo.y_H = y_H;

    % (No warping in this experiment)
    if use_warp
        error("This comparison script assumes use_warp=false (first model).");
    end

    % ----- Precompute Vecchia indices for this ordering (ORDERING-DEPENDENT!) -----
    fprintf('\n[%d/%d] Ordering: %s | Precomputing Vecchia indices...\n', oi, numel(ORDERINGS), ordName);

    resL = vecchia_approx_space_time_corr_fast1(X_L,[1,1],[1,1],ModelInfo.nn_size,1e-6,kernelStr,10,1,1,[]);
    resH = vecchia_approx_space_time_corr_fast1(X_H,[1,1],[1,1],ModelInfo.nn_size,1e-6,kernelStr,10,1,1,[]);
    ModelInfo.idxL_precomputed = extract_vecchia_indices(resL.B, ModelInfo.nn_size);
    ModelInfo.idxH_precomputed = extract_vecchia_indices(resH.B, ModelInfo.nn_size);

    % ----- Optimize hyp (same starting hyp0 for all orderings -> fair) -----
    fprintf('[%d/%d] Ordering: %s | Optimizing hyp with fminunc...\n', oi, numel(ORDERINGS), ordName);


    hyp0=rand(11,1);
    tStart = tic;
    [hyp_hat, fval] = fminunc(@likelihoodVecchia_nonstat_GLS_v3, hyp0, opts_opt);
    tElapsed = toc(tStart);

    % ----- Prediction -----
    likelihoodVecchia_nonstat_GLS_v3(hyp_hat); % populate caches for prediction

    if strcmp(ModelInfo.GLSType,"adaptive")
        [mu_p, s2_p] = predict_calibratedCM3_AdaptiveGLS_v4(Xstar_s, ModelInfo);
    else
        [mu_p, s2_p] = predict_calibratedCM3_fixed(Xstar_s, ModelInfo);
    end

    y_pred = mu_p;
    u_p    = mu_p + 1.96*sqrt(s2_p);
    l_p    = mu_p - 1.96*sqrt(s2_p);

    % ----- Metrics -----
    rmse = sqrt(mean((y_true_s - y_pred).^2));
    mae  = mean(abs(y_true_s - y_pred));
    cor  = corr(y_true_s, y_pred);
    picp = mean((y_true_s >= l_p) & (y_true_s <= u_p)) * 100;

    results = [results; {ordName, fval, rmse, mae, cor, picp, mean(y_pred), std(y_pred), tElapsed}]; %#ok<AGROW>

    predStore.(matlab.lang.makeValidName(ordName)).t      = Xstar_s(:,1);
    predStore.(matlab.lang.makeValidName(ordName)).y_pred = y_pred;
    predStore.(matlab.lang.makeValidName(ordName)).l_p    = l_p;
    predStore.(matlab.lang.makeValidName(ordName)).u_p    = u_p;

    fprintf('Done: %s | NLML=%.2f | RMSE=%.4f | MAE=%.4f | Corr=%.3f | PICP95=%.1f%% | time=%.1fs\n', ...
        ordName, fval, rmse, mae, cor, picp, tElapsed);
end

%% ---------------- Print & sort ----------------
fprintf('\n==================== RESULTS (as-run) ====================\n');
disp(results);

fprintf('\n==================== SORT BY RMSE (lower is better) ====================\n');
disp(sortrows(results,'RMSE'));

fprintf('\n==================== SORT BY NLML (lower is better) ====================\n');
disp(sortrows(results,'NLML'));

%% ---------------- Plot (single station, multiple orderings) ----------------
figure('Name',sprintf('Prediction comparison | hold-out station %d', hold_id));
plot(Xstar_s(:,1), y_true_s, 'LineWidth', 1.5); hold on;
leg = ["y\_true"];

for oi = 1:numel(ORDERINGS)
    ordName = string(ORDERINGS{oi});
    key = matlab.lang.makeValidName(ordName);
    plot(predStore.(key).t, predStore.(key).y_pred, 'LineWidth', 1.2);
    leg(end+1) = ordName; %#ok<SAGROW>
end

xlabel('Time');
ylabel('ws');
title(sprintf('Out-of-sample prediction (hold-out station %d) | nn=%d | %s', hold_id, nn_size, conf_tag), 'Interpreter','none');
legend(leg, 'Interpreter','none', 'Location','best');
grid on;

%% ========================================================================
%% HELPERS
%% ========================================================================
function p = build_ordering_perm(X, ordName, seedRandSpace, seedRandom)
% X = [Time, Lat, Lon]
    n = size(X,1);
    switch ordName
        case "Station-major"
            % sort by (Lat, Lon, Time)
            [~, p] = sortrows(X, [2 3 1]);

        case "Time-major"
            % deterministic sort by (Time, Lat, Lon)
            [~, p] = sortrows(X, [1 2 3]);

        case "Time-causal+RandSpace"
            % ordered by time; within each time slice randomize spatial order
            rng(seedRandSpace);
            tvals = X(:,1);
            [tuniq, ~] = unique(tvals, 'stable');

            p = zeros(n,1);
            pos = 1;
            for it = 1:numel(tuniq)
                idx_t = find(tvals == tuniq(it));
                idx_t = idx_t(randperm(numel(idx_t)));
                p(pos:pos+numel(idx_t)-1) = idx_t(:);
                pos = pos + numel(idx_t);
            end

        case "Random"
            rng(seedRandom);
            p = randperm(n)';

        otherwise
            error("Unknown ordering '%s'", ordName);
    end
end

function idx_mat = extract_vecchia_indices(B, nn)
    n = size(B,1);
    idx_mat = zeros(n,nn);
    for i = 2:n
        cols = find(B(i,1:i-1));
        if ~isempty(cols)
            len = min(length(cols),nn);
            idx_mat(i,1:len) = cols(end-len+1:end);
        end
    end
end
