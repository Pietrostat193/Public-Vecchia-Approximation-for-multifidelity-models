function results = compare_models_LOSO_realdata_withPI(data, paramBank, opts)
% Compare 3×MFGP (const / warped-const / empirical-rho) + GP-3D under LOSO,
% using *fixed* hyperparameters from a pre-fit parameter bank, and computing
% prediction intervals + coverage.
%
% INPUTS
%   data      : table with columns
%               Wind_speed, ws, Lat_LF, Lon_LF, Lat_HF, Lon_HF, Time, IDStation
%   paramBank : struct containing pre-fit hyperparameters for the 3 MFGP models
%               and (optionally) GP3D. See "build_paramBank_example" below.
%   opts      : struct of options (optional)
%
% OUTPUT
%   results   : table with per-station, per-model metrics including PI80/95.
%
% DEPENDENCIES (your project)
%   likelihoodVecchia_nonstat_GLS
%   predictVecchia_CM_calibrated2   (should ideally support [mu,var] outputs)
%   (optional for warped) likelihoodVecchia_nonstat_GLS_warped
%   (optional for warped) predictVecchia_CM_calibrated2_warped
%
% NOTES
%   - This function does *not* re-optimize hyperparameters: it rebuilds caches
%     per holdout (Vecchia factors, GLS offsets, etc.) by calling likelihood
%     once with the *fixed* hyp for that model.
%   - PI computation assumes Gaussian predictive distribution.
%
% Example:
%   paramBank = build_paramBank_example();
%   opts = struct('do_all_stations',true,'time_cap_per_station',400,'seed',42);
%   results = compare_models_LOSO_realdata_withPI(data.sorted_data, paramBank, opts);

    if nargin < 3, opts = struct(); end
    if ~isfield(opts,'holdout_station'), opts.holdout_station = []; end
    if ~isfield(opts,'do_all_stations'), opts.do_all_stations = false; end

    % --- Vecchia / model defaults
    if ~isfield(opts,'nn_size'),  opts.nn_size  = 50; end
    if ~isfield(opts,'cand_mult'),opts.cand_mult= 10; end
    if ~isfield(opts,'conditioning'), opts.conditioning = "Corr"; end
    if ~isfield(opts,'kernel'), opts.kernel = "RBF"; end

    % MeanFunction in your likelihood is GLS-centered internally; keep "zero" here.
    if ~isfield(opts,'MeanFunction'), opts.MeanFunction = "zero"; end

    % --- prediction / calibration defaults
    if ~isfield(opts,'calib_mode'), opts.calib_mode = "global_affine"; end
    if ~isfield(opts,'gamma_clip'), opts.gamma_clip = [0.25, 4.0]; end
    if ~isfield(opts,'lambda_ridge'), opts.lambda_ridge = 1e-8; end

    % --- dataset size control
    if ~isfield(opts,'time_cap_per_station'), opts.time_cap_per_station = 400; end
    if ~isfield(opts,'seed'), opts.seed = 42; end

    requiredVars = ["Wind_speed","ws","Lat_LF","Lon_LF","Lat_HF","Lon_HF","Time","IDStation"];
    missing = setdiff(requiredVars, string(data.Properties.VariableNames));
    if ~isempty(missing)
        error("Missing required columns: %s", strjoin(missing,", "));
    end

    if ~isnumeric(data.Time), data.Time = double(data.Time); end
    if ~isnumeric(data.IDStation), data.IDStation = double(data.IDStation); end

    stations = unique(data.IDStation);
    stations = stations(:);

    if isempty(opts.holdout_station)
        holdouts = stations;
        if ~opts.do_all_stations
            holdouts = stations(1);
        end
    else
        holdouts = opts.holdout_station(:);
    end

    % -------------------- Model registry --------------------
    % Keep names consistent with your reporting.
    models = { ...
        struct('name',"GP-ST (approx)",               'type',"GP3D"), ...
        struct('name',"MFGP (rho constant)",          'type',"MFGP", 'rho',"constant",          'warped',false), ...
        struct('name',"MFGP (rho constant, warped)",  'type',"MFGP", 'rho',"constant",          'warped',true ), ...
        struct('name',"MFGP (rho GP_scaled_emp)",     'type',"MFGP", 'rho',"GP_scaled_empirical",'warped',false) ...
    };

    % z-scores for central PIs
    z80 = 1.2815515655446004; % norminv(0.9)
    z95 = 1.959963984540054;  % norminv(0.975)

    results_rows = [];
    row = 0;

    for h = 1:numel(holdouts)
        hold_id = holdouts(h);
        fprintf('\n================ LOSO holdout station %g ================\n', hold_id);

        is_hold   = (data.IDStation == hold_id);
        train_tbl = data(~is_hold,:);
        test_tbl  = data(is_hold,:);

        % ---- cap times per station (train/test)
        if ~isempty(opts.time_cap_per_station)
            rng(opts.seed);
            train_tbl = cap_times_per_station(train_tbl, opts.time_cap_per_station);
            test_tbl  = cap_times_per_station(test_tbl,  opts.time_cap_per_station);
        end

        % ---- Build training arrays
        X_L = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
        y_L = double(train_tbl.Wind_speed);

        X_H = [double(train_tbl.Time), double(train_tbl.Lat_HF), double(train_tbl.Lon_HF)];
        y_H = double(train_tbl.ws);

        % ---- Test points: predict HF at held-out station
        Xstar  = [double(test_tbl.Time), double(test_tbl.Lat_HF), double(test_tbl.Lon_HF)];
        y_true = double(test_tbl.ws);

        fprintf('Train: nL=%d, nH=%d | Test(HF)=%d\n', size(X_L,1), size(X_H,1), size(Xstar,1));

        % Sorting (station-major: lat,lon then time) — consistent with your paper
        [~, pL] = sortrows(X_L, [2 3 1]);
        X_L = X_L(pL,:); y_L = y_L(pL);

        [~, pH] = sortrows(X_H, [2 3 1]);
        X_H = X_H(pH,:); y_H = y_H(pH);

        % -------------------- Evaluate all models (fixed params) --------------------
        for m = 1:numel(models)
            M = models{m};

            switch M.type
                case "GP3D"
                    % Either use pre-fit GP3D per holdout, or fit quickly here.
                    % (We keep it simple and fit here; you can cache per holdout if desired.)
                    [mu_pred, var_pred] = predict_gp3d(X_H, y_H, Xstar, paramBank, hold_id);

                case "MFGP"
                    [mu_pred, var_pred] = predict_mfgp_fixed( ...
                        X_L, y_L, X_H, y_H, Xstar, ...
                        paramBank, hold_id, ...
                        M.rho, M.warped, ...
                        opts);

                otherwise
                    error("Unknown model type: %s", M.type);
            end

            % ---- Prediction intervals + metrics
            s_pred = sqrt(max(var_pred, 0)); % guard negative numerical noise

            lo80 = mu_pred - z80*s_pred;
            hi80 = mu_pred + z80*s_pred;

            lo95 = mu_pred - z95*s_pred;
            hi95 = mu_pred + z95*s_pred;

            rmse = sqrt(mean((mu_pred - y_true).^2));
            mae  = mean(abs(mu_pred - y_true));

            % correlation (robust guard for degenerate cases)
            if numel(y_true) >= 2 && std(y_true) > 0 && std(mu_pred) > 0
                corr_val = corr(y_true, mu_pred);
            else
                corr_val = NaN;
            end

            PI80_cov = mean((y_true >= lo80) & (y_true <= hi80));
            PI95_cov = mean((y_true >= lo95) & (y_true <= hi95));
            PI80_w   = mean(hi80 - lo80);
            PI95_w   = mean(hi95 - lo95);
            pv_mean  = mean(var_pred);

            row = row + 1;
            results_rows(row).Station         = hold_id;
            results_rows(row).Model           = M.name;
            results_rows(row).RMSE            = rmse;
            results_rows(row).MAE             = mae;
            results_rows(row).CORR            = corr_val;
            results_rows(row).PI80_Coverage   = PI80_cov;
            results_rows(row).PI95_Coverage   = PI95_cov;
            results_rows(row).PI80_WidthMean  = PI80_w;
            results_rows(row).PI95_WidthMean  = PI95_w;
            results_rows(row).PredVarMean     = pv_mean;

            fprintf('[%s] RMSE=%.3f | MAE=%.3f | CORR=%.3f | PI80=%.2f | PI95=%.2f\n', ...
                M.name, rmse, mae, corr_val, PI80_cov, PI95_cov);
        end

        % -------------------- Optional: plot (1 station) --------------------
        if isfield(opts,'plot_each_station') && opts.plot_each_station
            plot_station_summary(results_rows, hold_id);
        end
    end

    results = struct2table(results_rows);

    % ---- Summary tables
    fprintf('\n=========== SUMMARY (mean over stations) ===========\n');
    disp(groupsummary(results, "Model", "mean", ["RMSE","MAE","CORR","PI80_Coverage","PI95_Coverage","PI80_WidthMean","PI95_WidthMean","PredVarMean"]));

    fprintf('\n=========== SUMMARY (std over stations) ===========\n');
    disp(groupsummary(results, "Model", "std",  ["RMSE","MAE","CORR","PI80_Coverage","PI95_Coverage","PI80_WidthMean","PI95_WidthMean","PredVarMean"]));
end

% ======================================================================
%                   FIXED-PARAM MFGP PREDICTION (Vecchia)
% ======================================================================
function [mu_pred, var_pred] = predict_mfgp_fixed(X_L, y_L, X_H, y_H, Xstar, paramBank, hold_id, RhoFunction, isWarped, opts)
% Rebuilds caches via likelihood call with *fixed* hyp from paramBank, then predicts.
    global ModelInfo;
    ModelInfo = struct();

    ModelInfo.X_L = X_L;  ModelInfo.y_L = y_L;
    ModelInfo.X_H = X_H;  ModelInfo.y_H = y_H;

    ModelInfo.jitter = 1e-8;
    ModelInfo.nn_size = opts.nn_size;
    ModelInfo.kernel = opts.kernel;
    ModelInfo.conditioning = opts.conditioning;
    ModelInfo.cand_mult = opts.cand_mult;
    ModelInfo.show_path_diag = false;

    ModelInfo.MeanFunction = opts.MeanFunction;  % GLS is handled inside your likelihood
    ModelInfo.cov_type = "RBF";
    ModelInfo.combination = "multiplicative";
    ModelInfo.RhoFunction = string(RhoFunction);

    % pick hyperparameters (fixed)
    hyp = get_fixed_hyp(paramBank, hold_id, string(RhoFunction), isWarped);

    % build caches (Vecchia factors, H chol, GLS offsets, gprModel_rho, etc.)
    if isWarped
        if exist('likelihoodVecchia_nonstat_GLS_warped','file') ~= 2
            error('Warped requested but likelihoodVecchia_nonstat_GLS_warped not found on path.');
        end
        likelihoodVecchia_nonstat_GLS_warped(hyp);
    else
        likelihoodVecchia_nonstat_GLS(hyp);
    end

    % predict + variance
    predOpts = struct();
    predOpts.calib_mode   = char(opts.calib_mode);
    predOpts.gamma_clip   = opts.gamma_clip;
    predOpts.lambda_ridge = opts.lambda_ridge;
    predOpts.gamma_subset = [];
    predOpts.seed         = opts.seed;

    if isWarped
        if exist('predictVecchia_CM_calibrated2_warped','file') == 2
            [mu_pred, var_pred] = get_pred_mean_var(@predictVecchia_CM_calibrated2_warped, Xstar, predOpts);
        else
            error('Warped requested but predictVecchia_CM_calibrated2_warped not found on path.');
        end
    else
        [mu_pred, var_pred] = get_pred_mean_var(@predictVecchia_CM_calibrated2, Xstar, predOpts);
    end
end

function hyp = get_fixed_hyp(paramBank, hold_id, rhoKey, isWarped)
% Expected schema:
%   paramBank.MFGP.constant.hyp
%   paramBank.MFGP.constant_warped.hyp
%   paramBank.MFGP.GP_scaled_empirical.hyp
%
% You can store either one shared hyp vector, or a per-station map.
% This helper supports both patterns.

    if ~isfield(paramBank,'MFGP')
        error('paramBank must contain paramBank.MFGP.* fields.');
    end

    if rhoKey == "constant" && ~isWarped
        node = paramBank.MFGP.constant;
    elseif rhoKey == "constant" && isWarped
        node = paramBank.MFGP.constant_warped;
    elseif rhoKey == "GP_scaled_empirical"
        node = paramBank.MFGP.GP_scaled_empirical;
    else
        error('Unknown rhoKey/isWarped combination.');
    end

    if isfield(node,'hyp_by_station') && ~isempty(node.hyp_by_station)
        key = sprintf('s%d', round(hold_id));
        if isfield(node.hyp_by_station, key)
            hyp = node.hyp_by_station.(key);
            return;
        else
            warning('No per-station hyp found for %s, using shared hyp.', key);
        end
    end

    if ~isfield(node,'hyp') || isempty(node.hyp)
        error('paramBank missing hyp for requested model.');
    end
    hyp = node.hyp;
end

% ======================================================================
%                           GP-3D BASELINE
% ======================================================================
function [mu_pred, var_pred] = predict_gp3d(X_H, y_H, Xstar, paramBank, hold_id)
% Simple GP baseline on 3 spatio-temporal coords with predictive variance.

    % Optional: use cached GP model per station (if you want).
    if isfield(paramBank,'GP3D') && isfield(paramBank.GP3D,'model_by_station') && ~isempty(paramBank.GP3D.model_by_station)
        key = sprintf('s%d', round(hold_id));
        if isfield(paramBank.GP3D.model_by_station, key)
            gprM = paramBank.GP3D.model_by_station.(key);
            [mu_pred, sd_pred] = predict(gprM, Xstar);
            var_pred = sd_pred.^2;
            return;
        end
    end

    % Fit quickly (exact for moderate nH). If too slow, cache outside and pass in.
    gprM = fitrgp(X_H, y_H, ...
        'KernelFunction','ardsquaredexponential', ...
        'BasisFunction','none', ...
        'FitMethod','exact', ...
        'PredictMethod','exact', ...
        'Standardize',false, ...
        'Sigma',0.01);

    [mu_pred, sd_pred] = predict(gprM, Xstar);
    var_pred = sd_pred.^2;
end

% ======================================================================
%                   SAFE MEAN/VAR FETCH FROM PREDICTOR
% ======================================================================
function [mu, v] = get_pred_mean_var(predictFcn, Xstar, predOpts)
% Tries [mu,var] first; if only mu is returned, falls back to a small nugget variance.
    try
        [mu, v] = predictFcn(Xstar, predOpts);
        if isempty(v)
            v = 1e-6 * ones(size(mu));
        end
    catch
        mu = predictFcn(Xstar, predOpts);
        % Fallback variance (NOT ideal): replace with your real variance routine if available.
        v = 1e-6 * ones(size(mu));
        warning('Predict function did not return variance. Using small fallback variance for PI.');
    end
end

% ======================================================================
%                             HELPERS
% ======================================================================
function tbl = cap_times_per_station(tbl, capN)
% Keep at most capN rows per station, sampling evenly over sorted Time.
    if isempty(capN) || capN <= 0, return; end
    ids = unique(tbl.IDStation);
    keep = false(height(tbl),1);
    for i = 1:numel(ids)
        idx = find(tbl.IDStation == ids(i));
        if numel(idx) <= capN
            keep(idx) = true;
        else
            [~, ord] = sort(tbl.Time(idx));
            idx = idx(ord);
            pick = unique(round(linspace(1, numel(idx), capN)));
            keep(idx(pick)) = true;
        end
    end
    tbl = tbl(keep,:);
end

function plot_station_summary(results_rows, station_id)
% Lightweight plot of true vs predictions if you store predictions externally.
% Here we only print a note; you can extend this to save per-station predictions.
    fprintf('[plot] station %g done (extend plot_station_summary to use stored preds).\n', station_id);
end

% ======================================================================
%                   EXAMPLE PARAM BANK CONSTRUCTION
% ======================================================================
function paramBank = build_paramBank_example()
% A template paramBank you should fill with your pre-fit hyp vectors.
%
% Option A (shared hyp for all holdouts):
%   paramBank.MFGP.constant.hyp = hyp_const;
%   paramBank.MFGP.constant_warped.hyp = hyp_const_warp;
%   paramBank.MFGP.GP_scaled_empirical.hyp = hyp_emp;
%
% Option B (per-station hyp; recommended if you fitted each LOSO fold separately):
%   paramBank.MFGP.constant.hyp_by_station.s100 = hyp_const_station100;
%   paramBank.MFGP.constant.hyp_by_station.s102 = hyp_const_station102;
%   ... etc ...

    paramBank = struct();

    paramBank.MFGP = struct();

    % ---- Fill these with your best fixed hyperparameters
    paramBank.MFGP.constant = struct('hyp', [], 'hyp_by_station', struct());
    paramBank.MFGP.constant_warped = struct('hyp', [], 'hyp_by_station', struct());
    paramBank.MFGP.GP_scaled_empirical = struct('hyp', [], 'hyp_by_station', struct());

    % Optional GP3D cache (if you prefit per station)
    paramBank.GP3D = struct();
    paramBank.GP3D.model_by_station = struct();
end
