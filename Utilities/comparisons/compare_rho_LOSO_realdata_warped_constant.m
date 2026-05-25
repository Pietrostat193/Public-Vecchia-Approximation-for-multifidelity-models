function results = compare_rho_LOSO_realdata_warped_constant(data, opts)
% compare_rho_LOSO_realdata_warped_constant_warmstart
% LOSO evaluation on real dataset, fitting ONLY rho="constant" with
% MeanFunction="zero", but WITH KCDF warping + inverse mapping.
%
% NEW: warm-start the warped optimization using the previously fitted
% (non-warped) rho-constant parameters stored in a results table, i.e.:
%   opts.prev_results_const  : table from compare_rho_LOSO_realdata(...)
%                             containing ModelInfo.hyp for case "rho constant"
%   opts.use_prev_hyp        : true/false (default true)
%   opts.warm_jitter_scale   : std dev of jitter added to warm-start for restarts
%
% Required columns in data table:
%   Wind_speed, ws, Lat_LF, Lon_LF, Lat_HF, Lon_HF, Time, IDStation
%
% Dependencies (must be on path):
%   likelihoodVecchia_nonstat_GLS
%   predict_calibratedCM3              % your predictor (warped scale)
%
% Warping dependencies (WMFGP-main):
%   KCDF_Estim
%   Gen_Lookup
%   Kernel_invNS
%
% Example:
%   opts2 = struct();
%   opts2.holdout_station = 525;
%   opts2.prev_results_const = results_from_nonwarped;   % <- table
%   opts2.warping_addpath = 'C:\...\WMFGP-main';
%   results_warp = compare_rho_LOSO_realdata_warped_constant_warmstart(data, opts2);

    if nargin < 2, opts = struct(); end
    if ~isfield(opts,'holdout_station'), opts.holdout_station = []; end
    if ~isfield(opts,'do_all_stations'), opts.do_all_stations = false; end

    % --- Vecchia / model defaults
    if ~isfield(opts,'nn_size'),  opts.nn_size  = 50; end
    if ~isfield(opts,'cand_mult'),opts.cand_mult= 10; end
    if ~isfield(opts,'conditioning'), opts.conditioning = "Corr"; end
    if ~isfield(opts,'kernel'), opts.kernel = "RBF"; end
    if ~isfield(opts,'cov_type'), opts.cov_type = "RBF"; end
    if ~isfield(opts,'combination'), opts.combination = "multiplicative"; end

    % --- Force these for your requested setting
    opts.MeanFunction = "zero";
    opts.RhoFunction  = "constant";

    % --- prediction defaults (kept for compatibility; not used directly here)
    if ~isfield(opts,'seed'), opts.seed = 42; end

    % --- optimization defaults
    if ~isfield(opts,'max_iter'), opts.max_iter = 200; end
    if ~isfield(opts,'max_fun'),  opts.max_fun  = 2000; end
    if ~isfield(opts,'n_starts'), opts.n_starts = 3; end

    % --- dataset size control
    if ~isfield(opts,'time_cap_per_station'), opts.time_cap_per_station = 400; end

    % --- warping defaults
    if ~isfield(opts,'warping_kernel'), opts.warping_kernel = 'Tria'; end
    if ~isfield(opts,'warping_addpath'), opts.warping_addpath = ''; end

    % --- warm-start defaults
    if ~isfield(opts,'prev_results_const'), opts.prev_results_const = []; end
    if ~isfield(opts,'use_prev_hyp'), opts.use_prev_hyp = true; end
    if ~isfield(opts,'warm_jitter_scale'), opts.warm_jitter_scale = 0.05; end % conservative
    if ~isfield(opts,'use_prev_only_first_start'), opts.use_prev_only_first_start = true; end

    if ~isempty(opts.warping_addpath)
        addpath(opts.warping_addpath);
    end

    requiredVars = ["Wind_speed","ws","Lat_LF","Lon_LF","Lat_HF","Lon_HF","Time","IDStation"];
    missing = setdiff(requiredVars, string(data.Properties.VariableNames));
    if ~isempty(missing)
        error("Missing required columns: %s", strjoin(missing,", "));
    end

    % Ensure numeric
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

    results_rows = [];
    row = 0;

    for h = 1:numel(holdouts)
        hold_id = holdouts(h);
        fprintf('\n================ LOSO holdout station %g (WARPED rho=constant + warmstart) ================\n', hold_id);

        is_hold   = (data.IDStation == hold_id);
        train_tbl = data(~is_hold,:);
        test_tbl  = data(is_hold,:);

        % -------- optional cap time points per station (train & test)
        if ~isempty(opts.time_cap_per_station)
            rng(opts.seed);
            train_tbl = cap_times_per_station(train_tbl, opts.time_cap_per_station);
            test_tbl  = cap_times_per_station(test_tbl,  opts.time_cap_per_station);
        end

        % Build training arrays
        X_L = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
        y_L = double(train_tbl.Wind_speed);

        X_H = [double(train_tbl.Time), double(train_tbl.Lat_HF), double(train_tbl.Lon_HF)];
        y_H = double(train_tbl.ws);

        % Test points: predict HF at held-out station
        Xstar  = [double(test_tbl.Time), double(test_tbl.Lat_HF), double(test_tbl.Lon_HF)];
        y_true = double(test_tbl.ws);

        fprintf('Train: nL=%d, nH=%d | Test(HF)=%d\n', size(X_L,1), size(X_H,1), size(Xstar,1));

        % Sorting (stabilizes Vecchia ordering)
        [~, pL] = sortrows(X_L, [2 3 1]);
        X_L = X_L(pL,:); y_L = y_L(pL);

        [~, pH] = sortrows(X_H, [2 3 1]);
        X_H = X_H(pH,:); y_H = y_H(pH);

        % ===================== WARPING STEP =====================
        kernelW = opts.warping_kernel;

        % Warp training outputs; keep lookup to inverse-map predictions.
        [~, y_H_warp, bgk_band_y_H, ~] = KCDF_Estim(y_H, kernelW);
        lookup_y_H = Gen_Lookup(y_H, bgk_band_y_H, kernelW);

        [~, y_L_warp, bgk_band_y_L, ~] = KCDF_Estim(y_L, kernelW);
        lookup_y_L = Gen_Lookup(y_L, bgk_band_y_L, kernelW); %#ok<NASGU>

        % ===================== FIT rho=constant on warped scale =====================
        global ModelInfo;
        ModelInfo = struct();

        ModelInfo.X_L = X_L;  ModelInfo.y_L = y_L_warp;
        ModelInfo.X_H = X_H;  ModelInfo.y_H = y_H_warp;

        ModelInfo.jitter = 1e-8;
        ModelInfo.nn_size = opts.nn_size;
        ModelInfo.kernel = opts.kernel;
        ModelInfo.conditioning = opts.conditioning;
        ModelInfo.cand_mult = opts.cand_mult;
        ModelInfo.show_path_diag = false;

        ModelInfo.MeanFunction = opts.MeanFunction;   % "zero"
        ModelInfo.cov_type = opts.cov_type;
        ModelInfo.combination = opts.combination;
        ModelInfo.RhoFunction = opts.RhoFunction;     % "constant"

        % Reset neighbor caches (safety)
        if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
        if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

        obj = @(hh) likelihoodVecchia_nonstat_GLS(hh);

        % ---- initialization: default + optional warm-start from previous non-warped fit
        hyp_default = default_hyp_init("constant");
        hyp0 = hyp_default;

        hyp_prev = [];
        if opts.use_prev_hyp && ~isempty(opts.prev_results_const)
            hyp_prev = extract_prev_hyp(opts.prev_results_const, hold_id);
            if ~isempty(hyp_prev) && numel(hyp_prev) == numel(hyp_default)
                hyp0 = hyp_prev;
                fprintf('Warm-start: using previous non-warped hyp for station %g.\n', hold_id);
            else
                fprintf('Warm-start: no usable previous hyp found for station %g (fallback to default init).\n', hold_id);
            end
        end

        % ---- multistart (first start = warm, others jitter)
        rng(opts.seed + 1000*h);
        [hyp_hat, nlml] = run_multistart(obj, hyp0, opts, hyp_prev);

        % finalize (cache internal objects)
        ModelInfo.hyp = hyp_hat;
        nlml = likelihoodVecchia_nonstat_GLS(hyp_hat);

        % ===================== PREDICTION (warped HF scale) =====================
        % Must return HF predictions in warped scale (matching ModelInfo.y_H)
        p_warp = predict_calibratedCM3(Xstar, ModelInfo);

        % ===================== INVERSE WARPING back to original HF scale =====================
        y_pred = Kernel_invNS(p_warp, lookup_y_H);

        % metrics in original scale
        rmse = sqrt(mean((y_pred - y_true).^2));
        mae  = mean(abs(y_pred - y_true));

        % store
        row = row + 1;
        results_rows(row).holdout_station = hold_id;
        results_rows(row).case_name = "rho constant (warped warmstart)";
        results_rows(row).RhoFunction = string(ModelInfo.RhoFunction);
        results_rows(row).MeanFunction = string(ModelInfo.MeanFunction);
        results_rows(row).NLML = nlml;
        results_rows(row).RMSE = rmse;
        results_rows(row).MAE  = mae;
        results_rows(row).nTrainL = size(X_L,1);
        results_rows(row).nTrainH = size(X_H,1);
        results_rows(row).nTestH  = size(Xstar,1);

        results_rows(row).ModelInfo = ModelInfo; %#ok<NASGU>
        results_rows(row).hyp_init_used = hyp0;  %#ok<NASGU>
        results_rows(row).hyp_prev_used = hyp_prev; %#ok<NASGU>

        results_rows(row).Xstar  = Xstar;   %#ok<NASGU>
        results_rows(row).y_true = y_true;  %#ok<NASGU>
        results_rows(row).mu_pred = y_pred; %#ok<NASGU>
        results_rows(row).mu_pred_warped = p_warp; %#ok<NASGU>

        results_rows(row).lookup_y_H = lookup_y_H; %#ok<NASGU>
        results_rows(row).bgk_band_y_H = bgk_band_y_H; %#ok<NASGU>
        results_rows(row).bgk_band_y_L = bgk_band_y_L; %#ok<NASGU>

        fprintf('[rho const (warped warmstart)] NLML=%.3f | RMSE=%.5f | MAE=%.5f\n', nlml, rmse, mae);

        % ===================== PLOTS =====================
        figure;
        plot(y_true,'k.-','DisplayName','HF true'); hold on;
        plot(y_pred,'o-','DisplayName','HF predicted (inv-warp)');
        grid on; legend('Location','best');
        title(sprintf('Holdout station %g | nn=%d | rho const (warped warmstart)', hold_id, opts.nn_size));
        xlabel('test index'); ylabel('ws (HF, original scale)');

        figure; grid on; hold on;
        scatter(y_true, y_pred, 22, 'filled');
        xlabel('HF true'); ylabel('HF predicted (inv-warp)');
        title(sprintf('Holdout station %g | Parity plot | rho const (warped warmstart)', hold_id));

        figure; grid on; hold on;
        r = y_pred - y_true;
        plot(r,'o-','DisplayName','residuals (inv-warp)');
        yline(0,'k-');
        legend('Location','best');
        title(sprintf('Holdout station %g | Residuals | rho const (warped warmstart)', hold_id));
        xlabel('test index'); ylabel('pred - true');
    end

    results = struct2table(results_rows);

    if height(results) > 1
        fprintf('\n=========== SUMMARY (mean over holdouts) ===========\n');
        G = groupsummary(results, "RhoFunction", "mean", ["RMSE","MAE","NLML"]);
        disp(G);
    end
end

% ============================ helpers ============================

function hyp_prev = extract_prev_hyp(prev_tbl, hold_id)
% Extract previous hyp (11x1) for a given holdout station from a prior results table.
% Expected that prev_tbl has columns:
%   holdout_station, RhoFunction, ModelInfo
% where ModelInfo is a struct containing .hyp
    hyp_prev = [];

    try
        if ~istable(prev_tbl), return; end
        if ~all(ismember(["holdout_station","RhoFunction","ModelInfo"], string(prev_tbl.Properties.VariableNames)))
            return;
        end

        % Robust compare (RhoFunction can be string/cellstr/char)
        rhoCol = prev_tbl.RhoFunction;
        if iscell(rhoCol), rhoCol = string(rhoCol); end
        rhoCol = string(rhoCol);

        idx = (double(prev_tbl.holdout_station) == double(hold_id)) & (rhoCol == "constant");
        if ~any(idx), return; end

        k = find(idx, 1, 'first');
        Mi = prev_tbl.ModelInfo{k};  % ModelInfo stored as cell of structs in most cases
        if isstruct(Mi) && isfield(Mi,'hyp') && ~isempty(Mi.hyp)
            hyp_prev = Mi.hyp(:);
        end
    catch
        hyp_prev = [];
    end
end

function tbl = cap_times_per_station(tbl, capN)
% Keep at most capN rows per station, sampling uniformly over time.
    if capN <= 0 || height(tbl) <= capN, return; end
    ids = unique(tbl.IDStation);
    keep = false(height(tbl),1);
    for i = 1:numel(ids)
        idx = find(tbl.IDStation == ids(i));
        if numel(idx) <= capN
            keep(idx) = true;
        else
            [~, ord] = sort(tbl.Time(idx));
            idx = idx(ord);
            pick = round(linspace(1, numel(idx), capN));
            keep(idx(pick)) = true;
        end
    end
    tbl = tbl(keep,:);
end

function [best_x, best_f] = run_multistart(obj, x0, opts, x_prev)
% Multi-start optimization with warm-start + jittered restarts.
% If x_prev is provided and opts.use_prev_only_first_start=true, then:
%   start{1} = x0 (warm)
%   start{k>1} = x0 + jitter
% Otherwise, still uses x0 for start{1} and jitter for others.

    n_starts = max(1, opts.n_starts);
    starts = cell(n_starts,1);
    starts{1} = x0;

    % small perturbations (log-params tolerate 0.02-0.2). You can tune opts.warm_jitter_scale.
    js = opts.warm_jitter_scale;
    for k = 2:n_starts
        jitter = js * randn(size(x0));
        starts{k} = x0 + jitter;
    end

    best_f = inf;
    best_x = x0;

    use_fminunc = license('test','optimization_toolbox');
    if use_fminunc
        o = optimoptions('fminunc', ...
            'Display','iter', ...
            'Algorithm','quasi-newton', ...
            'MaxIterations', opts.max_iter, ...
            'MaxFunctionEvaluations', opts.max_fun, ...
            'OptimalityTolerance', 1e-3, ...
            'StepTolerance', 1e-8);
    else
        o = optimset('Display','iter', ...
            'MaxIter', opts.max_iter, ...
            'MaxFunEvals', opts.max_fun);
    end

    for k = 1:n_starts
        xk0 = starts{k};
        try
            if use_fminunc
                [xk, fk] = fminunc(obj, xk0, o);
            else
                [xk, fk] = fminsearch(obj, xk0, o);
            end
        catch ME
            warning('Start %d failed: %s', k, ME.message);
            continue;
        end

        if fk < best_f
            best_f = fk;
            best_x = xk;
        end
    end
end

function hyp0 = default_hyp_init(RhoFunction)
% Same base initialization you used (11 params for constant rho).
    RF = string(RhoFunction);

    switch RF
        case "constant"
            hyp0 = zeros(11,1);
        otherwise
            hyp0 = zeros(14,1);
    end

    % base 11
    hyp0(1)  = log(1.0);  hyp0(2)  = log(0.20);  % LF time
    hyp0(3)  = log(1.0);  hyp0(4)  = log(0.20);  % HF time
    hyp0(5)  = 0.6;                              % rho
    hyp0(6)  = log(0.10); hyp0(7)  = log(0.10);  % eps_LF, eps_HF
    hyp0(8)  = log(1.0);  hyp0(9)  = log(1.0);   % LF space
    hyp0(10) = log(1.0);  hyp0(11) = log(1.0);   % HF space
end
