%% Trial_LOSO_1station_fit_MFGP_with_PI.m
% End-to-end trial: fit 3 MFGP variants + compute predictive intervals with predict_calibratedCM3.
% 1 holdout station only, 100 time points per station.

clear; clc; close all;

%% ---------------- USER SETTINGS ----------------
hold_id = 100;
capN    = 100;

% >>> If your real dataset is in a .mat file, set it here (otherwise leave empty):
dataFile = "C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy\South_Lombardy_sorted_data.mat";   % e.g. "RealData_south.mat" or "myRealData.mat"

opts = struct();
opts.nn_size       = 50;
opts.cand_mult     = 10;
opts.conditioning  = "Corr";
opts.kernel        = "RBF";
opts.MeanFunction  = "zero";
opts.cov_type      = "RBF";
opts.combination   = "multiplicative";

% Optimization (small for workflow check; increase later)
opts.max_iter = 60;
opts.max_fun  = 600;
opts.n_starts = 2;
opts.seed     = 42;

% Predictor calibration options (consumed by predict_calibratedCM3)
predOpts = struct();
predOpts.calib_mode   = "global_affine";
predOpts.gamma_clip   = [0.25, 4.0];
predOpts.lambda_ridge = 1e-8;
predOpts.gamma_subset = [];
predOpts.seed         = opts.seed;

predOpts.bin_Kt      = 4;
predOpts.bin_Ks      = 4;
predOpts.bin_min_pts = 15;

% Interval z-scores (avoid stats toolbox dependency)
z80 = 1.2815515655446004;  % norminv(0.9)
z95 = 1.959963984540054;   % norminv(0.975)

%% ---------------- LOAD YOUR SAVED RESULTS (warm-start sources) ----------------
load('Results_real__south_L.mat');  % expects table "results"
load('results_warp.mat');          % expects table "results_warp"

if ~exist('results','var') || ~istable(results)
    error('Results_real__south_L.mat must contain a table named "results".');
end
if ~exist('results_warp','var') || ~istable(results_warp)
    error('results_warp.mat must contain a table named "results_warp".');
end

%% ---------------- LOAD REAL DATA TABLE (AUTO-DETECT) ----------------
% Option A: load from file if provided
if strlength(string(dataFile)) > 0
    S = load(dataFile);
    data = localFindDataTableInStruct(S);
else
    % Option B: try to find in workspace (maybe you loaded it already)
    data = localFindDataTableInWorkspace();
end

% Validate required variables
requiredVars = ["Wind_speed","ws","Lat_LF","Lon_LF","Lat_HF","Lon_HF","Time","IDStation"];
missing = setdiff(requiredVars, string(data.Properties.VariableNames));
if ~isempty(missing)
    error("The detected data table is missing required columns: %s", strjoin(missing,", "));
end

% Ensure numeric
if ~isnumeric(data.Time),      data.Time      = double(data.Time); end
if ~isnumeric(data.IDStation), data.IDStation = double(data.IDStation); end


   
%% ---------------- SPLIT LOSO ----------------
is_hold   = (data.IDStation == hold_id);
train_tbl = data(~is_hold,:);
test_tbl  = data(is_hold,:);

if height(test_tbl) == 0
    error('No rows found for holdout station %g in data.IDStation.', hold_id);
end

% Cap time points per station (train + test)
train_tbl = cap_times_per_station(train_tbl, capN, opts.seed);
test_tbl  = cap_times_per_station(test_tbl,  capN, opts.seed);

% Training arrays
X_L = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
y_L = double(train_tbl.Wind_speed);

X_H = [double(train_tbl.Time), double(train_tbl.Lat_HF), double(train_tbl.Lon_HF)];
y_H = double(train_tbl.ws);

% Test points (HF)
Xstar  = [double(test_tbl.Time), double(test_tbl.Lat_HF), double(test_tbl.Lon_HF)];
y_true = double(test_tbl.ws);

fprintf('\nLOSO holdout station %g\n', hold_id);
fprintf('Train sizes: nL=%d, nH=%d | Test(HF)=%d\n', size(X_L,1), size(X_H,1), size(Xstar,1));

% Sorting for stable Vecchia ordering (space-major then time)
[~, pL] = sortrows(X_L, [2 3 1]);
X_L = X_L(pL,:); y_L = y_L(pL);

[~, pH] = sortrows(X_H, [2 3 1]);
X_H = X_H(pH,:); y_H = y_H(pH);

%% ---------------- DEFINE CASES ----------------
cases = {
    struct('key',"MFGP_const",     'name',"rho constant",          'RhoFunction',"constant",            'warp',false)
    struct('key',"MFGP_empGP",     'name',"rho GP_scaled_emp",     'RhoFunction',"GP_scaled_empirical",'warp',false)
    struct('key',"MFGP_constWarp", 'name',"rho constant (warped)", 'RhoFunction',"constant",            'warp',true )
};

OUT = table();

%% ---------------- FIT + PREDICT EACH CASE ----------------
for c = 1:numel(cases)
    C = cases{c};

    fprintf('\n---------------- %s ----------------\n', C.name);

    % Create / populate global ModelInfo expected by likelihood + predictor
    global ModelInfo
    ModelInfo = struct();
    ModelInfo.X_L = X_L;
    ModelInfo.X_H = X_H;
    ModelInfo.cov_type     = opts.cov_type;
    ModelInfo.combination  = opts.combination;
    ModelInfo.RhoFunction  = C.RhoFunction;

    ModelInfo.jitter       = 1e-8;
    ModelInfo.nn_size      = opts.nn_size;
    ModelInfo.kernel       = opts.kernel;
    ModelInfo.conditioning = opts.conditioning;
    ModelInfo.cand_mult    = opts.cand_mult;
    ModelInfo.show_path_diag = false;
    ModelInfo.MeanFunction = opts.MeanFunction;

    % ---- warping (monotone log1p; inverse expm1)
    if ~C.warp
        ModelInfo.y_L = y_L;
        ModelInfo.y_H = y_H;
    else
        ModelInfo.y_L = log1p(max(y_L,0));
        ModelInfo.y_H = log1p(max(y_H,0));
    end

    % ---- objective
    obj = @(hh) likelihoodVecchia_nonstat_GLS(hh);

    % ---- warm-start hyp (try load from saved results objects)
    hyp0 = getWarmStartHyp(C, hold_id, results, results_warp);

    % ---- multi-start optimization
    rng(opts.seed + 100*c);
    [hyp_hat, nlml_hat] = run_multistart(obj, hyp0, opts);

    % ---- finalize Vecchia caches (debug_vecchia needed by predictor)
    ModelInfo.hyp = hyp_hat;
    nlml_hat = likelihoodVecchia_nonstat_GLS(hyp_hat); %#ok<NASGU>

    % ---- predict mean + variance using YOUR function
    [mu_w, s2_w] = predict_calibratedCM3(Xstar, predOpts);

    mu_w = mu_w(:);
    s2_w = max(0, s2_w(:));

    % ---- unwarp if needed + PI
    if ~C.warp
        mu_pred = mu_w;
        s2_pred = s2_w;
        PI80 = [mu_pred - z80*sqrt(s2_pred), mu_pred + z80*sqrt(s2_pred)];
        PI95 = [mu_pred - z95*sqrt(s2_pred), mu_pred + z95*sqrt(s2_pred)];
    else
        PI80_w = [mu_w - z80*sqrt(s2_w), mu_w + z80*sqrt(s2_w)];
        PI95_w = [mu_w - z95*sqrt(s2_w), mu_w + z95*sqrt(s2_w)];
        mu_pred = expm1(mu_w);

        PI80 = [expm1(PI80_w(:,1)), expm1(PI80_w(:,2))];
        PI95 = [expm1(PI95_w(:,1)), expm1(PI95_w(:,2))];

        % delta-method approx variance on original scale
        s2_pred = (exp(mu_w).^2) .* s2_w;
    end

    % ---- metrics
    rmse = sqrt(mean((mu_pred - y_true).^2));
    mae  = mean(abs(mu_pred - y_true));
    corrv = corr(mu_pred, y_true);

    % ---- coverage + mean width
    PI80_cov = mean(y_true >= PI80(:,1) & y_true <= PI80(:,2));
    PI95_cov = mean(y_true >= PI95(:,1) & y_true <= PI95(:,2));
    PI80_wid = mean(PI80(:,2) - PI80(:,1));
    PI95_wid = mean(PI95(:,2) - PI95(:,1));

    fprintf('NLML=%.2f | RMSE=%.4f | MAE=%.4f | CORR=%.4f\n', nlml_hat, rmse, mae, corrv);
    fprintf('PI80: cov=%.2f, meanWidth=%.4f | PI95: cov=%.2f, meanWidth=%.4f\n', ...
        PI80_cov, PI80_wid, PI95_cov, PI95_wid);

    % ---- store one-row summary + keep vectors for inspection
    newRow = table();
    newRow.holdout_station = hold_id;
    newRow.modelKey        = string(C.key);
    newRow.case_name       = string(C.name);
    newRow.RhoFunction     = string(C.RhoFunction);
    newRow.warped          = C.warp;
    newRow.NLML            = nlml_hat;
    newRow.RMSE            = rmse;
    newRow.MAE             = mae;
    newRow.CORR            = corrv;
    newRow.PI80_Coverage   = PI80_cov;
    newRow.PI80_WidthMean  = PI80_wid;
    newRow.PI95_Coverage   = PI95_cov;
    newRow.PI95_WidthMean  = PI95_wid;
    newRow.PredVarMean     = mean(s2_pred);

    newRow.mu_pred = {mu_pred};
    newRow.s2_pred = {s2_pred};
    newRow.y_true  = {y_true};
    newRow.Xstar   = {Xstar};
    newRow.PI80    = {PI80};
    newRow.PI95    = {PI95};
    newRow.hyp_hat = {hyp_hat};

    OUT = [OUT; newRow]; %#ok<AGROW>

    % ---- sanity plot
    figure('Name', char(C.name));
    plot(y_true,'k.-','DisplayName','HF true'); hold on;
    plot(mu_pred,'o-','DisplayName','pred mean');
    grid on; legend('Location','best');
    title(sprintf('%s | holdout %g | nTest=%d', C.name, hold_id, numel(y_true)));
    xlabel('test index'); ylabel('ws');
end

disp(' ');
disp('===== TRIAL SUMMARY (this station only) =====');
varsWanted = ["holdout_station","modelKey","case_name","RhoFunction","warped","NLML","RMSE","MAE","CORR", ...
              "PI80_Coverage","PI80_WidthMean","PI95_Coverage","PI95_WidthMean","PredVarMean"];
varsHave   = intersect(varsWanted, string(OUT.Properties.VariableNames), 'stable');
disp(OUT(:, varsHave));

%% ============================ HELPERS ============================

function data = localFindDataTableInWorkspace()
% Try to find a table in workspace that looks like your real dataset.
    candNames = ["data","sorted_data","sortedData","tbl","T","data_all","DATA"];
    data = [];

    for k = 1:numel(candNames)
        nm = candNames(k);
        if evalin('base', sprintf("exist('%s','var')", nm))
            v = evalin('base', nm);
            if istable(v)
                data = v;
                fprintf('Detected real-data table from workspace variable "%s".\n', nm);
                return;
            end
        end
    end

    % If not found by name, scan workspace for any table with required fields
    W = evalin('base','whos');
    for i = 1:numel(W)
        if strcmp(W(i).class,'table')
            v = evalin('base', W(i).name);
            if localLooksLikeRealDataTable(v)
                data = v;
                fprintf('Detected real-data table by structure: workspace variable "%s".\n', W(i).name);
                return;
            end
        end
    end

    % If still not found, error with helpful info
    tabVars = string({W(strcmp({W.class},'table')).name});
    error(['Could not find a real-data table in workspace.\n' ...
           'Tables currently in workspace: %s\n' ...
           'Either:\n' ...
           '  (1) load your data table into workspace, OR\n' ...
           '  (2) set dataFile="yourData.mat" at the top of the script.\n'], ...
           strjoin(tabVars, ", "));
end

function data = localFindDataTableInStruct(S)
% Find the first table in struct S that looks like the real dataset.
    fn = string(fieldnames(S));
    for i = 1:numel(fn)
        v = S.(fn(i));
        if istable(v) && localLooksLikeRealDataTable(v)
            data = v;
            fprintf('Detected real-data table from file variable "%s".\n', fn(i));
            return;
        end
    end

    % If none match, error with listing
    tableNames = fn(arrayfun(@(x) istable(S.(x)), fn));
    error(['Loaded file does not contain a suitable real-data table.\n' ...
           'Table variables in file: %s\n' ...
           'Make sure it includes required columns (Wind_speed, ws, Lat_LF, Lon_LF, Lat_HF, Lon_HF, Time, IDStation).'], ...
           strjoin(tableNames, ", "));
end

function tf = localLooksLikeRealDataTable(T)
    requiredVars = ["Wind_speed","ws","Lat_LF","Lon_LF","Lat_HF","Lon_HF","Time","IDStation"];
    tf = istable(T) && all(ismember(requiredVars, string(T.Properties.VariableNames)));
end

function tbl = cap_times_per_station(tbl, capN, seed)
% Keep at most capN rows per station by sampling evenly over sorted Time.
    if isempty(capN) || capN <= 0, return; end
    if height(tbl) == 0, return; end

    ids = unique(tbl.IDStation);
    keep = false(height(tbl),1);

    for i = 1:numel(ids)
        idx = find(tbl.IDStation == ids(i));
        if numel(idx) <= capN
            keep(idx) = true;
        else
            rng(seed + double(ids(i)));
            [~, ord] = sort(tbl.Time(idx));
            idx = idx(ord);
            pick = round(linspace(1, numel(idx), capN));
            keep(idx(pick)) = true;
        end
    end
    tbl = tbl(keep,:);
end

function hyp0 = getWarmStartHyp(C, hold_id, results, results_warp)
% Try to retrieve a fitted hyp vector from saved tables.
% Falls back to default initialization if not found.

    if C.warp
        r = results_warp(results_warp.holdout_station==hold_id, :);
        if ~isempty(r)
            MI = r.ModelInfo(1);
            if iscell(MI), MI = MI{1}; end
            if isstruct(MI) && isfield(MI,'hyp') && ~isempty(MI.hyp)
                hyp0 = MI.hyp;
                fprintf('Warm-start from results_warp.ModelInfo.hyp (len=%d)\n', numel(hyp0));
                return;
            end
        end
        hyp0 = default_hyp_init("constant");
        fprintf('Warm-start fallback: default constant hyp (len=%d)\n', numel(hyp0));
        return;
    end

    rf = string(C.RhoFunction);
    r = results(results.holdout_station==hold_id & string(results.RhoFunction)==rf, :);
    if ~isempty(r)
        MI = r.ModelInfo(1);
        if iscell(MI), MI = MI{1}; end
        if isstruct(MI) && isfield(MI,'hyp') && ~isempty(MI.hyp)
            hyp0 = MI.hyp;
            fprintf('Warm-start from results.ModelInfo.hyp (len=%d)\n', numel(hyp0));
            return;
        end
    end

    if rf == "constant"
        hyp0 = default_hyp_init("constant");
    else
        hyp0 = default_hyp_init("GP_scaled_empirical");
    end
    fprintf('Warm-start fallback: default hyp for %s (len=%d)\n', rf, numel(hyp0));
end

function [best_x, best_f] = run_multistart(obj, x0, opts)
% Multi-start optimization with warm-start + jittered restarts.
    n_starts = max(1, opts.n_starts);
    starts = cell(n_starts,1);
    starts{1} = x0;

    for k = 2:n_starts
        jitter = 0.10 * randn(size(x0));
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
% Base 11 + optional rho-GP params => 14.
    RF = string(RhoFunction);

    switch RF
        case "constant"
            hyp0 = zeros(11,1);
        otherwise
            hyp0 = zeros(14,1);
    end

    hyp0(1)  = log(1.0);  hyp0(2)  = log(0.20);
    hyp0(3)  = log(1.0);  hyp0(4)  = log(0.20);
    hyp0(5)  = 0.6;
    hyp0(6)  = log(0.10); hyp0(7)  = log(0.10);
    hyp0(8)  = log(1.0);  hyp0(9)  = log(1.0);
    hyp0(10) = log(1.0);  hyp0(11) = log(1.0);

    if numel(hyp0) >= 14
        hyp0(12) = log(0.5);
        hyp0(13) = log(1.0);
        hyp0(14) = log(1.0);
    end
end
