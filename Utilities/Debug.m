%% Debug_predict_calibratedCM3_1station.m
% Run line-by-line to debug why predictive variance is zero in predict_calibratedCM3.

%clear; clc; close all;
%{
%% ---------- USER SETTINGS ----------
hold_id = 100;
capN    = 100;

dataFile = "C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy\South_Lombardy_sorted_data.mat";

% pick ONE case to debug first:
CASE.key         = "MFGP_const";
CASE.name        = "rho constant";
CASE.RhoFunction = "constant";
CASE.warp        = false;

% Vecchia/model defaults
opts = struct();
opts.nn_size       = 50;
opts.cand_mult     = 10;
opts.conditioning  = "Corr";
opts.kernel        = "RBF";
opts.MeanFunction  = "zero";
opts.cov_type      = "RBF";
opts.combination   = "multiplicative";

% optimization
opts.max_iter = 30;      % keep small for debugging
opts.max_fun  = 300;
opts.n_starts = 1;
opts.seed     = 42;

% predictor opts (used by predict_calibratedCM3)
predOpts = struct();
predOpts.calib_mode   = "global_affine";
predOpts.gamma_clip   = [0.25, 4.0];
predOpts.lambda_ridge = 1e-8;
predOpts.gamma_subset = [];
predOpts.seed         = opts.seed;

predOpts.bin_Kt      = 4;
predOpts.bin_Ks      = 4;
predOpts.bin_min_pts = 15;

%% ---------- LOAD REAL DATA TABLE ----------
S = load(dataFile);
data = localFindDataTableInStruct(S);

requiredVars = ["Wind_speed","ws","Lat_LF","Lon_LF","Lat_HF","Lon_HF","Time","IDStation"];
missing = setdiff(requiredVars, string(data.Properties.VariableNames));
if ~isempty(missing)
    error("Missing required columns: %s", strjoin(missing,", "));
end
if ~isnumeric(data.Time),      data.Time      = double(data.Time); end
if ~isnumeric(data.IDStation), data.IDStation = double(data.IDStation); end

%% ---------- SPLIT LOSO ----------
is_hold   = (data.IDStation == hold_id);
train_tbl = data(~is_hold,:);
test_tbl  = data(is_hold,:);

if height(test_tbl)==0
    error("No rows for holdout station %g", hold_id);
end

train_tbl = cap_times_per_station(train_tbl, capN, opts.seed);
test_tbl  = cap_times_per_station(test_tbl,  capN, opts.seed);

X_L = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
y_L = double(train_tbl.Wind_speed);

X_H = [double(train_tbl.Time), double(train_tbl.Lat_HF), double(train_tbl.Lon_HF)];
y_H = double(train_tbl.ws);

Xstar  = [double(test_tbl.Time), double(test_tbl.Lat_HF), double(test_tbl.Lon_HF)];
y_true = double(test_tbl.ws);

fprintf("\nHoldout station %g | Train nL=%d nH=%d | Test=%d\n", ...
    hold_id, size(X_L,1), size(X_H,1), size(Xstar,1));

% stable ordering
[~, pL] = sortrows(X_L, [2 3 1]); X_L = X_L(pL,:); y_L = y_L(pL);
[~, pH] = sortrows(X_H, [2 3 1]); X_H = X_H(pH,:); y_H = y_H(pH);

%% ---------- BUILD ModelInfo (GLOBAL) ----------
global ModelInfo
ModelInfo = struct();
ModelInfo.X_L = X_L;
ModelInfo.X_H = X_H;
ModelInfo.cov_type     = opts.cov_type;
ModelInfo.combination  = opts.combination;
ModelInfo.RhoFunction  = CASE.RhoFunction;
ModelInfo.MeanFunction = opts.MeanFunction;

ModelInfo.jitter       = 1e-8;
ModelInfo.nn_size      = opts.nn_size;
ModelInfo.kernel       = opts.kernel;
ModelInfo.conditioning = opts.conditioning;
ModelInfo.cand_mult    = opts.cand_mult;
ModelInfo.show_path_diag = false;

if ~CASE.warp
    ModelInfo.y_L = y_L;
    ModelInfo.y_H = y_H;
else
    ModelInfo.y_L = log1p(max(y_L,0));
    ModelInfo.y_H = log1p(max(y_H,0));
end

%% ---------- INITIAL HYP ----------
% start simple (no warm-start) to avoid importing broken caches
hyp0 = default_hyp_init(CASE.RhoFunction);

fprintf("Initial hyp0 length = %d\n", numel(hyp0));

%% ---------- FIT (single start) ----------
obj = @(hh) likelihoodVecchia_nonstat_GLS(hh);

rng(opts.seed);
[hyp_hat, nlml_hat] = run_multistart(obj, hyp0, opts);

fprintf("After optimization: nlml=%.3f\n", nlml_hat);

% IMPORTANT: call likelihood again to populate debug_vecchia
ModelInfo.hyp = hyp_hat;
nlml_hat2 = likelihoodVecchia_nonstat_GLS(hyp_hat);
fprintf("Final likelihood call done: nlml=%.3f\n", nlml_hat2);

%% ---------- CHECK debug_vecchia EXISTS ----------
assert(isfield(ModelInfo,'debug_vecchia') && isstruct(ModelInfo.debug_vecchia), ...
    "ModelInfo.debug_vecchia missing. likelihood did not populate caches.");

V = ModelInfo.debug_vecchia;
needed = {'A','D_inv','R','perm'};
for k = 1:numel(needed)
    assert(isfield(V,needed{k}) && ~isempty(V.(needed{k})), ...
        "debug_vecchia.%s missing/empty", needed{k});
end
fprintf("debug_vecchia OK: A(%dx%d) D_inv(%dx%d)\n", size(V.A,1), size(V.A,2), size(V.D_inv,1), size(V.D_inv,2));

% ---------- CALL YOUR PREDICTOR ----------
[muH, s2H] = predict_calibratedCM3(Xstar, predOpts);

muH = muH(:);
s2H = s2H(:);

fprintf("\nPREDICT output: muH[%d], s2H[%d]\n", numel(muH), numel(s2H));
fprintf("s2H stats: min=%.4g max=%.4g mean=%.4g\n", min(s2H), max(s2H), mean(s2H));
%}
[s2H] = compute_MF_Vecchia_Variance(Xstar, rho_star, ModelInfo);
%% ---------- IF VARIANCE IS ~0: DECOMPOSE MANUALLY ----------
% This block replicates the key variance calculation in CM3
% to tell you where it collapses.

fprintf("\n--- MANUAL VARIANCE DIAGNOSTIC ---\n");

% shorthand
M = ModelInfo;
X_L = M.X_L; X_H = M.X_H;
y   = [M.y_L; M.y_H];
nL  = size(X_L,1); nH = size(X_H,1); N = nL+nH;

applyKinv = @(v) apply_Kinv_local(v, V.A, V.D_inv, V.R, V.perm);

alpha = applyKinv(y);

% GLS pieces (same as CM3)
Z = [ [ones(nL,1); zeros(nH,1)], [zeros(nL,1); ones(nH,1)] ];
KinvZ = [applyKinv(Z(:,1)), applyKinv(Z(:,2))];
m_GLS = (Z.'*KinvZ) \ (Z.'*alpha);
resid = alpha - KinvZ*m_GLS;

rho_star = compute_rho_star(Xstar, X_H, M.hyp, M.RhoFunction);
rho_star = rho_star(:);

fprintf("rho_star stats: min=%.4g max=%.4g mean=%.4g\n", ...
    min(rho_star), max(rho_star), mean(rho_star));

% Passing M as the first argument, [] for pick, and rho_star as the third
[qLstar, qHstar] = build_q_blocks_HF_rho(M, [], rho_star);
qstar = [qLstar, qHstar];

% prior diag
kss = prior_diag_kss_HF(rho_star, M);
kss = kss(:);

% reduction
Kinv_qT  = applyKinv(qstar');                % N x n*
reduction = sum(qstar' .* Kinv_qT, 1)';      % n* x 1

d = kss - reduction;
s2_0 = max(0, d);

fprintf("kss stats: min=%.4g max=%.4g mean=%.4g\n", min(kss), max(kss), mean(kss));
fprintf("reduction:  min=%.4g max=%.4g mean=%.4g\n", min(reduction), max(reduction), mean(reduction));
fprintf("kss-red:    min=%.4g max=%.4g mean=%.4g\n", min(d), max(d), mean(d));
fprintf("s2_0:       min=%.4g max=%.4g mean=%.4g\n", min(s2_0), max(s2_0), mean(s2_0));

% check for “tiny negative everywhere” => numerical cancellation
fprintf("Fraction(d<0) = %.2f\n", mean(d<0));

%% ---------- QUICK PATCH TESTS (DO NOT KEEP; just to diagnose) ----------
% (1) add HF noise to kss (if hyp(7) is log noise variance)
kss_plus = kss + exp(M.hyp(7));
d_plus   = kss_plus - reduction;
s2_plus  = max(0, d_plus);
fprintf("\nPatch test +exp(hyp(7)):\n");
fprintf("s2_plus mean=%.4g (was %.4g)\n", mean(s2_plus), mean(s2_0));

% (2) add small jitter to kss
kss_j = kss + 1e-6;
s2_j  = max(0, kss_j - reduction);
fprintf("Patch test +1e-6 jitter:\n");
fprintf("s2_j mean=%.4g\n", mean(s2_j));

%% ---------- BASIC METRICS ----------
rmse = sqrt(mean((muH - y_true).^2));
mae  = mean(abs(muH - y_true));
corrv = corr(muH, y_true);

fprintf("\nMEAN metrics (no PI): RMSE=%.4f MAE=%.4f CORR=%.4f\n", rmse, mae, corrv);

%% ---------- PLOT ----------
figure;
plot(y_true,'k.-','DisplayName','HF true'); hold on;
plot(muH,'o-','DisplayName','pred mean');
grid on; legend('Location','best');
title(sprintf('Debug trial | %s | holdout %g', CASE.name, hold_id));
xlabel('test index'); ylabel('ws');

%% ============================ LOCAL FUNCTIONS ============================

function data = localFindDataTableInStruct(S)
    fn = string(fieldnames(S));
    for i = 1:numel(fn)
        v = S.(fn(i));
        if istable(v)
            requiredVars = ["Wind_speed","ws","Lat_LF","Lon_LF","Lat_HF","Lon_HF","Time","IDStation"];
            if all(ismember(requiredVars, string(v.Properties.VariableNames)))
                data = v;
                fprintf('Detected data table: "%s"\n', fn(i));
                return;
            end
        end
    end
    error("No suitable data table found in the loaded .mat struct.");
end

function tbl = cap_times_per_station(tbl, capN, seed)
    if isempty(capN) || capN <= 0 || height(tbl)==0, return; end
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

function [best_x, best_f] = run_multistart(obj, x0, opts)
    best_x = x0;
    best_f = inf;
    use_fminunc = license('test','optimization_toolbox');

    if use_fminunc
        o = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton', ...
            'MaxIterations', opts.max_iter, 'MaxFunctionEvaluations', opts.max_fun);
        [xk,fk] = fminunc(obj, x0, o);
    else
        o = optimset('Display','iter','MaxIter',opts.max_iter,'MaxFunEvals',opts.max_fun);
        [xk,fk] = fminsearch(obj, x0, o);
    end

    if fk < best_f
        best_f = fk;
        best_x = xk;
    end
end

function hyp0 = default_hyp_init(RhoFunction)
    RF = string(RhoFunction);
    if RF == "constant"
        hyp0 = zeros(11,1);
    else
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

function x = apply_Kinv_local(v, A, Dinv, R, p)
    Dy   = Dinv * v;
    rhs  = A.' * Dy;
    rhsP = rhs(p, :);

    tmp  = R' \ rhsP;
    zP   = R  \ tmp;

    z = zeros(size(rhs), 'like', rhs);
    z(p,:) = zP;

    x = Dy - Dinv * (A * z);
end

function [qL_H, qH_H] = build_q_blocks_HF_rho(M, pick, rho_H)
% EXACTLY like your predictVecchia_CM_calibrated2 build_q_on_XH_rho
    X_L = M.X_L; X_H = M.X_H; hyp = M.hyp;
    [kt, ks] = pick_kernels(M.cov_type);
    comb = lower(string(M.combination));

    if nargin < 2 || isempty(pick)
        XH_t = X_H(:,1); XH_s = X_H(:,2:3);
        rho = rho_H(:);
    else
        XH_t = X_H(pick,1); XH_s = X_H(pick,2:3);
        rho = rho_H(:);
    end

    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    Kt_L  = kt(XH_t, X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
    Ks_L  = ks(XH_s, X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

    Kt_HL = kt(XH_t, X_H(:,1),   [s_sig_LF_t, t_ell_LF]);
    Ks_HL = ks(XH_s, X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);
    Kt_d  = kt(XH_t, X_H(:,1),   [s_sig_HF_t, t_ell_HF]);
    Ks_d  = ks(XH_s, X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);

    switch comb
        case 'additive'
            qL_base = (Kt_L  + Ks_L);
            qH_base = (Kt_HL + Ks_HL);
            qd_base = (Kt_d  + Ks_d);
        case 'multiplicative'
            qL_base = (Kt_L  .* Ks_L);
            qH_base = (Kt_HL .* Ks_HL);
            qd_base = (Kt_d  .* Ks_d);
        otherwise
            error('Invalid combination: %s', string(M.combination));
    end

    qL_H = qL_base .* rho;                  % nH x nL (or subset)
    qH_H = qH_base .* (rho.^2) + qd_base;   % nH x nH (or subset)
end

function kss = prior_diag_kss_HF(rho_star, M)
% Prior variance diagonal k_H(x*,x*) consistent with your kernels:
% - for k1/k_matern as implemented, diag = sigma (not sigma^2)
% - combination additive: k = kt + ks
% - combination multiplicative: k = kt .* ks  => diag = sigma_t * sigma_s
%
% HF: k_HH(x*,x*) = rho^2 * k_L(x*,x*) + k_d(x*,x*)

    hyp = M.hyp;
    comb = lower(string(M.combination));

    sig_LF_t = exp(hyp(1));
    sig_HF_t = exp(hyp(3));
    sig_LF_s = exp(hyp(8));
    sig_HF_s = exp(hyp(10));

    switch comb
        case 'additive'
            kL_ss = sig_LF_t + sig_LF_s;
            kd_ss = sig_HF_t + sig_HF_s;
        case 'multiplicative'
            kL_ss = sig_LF_t * sig_LF_s;
            kd_ss = sig_HF_t * sig_HF_s;
        otherwise
            error('Invalid combination: %s', string(M.combination));
    end

    rho = rho_star(:);
    kss = (rho.^2) .* kL_ss + kd_ss;
end

function [kt, ks] = pick_kernels(cov_type)
% SAME mapping as your base function
    switch string(cov_type)
        case {'RBF','RBF_separate_rho'}
            kt = @k1;       ks = @k1;
        case 'Matern'
            kt = @k_matern; ks = @k_matern;
        case 'Mix'
            kt = @k1;       ks = @k_matern;
        otherwise
            error('Unknown cov_type: %s', string(cov_type));
    end
end

