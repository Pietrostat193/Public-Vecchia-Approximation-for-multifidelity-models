%% reviewer_decomp_vecchia_experiment_v4_CORRonly.m
% Comparison at FIXED hyperparameters (propagation test):
%  1) Exact GP (likelihood2Dsp)  [baseline truth]
%  2) nlml_vecchia_fullMF        [your Vecchia core]
%  3) likelihoodVecchia_nonstat_GLS (legacy/reviewer sanity check)
%
% Conditioning: Corr ONLY (MinMax removed)

clear; clc; rng(12345);

%% -------------------- 0) Simulate Data --------------------
seed = rng;
out  = simulate_data(seed, 0.8);

X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
y_test = out.HF_test.fH(:);

%% -------------------- 1) Build COMPLETE ModelInfo --------------------
clear global ModelInfo
global ModelInfo
ModelInfo = struct();

% Data fields
ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
ModelInfo.y_H = out.HF_train.fH(:);
ModelInfo.X_L = [out.LF.t, out.LF.s1, out.LF.s2];
ModelInfo.y_L = out.LF.fL(:);

% Covariance & Kernel fields (as required by likelihood2Dsp / your setup)
ModelInfo.cov_type    = "RBF";
ModelInfo.kernel      = "RBF";
ModelInfo.combination = "multiplicative";
ModelInfo.jitter      = 1e-6;

% Logic switches (keep as in your experiment)
ModelInfo.MeanFunction    = "GP_res";
ModelInfo.RhoFunction     = "constant";
ModelInfo.usePermutation  = true;
ModelInfo.show_path_diag  = false;

% Vecchia defaults (Corr only)
ModelInfo.nn_size      = 20;
ModelInfo.conditioning = "Corr";   % <-- Corr ONLY
ModelInfo.cand_mult    = 50;

%% -------------------- 2) Fit "Exact GP" (Baseline) --------------------
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','iter', ...
    'MaxIterations', 100, ...
    'FunctionTolerance',1e-8);

% Initialize hyperparameters (assuming 18 for GP_res configuration)
hyp_init = 0.1 * ones(18,1);

fprintf('\n=== Step 1: Fitting exact model (likelihood2Dsp) ===\n');
[hyp_base, ~] = fminunc(@likelihood2Dsp, hyp_init, options);

% Evaluate once more to populate exact diagnostics in ModelInfo
ModelInfo.hyp = hyp_base;
base_nlml = likelihood2Dsp(hyp_base);

% Capture Exact Truth
alpha_exact  = ModelInfo.alpha;               % Exact K^{-1}(y-m)
logdet_exact = 2 * ModelInfo.log_det_classic; % Your code stores half-logdet
y_joint      = [ModelInfo.y_L; ModelInfo.y_H];

% Baseline prediction (optional)
p_base    = predict2Dsp(X_test);
rmse_base = sqrt(mean((p_base(:) - y_test).^2));
fprintf('\nEXACT baseline computed: RMSE = %.6f\n', rmse_base);

%% -------------------- 3) Comparison Loop (Corr only) --------------------
sizes = [10 20 40 60];
RES = table();
row = 0;

fprintf('\n=== Step 2: Evaluating Vecchia approximations at fixed hyp (Corr only) ===\n');

N = numel(y_joint);

for i = 1:numel(sizes)

    nn = sizes(i);

    % Corr settings
    ModelInfo.conditioning = "Corr";
    ModelInfo.nn_size      = nn;
    ModelInfo.cand_mult    = max(10, nn);  % needed for Corr

    % --- IMPORTANT: evaluate approximations AT FIXED hyp (propagation test)
    hyp_use = hyp_base;

    % ---------------------------------------------------------------------
    % (A) YOUR FUNCTION: nlml_vecchia_fullMF
    % ---------------------------------------------------------------------
    % Clear neighbor caches for fair rebuild
    if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
    if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

    t_start = tic;
    nlml_A  = nlml_vecchia_fullMF(hyp_use);
    t_A     = toc(t_start);

    % Diagnostics from nlml_vecchia_fullMF
    alpha_A = ModelInfo.SIy;       % approx K^{-1} * y_tilde
    logdetA = ModelInfo.log_det_K; % approx log|K|
    ytil_A  = ModelInfo.y_tilde;   % vector actually used inside (for quad)

    relErr_alpha_A = norm(alpha_A - alpha_exact) / max(norm(alpha_exact), 1e-12);
    relErr_logdet_A = abs(logdetA - logdet_exact) / max(abs(logdet_exact), 1e-12);

    row = row + 1;
    RES.Method(row,1)        = "nlml_vecchia_fullMF";
    RES.m(row,1)             = nn;
    RES.NLML(row,1)          = nlml_A;
    RES.relErr_alpha(row,1)  = relErr_alpha_A;
    RES.relErr_logdet(row,1) = relErr_logdet_A;
    RES.Time(row,1)          = t_A;

    % ---------------------------------------------------------------------
    % (B) LEGACY/REVIEWER: likelihoodVecchia_nonstat_GLS
    % ---------------------------------------------------------------------
    % Clear caches again so legacy gets comparable rebuild
    if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
    if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end
    if isfield(ModelInfo,'debug_vecchia'), ModelInfo = rmfield(ModelInfo,'debug_vecchia'); end

    t_start = tic;
    nlml_B  = likelihoodVecchia_nonstat_GLS(hyp_use);
    t_B     = toc(t_start);

    dbg = ModelInfo.debug_vecchia;

    % alpha from legacy
    alpha_B = dbg.SIy;
    relErr_alpha_B = norm(alpha_B - alpha_exact) / max(norm(alpha_exact), 1e-12);

    % logdet from legacy:
    % Preferred: if likelihoodVecchia_nonstat_GLS stores dbg.logdetK and dbg.y_tilde, use them.
    % Fallback: back out logdet from NLML identity ONLY if the quadratic uses y_joint.
    logdetB = NaN;
    relErr_logdet_B = NaN;

    if isfield(dbg,'logdetK')
        logdetB = dbg.logdetK;
        relErr_logdet_B = abs(logdetB - logdet_exact) / max(abs(logdet_exact), 1e-12);
    else
        % Fallback (only safe if mean adjustment is zero so y_used == y_joint)
        quad_B = y_joint' * alpha_B;
        logdetB = 2*(nlml_B - 0.5*quad_B - 0.5*N*log(2*pi));
        relErr_logdet_B = abs(logdetB - logdet_exact) / max(abs(logdet_exact), 1e-12);
    end

    row = row + 1;
    RES.Method(row,1)        = "likelihoodVecchia_nonstat_GLS";
    RES.m(row,1)             = nn;
    RES.NLML(row,1)          = nlml_B;
    RES.relErr_alpha(row,1)  = relErr_alpha_B;
    RES.relErr_logdet(row,1) = relErr_logdet_B;
    RES.Time(row,1)          = t_B;

    fprintf('m: %d | alphaErr new %.3e | legacy %.3e | time new %.3fs legacy %.3fs\n', ...
        nn, relErr_alpha_A, relErr_alpha_B, t_A, t_B);
end

%% -------------------- 4) Final Results & Visualization --------------------
disp(' ');
disp('=== FINAL RESULTS TABLE ===');
disp(RES);

% Optional: likelihood gap vs exact (helps reviewers)
RES.NLML_gap = RES.NLML - base_nlml;

figure('Color','w','Position',[100 100 1200 420]);

% (1) Alpha error
subplot(1,3,1); hold on; grid on; box on;
meths = unique(RES.Method);
for k = 1:numel(meths)
    idx = RES.Method == meths(k);
    plot(RES.m(idx), RES.relErr_alpha(idx), '-o', 'LineWidth', 1.8, 'DisplayName', meths(k));
end
set(gca,'YScale','log');
xlabel('Neighbor size (m)');
ylabel('Relative Error in \alpha');
title('\alpha error (Corr)');
legend('Location','best');

% (2) Logdet error
subplot(1,3,2); hold on; grid on; box on;
for k = 1:numel(meths)
    idx = RES.Method == meths(k);
    plot(RES.m(idx), RES.relErr_logdet(idx), '-o', 'LineWidth', 1.8, 'DisplayName', meths(k));
end
set(gca,'YScale','log');
xlabel('Neighbor size (m)');
ylabel('Relative Error in log|K|');
title('logdet error (Corr)');
legend('Location','best');

% (3) Runtime
subplot(1,3,3); hold on; grid on; box on;
for k = 1:numel(meths)
    idx = RES.Method == meths(k);
    plot(RES.m(idx), RES.Time(idx), '-o', 'LineWidth', 1.8, 'DisplayName', meths(k));
end
set(gca,'YScale','log');
xlabel('Neighbor size (m)');
ylabel('Time (seconds)');
title('Runtime (Corr)');
legend('Location','best');

%% -------------------- 5) Notes for correctness --------------------
% 1) The fallback reconstruction of logdet for legacy likelihood assumes:
%    NLML = 0.5*y'K^{-1}y + 0.5*log|K| + 0.5*N*log(2*pi)
%    using y_joint. If likelihoodVecchia_nonstat_GLS uses mean-adjusted y_tilde,
%    patch it to store dbg.y_tilde and dbg.logdetK for an exact comparison.
%
% Suggested patch inside likelihoodVecchia_nonstat_GLS:
%    dbg = ModelInfo.debug_vecchia;
%    dbg.logdetK = log_det_K;   % exact log|K| computed internally
%    dbg.y_tilde = y_tilde;     % vector used in quadratic form
%    ModelInfo.debug_vecchia = dbg;