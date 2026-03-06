%% reviewer_decomp_vecchia_experiment.m
% Targeted experiment requested by reviewer:
% Compare (i) Exact GP vs (ii) Decomposed Vecchia (your current approach)
% on a manageable dataset where exact inference is feasible.
%
% Outputs:
%  - rel error in alpha = K^{-1}y
%  - rel error in quad form y'K^{-1}y
%  - rel error in log|K|
%  - HF test RMSE
%  - (if predictive variances available) 95% PI coverage + variance error
%
% Uses ONLY existing functions:
%   simulate_data, likelihood2Dsp, predict2Dsp,
%   likelihoodVecchia_nonstat_GLS, predictVecchia_CM_calibrated
%
clear; clc; rng(12345);

%% -------------------- 0) Simulate data (small enough for exact) --------------------
seed = rng;
out  = simulate_data(seed, 0.8);  % 80% stations train (your simulator)

X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
y_test = out.HF_test.fH(:);

%% -------------------- 1) Build ModelInfo --------------------
clear global ModelInfo
global ModelInfo
ModelInfo = struct();

ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
ModelInfo.y_H = out.HF_train.fH(:);
ModelInfo.X_L = [out.LF.t, out.LF.s1, out.LF.s2];
ModelInfo.y_L = out.LF.fL(:);

ModelInfo.cov_type    = "RBF";
ModelInfo.kernel      = "RBF";
ModelInfo.combination = "multiplicative";
ModelInfo.jitter      = 1e-6;

ModelInfo.MeanFunction = "zero";
ModelInfo.RhoFunction  = "constant";

% Vecchia config (will be overwritten in loop)
ModelInfo.nn_size = 20;
ModelInfo.conditioning = "Corr";
ModelInfo.cand_mult = 50;

%% -------------------- 2) Fit "Exact GP" once (baseline hyp) --------------------
% You can also just set hyp_base = ... if you already have best hyp.
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'SpecifyObjectiveGradient', false, ...
    'FiniteDifferenceType','central', ...
    'FiniteDifferenceStepSize',1e-4, ...
    'Display','iter', ...
    'MaxIterations', 200, ...
    'MaxFunctionEvaluations', 5000, ...
    'FunctionTolerance',1e-8, ...
    'StepTolerance',1e-8);

hyp_init = rand(11,1);
options  = optimoptions(options, 'TypicalX', 1 + abs(hyp_init));

fprintf('\n=== Fitting exact model (likelihood2Dsp) ===\n');
[hyp_base, fval_base] = fminunc(@likelihood2Dsp, hyp_init, options);

% Evaluate once more to ensure ModelInfo fields are populated
ModelInfo.hyp = hyp_base;
base_nlml = likelihood2Dsp(hyp_base);

% Extract exact diagnostics from ModelInfo populated by likelihood2Dsp
% alpha_exact: exact K^{-1}y
alpha_exact = ModelInfo.alpha;
y_joint     = [ModelInfo.y_L; ModelInfo.y_H];

% Exact log|K|: likelihood2Dsp stores sum(log(diag(L))) as log_det_classic
% NOTE: In your likelihood2Dsp, ModelInfo.log_det_classic is sum(log(diag(L)))
% so log|K| = 2*sum(log(diag(L))).
logdet_exact = 2 * ModelInfo.log_det_classic;

quad_exact   = y_joint' * alpha_exact;

% Exact prediction mean
p_base = predict2Dsp(X_test);
p_base = p_base(:);
rmse_base = sqrt(mean((p_base - y_test).^2));

fprintf('\nEXACT baseline:\n');
fprintf('  NLML        = %.6f\n', base_nlml);
fprintf('  log|K|      = %.6f\n', logdet_exact);
fprintf('  y''K^{-1}y   = %.6f\n', quad_exact);
fprintf('  RMSE (HF)   = %.6f\n', rmse_base);

%% -------------------- 3) Vecchia sweep (Decomposed Vecchia) --------------------
sizes = [10 20 30 40 60];
conds = ["MinMax","Corr"];   % your two conditioning modes

% Storage
RES = table();
row = 0;

% (Optional) If you have variance-capable predictors, set these handles:
%   exact_pred_var_fun    = @(X) predict2Dsp_with_var(X);
%   vecchia_pred_var_fun  = @(X) predictVecchia_CM_calibrated_with_var(X);
% If you don't, leave as [] and coverage will be skipped.
exact_pred_var_fun   = [];  % <-- plug in if you have it
vecchia_pred_var_fun = [];  % <-- plug in if you have it

for jc = 1:numel(conds)
    ModelInfo.conditioning = conds(jc);

    % warm-start vecchia optimizer from exact solution
    hyp_seed = hyp_base;

    for i = 1:numel(sizes)
        nn = sizes(i);
        ModelInfo.nn_size   = nn;
        ModelInfo.cand_mult = max(10, nn);  % safe for Corr, harmless for MinMax

        % Ensure neighbor caches rebuilt for fair comparison across nn/cond
        if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
        if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

        % --- IMPORTANT for reviewer: evaluate approximations AT FIXED hyp (propagation test)
        % We use hyp_base to isolate approximation error, not refit error.
        hyp_use = hyp_base;

        % Run likelihood once (populates ModelInfo.debug_vecchia, SIy, etc.)
        nlml_v = likelihoodVecchia_nonstat_GLS(hyp_use);

        % Pull diagnostics from cache:
        % - alpha_vec = K^{-1}y, stored as ModelInfo.debug_vecchia.SIy (raw)
        % - logdet_vec should be reconstructed from log_det_* terms.
        %
        % Your likelihoodVecchia_nonstat_GLS stores:
        %  dbg = ModelInfo.debug_vecchia
        %  dbg.SIy = K^{-1}y (raw)
        dbg = ModelInfo.debug_vecchia;

        alpha_vec = dbg.SIy;

        % Reconstruct log|K| from pieces available inside likelihoodVecchia_nonstat_GLS:
        % It computed log_det_W, log_det_H, log_det_D (but in your posted code
        % those are local variables; they are NOT stored unless you add them).
        %
        % If you DID NOT patch likelihoodVecchia_nonstat_GLS to store them,
        % we can still get log|K| indirectly from NLML parts only if we also
        % know y'K^{-1}y and N, which we do, and we can back it out:
        %
        % NLML = 0.5*y'K^{-1}y + 0.5*log|K| + 0.5*N*log(2pi)
        %
        % BUT: your NLML uses y_tilde (mean-adjusted). In our setup MeanFunction="zero"
        % so y_tilde=y and this is valid. If MeanFunction changes, do NOT back out.
        N = numel(y_joint);
        quad_vec = y_joint' * alpha_vec;

        % Back out log|K| from NLML (valid only for zero mean):
        if ModelInfo.MeanFunction == "zero"
            logdet_vec = 2*(nlml_v - 0.5*quad_vec - 0.5*N*log(2*pi));
        else
            logdet_vec = NaN;
        end

        % Relative errors vs exact (reviewer focus)
        relErr_alpha = norm(alpha_vec - alpha_exact) / max(norm(alpha_exact), 1e-12);
        relErr_quad  = abs(quad_vec - quad_exact) / max(abs(quad_exact), 1e-12);
        relErr_logdet = abs(logdet_vec - logdet_exact) / max(abs(logdet_exact), 1e-12);

        % Prediction mean + RMSE
        yhat = predictVecchia_CM_calibrated2(X_test);
        yhat = yhat(:);
        rmse_v = sqrt(mean((yhat - y_test).^2));

        % Variance + coverage (only if you have variance predictors)
        cov95 = NaN; varRelErr = NaN; meanWidth95 = NaN;
        if ~isempty(exact_pred_var_fun) && ~isempty(vecchia_pred_var_fun)
            [mu_e, s2_e] = exact_pred_var_fun(X_test);
            [mu_v, s2_v] = vecchia_pred_var_fun(X_test);

            mu_e = mu_e(:); s2_e = max(s2_e(:), 1e-12);
            mu_v = mu_v(:); s2_v = max(s2_v(:), 1e-12);

            varRelErr = median(abs(s2_v - s2_e) ./ s2_e);

            lo = mu_v - 1.96*sqrt(s2_v);
            hi = mu_v + 1.96*sqrt(s2_v);
            cov95 = mean((y_test >= lo) & (y_test <= hi));
            meanWidth95 = mean(hi - lo);
        end

        row = row + 1;
        RES.Method(row,1) = "DecomposedVecchia";
        RES.Conditioning(row,1) = conds(jc);
        RES.m(row,1) = nn;

        RES.NLML(row,1) = nlml_v;
        RES.logdetK(row,1) = logdet_vec;
        RES.quad(row,1) = quad_vec;

        RES.relErr_alpha(row,1) = relErr_alpha;
        RES.relErr_quad(row,1)  = relErr_quad;
        RES.relErr_logdet(row,1)= relErr_logdet;

        RES.RMSE(row,1) = rmse_v;
        RES.cov95(row,1)= cov95;
        RES.varRelErr(row,1) = varRelErr;
        RES.meanWidth95(row,1)= meanWidth95;

        fprintf('Cond=%s m=%d | relErr(alpha)=%.3e relErr(logdet)=%.3e RMSE=%.4f\n', ...
            string(conds(jc)), nn, relErr_alpha, relErr_logdet, rmse_v);

        % warm start if you later decide to refit under vecchia
        hyp_seed = hyp_use;
    end
end

%% -------------------- 4) Print + plot --------------------
disp(' ');
disp('=== RESULTS (Exact vs Decomposed Vecchia at fixed hyp_base) ===');
disp(RES);

% Plot: logdet error vs m
figure('Color','w'); hold on; grid on; box on;
for jc=1:numel(conds)
    idx = RES.Conditioning==conds(jc);
    plot(RES.m(idx), RES.relErr_logdet(idx), '-o','LineWidth',1.5, 'DisplayName', string(conds(jc)));
end
xlabel('m (nn\_size)'); ylabel('Relative error in log|K|');
title('Decomposed Vecchia: log|K| error vs neighbor size');
legend('Location','best');

% Plot: alpha error vs m
figure('Color','w'); hold on; grid on; box on;
for jc=1:numel(conds)
    idx = RES.Conditioning==conds(jc);
    plot(RES.m(idx), RES.relErr_alpha(idx), '-o','LineWidth',1.5, 'DisplayName', string(conds(jc)));
end
xlabel('m (nn\_size)'); ylabel('Relative error in \alpha = K^{-1}y');
title('Decomposed Vecchia: K^{-1}y error vs neighbor size');
legend('Location','best');

% Plot: RMSE vs m + exact RMSE
figure('Color','w'); hold on; grid on; box on;
for jc=1:numel(conds)
    idx = RES.Conditioning==conds(jc);
    plot(RES.m(idx), RES.RMSE(idx), '-o','LineWidth',1.5, 'DisplayName', string(conds(jc)));
end
yline(rmse_base,'--','Exact RMSE','LineWidth',1.2);
xlabel('m (nn\_size)'); ylabel('RMSE (HF test)');
title('Predictive mean accuracy vs neighbor size');
legend('Location','best');

% Coverage plot if available
if any(~isnan(RES.cov95))
    figure('Color','w'); hold on; grid on; box on;
    for jc=1:numel(conds)
        idx = RES.Conditioning==conds(jc);
        plot(RES.m(idx), RES.cov95(idx), '-o','LineWidth',1.5, 'DisplayName', string(conds(jc)));
    end
    yline(0.95,':','Nominal 0.95','LineWidth',1.2);
    xlabel('m (nn\_size)'); ylabel('Empirical 95% PI coverage');
    title('Predictive uncertainty calibration vs neighbor size');
    legend('Location','best');
else
    fprintf('\nNOTE: Coverage/variance not computed (no variance-returning predictors were provided).\n');
    fprintf('      If you have a function that returns (mu,s2) for exact and/or Vecchia,\n');
    fprintf('      plug them into exact_pred_var_fun and vecchia_pred_var_fun near the top.\n');
end
