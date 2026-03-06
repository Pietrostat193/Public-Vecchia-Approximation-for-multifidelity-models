%% reviewer_decomp_vecchia_experiment_v3.m
% Comparison: (1) Exact GP vs (2) nlml_vecchia_fullMF (MinMax) vs (3) nlml_vecchia_fullMF (Corr)
clear; clc; rng(12345);

%% -------------------- 0) Simulate Data --------------------
seed = rng;
out  = simulate_data(seed, 0.8);  
X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
y_test = out.HF_test.fH(:);

%% -------------------- 1) Model Configuration --------------------
global ModelInfo
ModelInfo = struct();
ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
ModelInfo.y_H = out.HF_train.fH(:);
ModelInfo.X_L = [out.LF.t, out.LF.s1, out.LF.s2];
ModelInfo.y_L = out.LF.fL(:);
ModelInfo.kernel      = "RBF";
ModelInfo.jitter      = 1e-6;
ModelInfo.MeanFunction = "GP_res"; % Using the complex mean structure
ModelInfo.RhoFunction  = "constant";

%% -------------------- 2) Baseline: Exact GP --------------------
% We fit the exact model to get the "ground truth" hyperparameters
fprintf('\n=== Step 1: Fitting Exact Baseline (likelihood2Dsp) ===\n');
hyp_init = 0.1 * ones(18,1); 
options = optimoptions('fminunc','Display','none','MaxIterations',100);
[hyp_base, ~] = fminunc(@likelihood2Dsp, hyp_init, options);

% Extract exact values
ModelInfo.hyp = hyp_base;
base_nlml    = likelihood2Dsp(hyp_base);
alpha_exact  = ModelInfo.alpha; % Populated by likelihood2Dsp
logdet_exact = 2 * ModelInfo.log_det_classic;
y_joint      = [ModelInfo.y_L; ModelInfo.y_H];
p_base       = predict2Dsp(X_test);
rmse_base    = sqrt(mean((p_base(:) - y_test).^2));

fprintf('Exact Baseline Computed. RMSE: %.4f\n', rmse_base);

%% -------------------- 3) The Comparison Loop --------------------
sizes = [10 20 40 60];
conds = ["MinMax", "Corr"];
RES = table();
row = 0;

fprintf('\n=== Step 2: Evaluating nlml_vecchia_fullMF across variations ===\n');

for jc = 1:numel(conds)
    ModelInfo.conditioning = conds(jc);
    
    for i = 1:numel(sizes)
        nn = sizes(i);
        ModelInfo.nn_size = nn;
        ModelInfo.cand_mult = max(10, nn);
        
        % Force rebuild of neighbor cache for this specific nn/conditioning
        if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
        if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end
        
        % --- CALL YOUR SPECIFIC FUNCTION ---
        % We evaluate at hyp_base to see how well the approximation 
        % matches the exact math for the same parameters.
        t_start = tic;
        nlml_v = nlml_vecchia_fullMF(hyp_base);
        t_eval = toc(t_start);
        
        % --- Diagnostics from your nlml_vecchia_fullMF internal state ---
        % We pull alpha (K^-1 * y) from ModelInfo.SIy which your function populates
        alpha_vec = ModelInfo.SIy;
        
        % Calculate Relative Errors
        relErr_alpha = norm(alpha_vec - alpha_exact) / max(norm(alpha_exact), 1e-12);
        
        % Back out log-determinant error (using the zero-mean equivalent for the metric)
        % NLML = 0.5*quad + 0.5*logdet + const.
        quad_vec = (ModelInfo.y_tilde' * alpha_vec);
        N = numel(y_joint);
        logdet_vec = 2*(nlml_v - 0.5*quad_vec - 0.5*N*log(2*pi));
        relErr_logdet = abs(logdet_vec - logdet_exact) / max(abs(logdet_exact), 1e-12);
        
        % Prediction Mean Accuracy
        yhat = predictVecchia_CM_calibrated2(X_test);
        rmse_v = sqrt(mean((yhat(:) - y_test).^2));
        
        % Store Results
        row = row + 1;
        RES.Conditioning(row,1) = conds(jc);
        RES.m(row,1) = nn;
        RES.NLML(row,1) = nlml_v;
        RES.relErr_alpha(row,1) = relErr_alpha;
        RES.relErr_logdet(row,1) = relErr_logdet;
        RES.RMSE(row,1) = rmse_v;
        RES.Time(row,1) = t_eval;
        
        fprintf('Mode: %s | m: %d | Alpha Err: %.2e | logdet Err: %.2e | RMSE: %.4f\n', ...
            conds(jc), nn, relErr_alpha, relErr_logdet, rmse_v);
    end
end

%% -------------------- 4) Reporting --------------------
fprintf('\n=== FINAL COMPARISON TABLE ===\n');
disp(RES);

% Visualization of Approximation Convergence
figure('Color','w','Name','Approximation Error vs Exact GP');
subplot(1,2,1); hold on; grid on;
for jc = 1:numel(conds)
    idx = RES.Conditioning == conds(jc);
    plot(RES.m(idx), RES.relErr_alpha(idx), '-o', 'LineWidth', 1.5, 'DisplayName', conds(jc));
end
set(gca, 'YScale', 'log');
xlabel('Neighbor Size (m)'); ylabel('Rel. Error in K^{-1}y');
title('Linear System Accuracy'); legend;

subplot(1,2,2); hold on; grid on;
for jc = 1:numel(conds)
    idx = RES.Conditioning == conds(jc);
    plot(RES.m(idx), RES.relErr_logdet(idx), '-o', 'LineWidth', 1.5, 'DisplayName', conds(jc));
end
set(gca, 'YScale', 'log');
xlabel('Neighbor Size (m)'); ylabel('Rel. Error in log|K|');
title('Log-Determinant Accuracy'); legend;

% RMSE Comparison
figure('Color','w','Name','Predictive Performance');
hold on; grid on;
for jc = 1:numel(conds)
    idx = RES.Conditioning == conds(jc);
    plot(RES.m(idx), RES.RMSE(idx), '-s', 'LineWidth', 1.5, 'DisplayName', ['Vecchia ('+conds(jc)+')']);
end
yline(rmse_base, '--r', 'Exact GP RMSE', 'LineWidth', 1.2);
xlabel('Neighbor Size (m)'); ylabel('Test RMSE');
title('RMSE: Exact GP vs. nlml\_vecchia\_fullMF variations');
legend;