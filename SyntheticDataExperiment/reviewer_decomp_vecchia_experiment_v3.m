%% reviewer_decomp_vecchia_experiment_v4.m
% Comparison: 
%  1) Exact GP (likelihood2Dsp)
%  2) nlml_vecchia_fullMF with MinMax conditioning
%  3) nlml_vecchia_fullMF with Corr conditioning
%  4) (Optional) The original likelihoodVecchia_nonstat_GLS for sanity check

clear; clc; rng(12345);

%% -------------------- 0) Simulate Data --------------------
seed = rng;
out  = simulate_data(seed, 0.8);  
X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
y_test = out.HF_test.fH(:);

%% -------------------- 1) Build COMPLETE ModelInfo --------------------
% Including EVERY field required by likelihood2Dsp and your Vecchia functions
clear global ModelInfo
global ModelInfo
ModelInfo = struct();

% Data fields
ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
ModelInfo.y_H = out.HF_train.fH(:);
ModelInfo.X_L = [out.LF.t, out.LF.s1, out.LF.s2];
ModelInfo.y_L = out.LF.fL(:);

% Covariance & Kernel fields
ModelInfo.cov_type    = "RBF";      % <--- Fixed: Added missing recognized field
ModelInfo.kernel      = "RBF";
ModelInfo.combination = "multiplicative";
ModelInfo.jitter      = 1e-6;

% Logic Switches
ModelInfo.MeanFunction = "GP_res";   % Full Mean Function Logic
ModelInfo.RhoFunction  = "constant";
ModelInfo.usePermutation = true;
ModelInfo.show_path_diag = false;

% Vecchia specific defaults
ModelInfo.nn_size = 20;
ModelInfo.conditioning = "Corr";
ModelInfo.cand_mult = 50;

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
alpha_exact  = ModelInfo.alpha; % Exact K^{-1}(y-m)
logdet_exact = 2 * ModelInfo.log_det_classic;
y_joint      = [ModelInfo.y_L; ModelInfo.y_H];
p_base       = predict2Dsp(X_test);
rmse_base    = sqrt(mean((p_base(:) - y_test).^2));

fprintf('\nEXACT baseline computed: RMSE = %.6f\n', rmse_base);

%% -------------------- 3) Comparison Loop: nlml_vecchia_fullMF --------------------
sizes = [10 20 40 60];
conds = ["MinMax", "Corr"];
RES = table();
row = 0;

fprintf('\n=== Step 2: Evaluating nlml_vecchia_fullMF at fixed hyp ===\n');

for jc = 1:numel(conds)
    ModelInfo.conditioning = conds(jc);
    
    for i = 1:numel(sizes)
        nn = sizes(i);
        ModelInfo.nn_size = nn;
        ModelInfo.cand_mult = max(10, nn);
        
        % Ensure caches are cleared for fair comparison
        if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
        if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end
        
        % --- RUN YOUR SPECIFIC FUNCTION ---
        t_start = tic;
        % We call your specific nlml_vecchia_fullMF function
        nlml_v = nlml_vecchia_fullMF(hyp_base); 
        t_eval = toc(t_start);
        
        % --- Extract diagnostics from the call ---
        % Your function likely populates ModelInfo.SIy (K^-1 * y_tilde)
        alpha_vec = ModelInfo.SIy;
        y_tilde   = ModelInfo.y_tilde; 
        
        % Calculate errors relative to exact GP
        relErr_alpha = norm(alpha_vec - alpha_exact) / max(norm(alpha_exact), 1e-12);
        
        % Back out log-determinant (Valid for likelihood comparison)
        quad_vec = (y_tilde' * alpha_vec);
        N = numel(y_joint);
        logdet_vec = 2*(nlml_v - 0.5*quad_vec - 0.5*N*log(2*pi));
        relErr_logdet = abs(logdet_vec - logdet_exact) / max(abs(logdet_exact), 1e-12);
        
        % Test Prediction Mean (using the calibrated predictor)
        %yhat = predictVecchia_CM_calibrated(X_test);
        %rmse_v = sqrt(mean((yhat(:) - y_test).^2));
        
        % Store Results
        row = row + 1;
        RES.Method(row,1) = "nlml_vecchia_fullMF";
        RES.Conditioning(row,1) = conds(jc);
        RES.m(row,1) = nn;
        RES.NLML(row,1) = nlml_v;
        RES.relErr_alpha(row,1) = relErr_alpha;
        RES.relErr_logdet(row,1) = relErr_logdet;
        %RES.RMSE(row,1) = rmse_v;
        RES.Time(row,1) = t_eval;
        
        fprintf('Mode: %s | m: %d | alpha_Err: %.3e | RMSE: %.4f\n', ...
            conds(jc), nn, relErr_alpha);
    end
end

%% -------------------- 4) Final Results & Visualization --------------------
disp(' ');
disp('=== FINAL RESULTS TABLE ===');
disp(RES);

% Plotting the Approximation Accuracy
figure('Color','w','Position',[100 100 1000 400]);

subplot(1,2,1); hold on; grid on; box on;
for jc = 1:numel(conds)
    idx = RES.Conditioning == conds(jc);
    plot(RES.m(idx), RES.relErr_alpha(idx), '-o', 'LineWidth', 1.5, 'DisplayName', conds(jc));
end
set(gca, 'YScale', 'log');
xlabel('Neighbor size (m)'); ylabel('Rel. Error in \alpha = K^{-1}(y-m)');
title('Linear System Error'); legend('Location','best');

