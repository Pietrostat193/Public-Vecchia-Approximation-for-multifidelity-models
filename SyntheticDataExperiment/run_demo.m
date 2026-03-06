clear; clc; rng(12345);

%% Data
seed = rng;                 % keep RNG state
out = simulate_data(seed, 0.8);  % 80% stations train

X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
y_test = out.HF_test.fH;

%% Optimizer options
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'SpecifyObjectiveGradient', false, ...
    'FiniteDifferenceType','central', ...
    'FiniteDifferenceStepSize',1e-4, ...
    'TypicalX', 1 + abs(rand(11,1)), ...
    'Display','iter', ...
    'MaxIterations', 200, ...
    'MaxFunctionEvaluations', 5000, ...
    'FunctionTolerance',1e-8, ...
    'StepTolerance',1e-8);

%% ModelInfo (global)
clear global ModelInfo
global ModelInfo
ModelInfo = struct();
ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
ModelInfo.y_H = out.HF_train.fH;
ModelInfo.X_L = [out.LF.t, out.LF.s1, out.LF.s2];
ModelInfo.y_L = out.LF.fL;
ModelInfo.cov_type    = "RBF";       % for baseline code compatibility
ModelInfo.kernel      = "RBF";
ModelInfo.combination = "multiplicative";
ModelInfo.jitter      = 1e-6;
ModelInfo.MeanFunction = "zero";
ModelInfo.RhoFunction  = "constant";

%% --- Baseline (exact) via likelihood2Dsp ---
hyp_init = rand(11,1);
options  = optimoptions(options, 'TypicalX', 1 + abs(hyp_init));

[hyp_base, fval_base ] = fminunc(@likelihood2Dsp, hyp_init, options);
ModelInfo.hyp = hyp_base;
base_nll = likelihood2Dsp(hyp_base);     % should match fval_base
p_base   = predict2Dsp(X_test);
rmse_base = sqrt(mean((p_base - y_test).^2));
fprintf('Baseline exact: NLML=%.6f, RMSE=%.6f\n', base_nll, rmse_base);

%% --- Vecchia sweeps with automatic restarts on failure ---
sizes = [10 25 40 50 90];
conds = ["MinMax","Corr"];

NLML_opt   = nan(numel(sizes), numel(conds));
RMSE_opt   = nan(numel(sizes), numel(conds));
L2_param   = nan(numel(sizes), numel(conds));
HYP_store  = cell(numel(sizes), numel(conds));

max_restarts = 5;                  % total attempts per (cond, nn)
exitflag_log = nan(numel(sizes), numel(conds)); 


for jc = 1:numel(conds)
    ModelInfo.conditioning = conds(jc);

    % warm start per-conditioning from the baseline
    hyp_seed_for_next = hyp_base;

    for i = 1:numel(sizes)
        nn = sizes(i);
        ModelInfo.nn_size   = nn;

        % For Corr, give a generous candidate pool; harmless for MinMax.
        ModelInfo.cand_mult = max(10, nn);

        % Ensure neighbor caches are rebuilt for this (cond, nn)
        if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
        if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

        % ---- Optimise with automatic random restarts on failure ----
        bestF = inf; bestHyp = []; bestExit = -999;
        attempts = 0;
        startHyp = hyp_seed_for_next;               % first attempt uses warm-start

        while attempts < max_restarts
            attempts = attempts + 1;
            try
                [htry, ftry, exitflag] = fminunc(@likelihoodVecchia_nonstat_GLS, startHyp, options);
                % Accept if converged or if ftry improved the best so far
                if isfinite(ftry) && (ftry < bestF || attempts == 1)
                    bestF = ftry; bestHyp = htry; bestExit = exitflag;
                end
                % success criteria: a positive exitflag is a good sign
                if isfinite(bestF) && bestExit > 0
                    break; % converged
                else
                    % regenerate a fresh random start and try again
                    startHyp = rand(11,1);
                end
            catch
                % regenerate a fresh random start and try again
                startHyp = rand(11,1);
            end
        end

        % If still empty (shouldn't happen), fall back to baseline
        if isempty(bestHyp)
            warning('All attempts failed for cond=%s, nn=%d. Using hyp_base.', string(conds(jc)), nn);
            bestHyp = hyp_base; bestF = likelihoodVecchia_nonstat_GLS(bestHyp);
        end

        % Record results
        NLML_opt(i, jc)  = bestF;
        HYP_store{i, jc} = bestHyp;
        exitflag_log(i,jc) = bestExit; 

        % refresh caches at final hyp before predicting
        ModelInfo.hyp = bestHyp;
        likelihoodVecchia_nonstat_GLS(bestHyp);

        % predict HF test
        try
            yhat = predictVecchia_CM_calibrated(X_test);
          
            RMSE_opt(i, jc) = sqrt(mean((yhat - y_test).^2));
        catch ME
            warning('Prediction failed (cond=%s, nn=%d): %s', conds(jc), nn, ME.message);
            RMSE_opt(i, jc) = NaN;
        end

        % parameter discrepancy from baseline
        L2_param(i, jc) = norm(bestHyp - hyp_base);

        % carry forward as warm start for next nn_size in this conditioning
        hyp_seed_for_next = bestHyp;

        fprintf('Cond=%s, nn=%d | NLML=%.6f | RMSE=%.6f | ||Δhyp||=%.4f | attempts=%d\n', ...
            conds(jc), nn, NLML_opt(i, jc), RMSE_opt(i, jc), L2_param(i, jc), attempts);
    end
end

%% --- Summary table (optional) ---
T = table();
[cc, rr] = ndgrid(conds, sizes);
T.Conditioning = cc(:);
T.nn_size      = rr(:);
T.NLML         = NLML_opt(:);
T.RMSE         = RMSE_opt(:);
T.ParamL2_fromBase = L2_param(:);
disp(T);

%% --- Plot 1: NLML vs nn_size with baseline line ---
figure('Name','NLML vs nn_size');
hold on; grid on; box on;
plot(sizes, NLML_opt(:,1), '-o', 'LineWidth', 1.5, 'DisplayName','MinMax');
plot(sizes, NLML_opt(:,2), '-s', 'LineWidth', 1.5, 'DisplayName','Corr');
yline(base_nll, '--', 'Exact baseline', 'LineWidth', 1.25);
xlabel('nn\_size'); ylabel('Negative Log-Marginal Likelihood (NLML)');
title('Vecchia NLML vs nn\_size (MinMax & Corr) with Exact Baseline');
legend('Location','best');

%% --- Plot 2: Hyperparameter discrepancy (L2) vs nn_size ---
figure('Name','Param discrepancy vs nn_size');
hold on; grid on; box on;
plot(sizes, L2_param(:,1), '-o', 'LineWidth', 1.5, 'DisplayName','MinMax');
plot(sizes, L2_param(:,2), '-s', 'LineWidth', 1.5, 'DisplayName','Corr');
xlabel('nn\_size'); ylabel('|| hyp - hyp\_base ||_2');
title('Hyperparameter discrepancy from exact baseline');
legend('Location','best');

%% (Optional) Plot 3: RMSE vs nn_size
figure('Name','RMSE vs nn_size');
hold on; grid on; box on;
plot(sizes, RMSE_opt(:,1), '-o', 'LineWidth', 1.5, 'DisplayName','MinMax');
plot(sizes, RMSE_opt(:,2), '-s', 'LineWidth', 1.5, 'DisplayName','Corr');
yline(rmse_base, '--', 'Exact baseline RMSE', 'LineWidth', 1.25);
xlabel('nn\_size'); ylabel('RMSE on HF test');
title('Test RMSE vs nn\_size');
legend('Location','best');



hyp_10_mm=HYP_store{1,1};
hyp_10_c=HYP_store{1,2};
hyp_50_c=HYP_store{5,2};



ModelInfo.nn_size=10;
ModelInfo.conditioning="MinMax";
likelihoodVecchia_nonstat_GLS(hyp_10_mm);
ModelInfo.hyp=hyp_10_mm;
p_v10mm=predictVecchia_CM_calibrated2(X_test);

ModelInfo.conditioning="Corr";
likelihoodVecchia_nonstat_GLS(hyp_10_c);
ModelInfo.hyp=hyp_10_c;
p_v10c=predictVecchia_CM_calibrated2(X_test);


ModelInfo.nn_size=50;
ModelInfo.conditioning="Corr";
likelihoodVecchia_nonstat_GLS(hyp_50_c);
ModelInfo.hyp=hyp_50_c;
p_v50c=predictVecchia_CM_calibrated2(X_test);






% --- Make vectors & x-axis ------------------------------------------------
% --- Ensure each series is a single column (avoid extra lines) -----------
% --- Inputs (make sure these exist) ---------------------------------------
% p_base:   n×1
% p_v10mm:  n×1
% p_v10c:   n×1
% p_v50c:   n×1


% Ensure column vectors
p_base  = p_base(:);
p_v10mm = p_v10mm(:);
p_v10c  = p_v10c(:);
p_v50c  = p_v50c(:);

% Use index on x-axis
N  = numel(p_base);
I  = 1:N;

% Colors
co = lines(3);          % three distinct colors for the three panels
base_col = [0 0 0];

% Helper for MAE
mae = @(a,b) mean(abs(a(:)-b(:)));

figure('Color','w','Position',[120 80 950 720]);
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

% -------- Panel 1: m=10, MinMax -----------------------------------------
nexttile; hold on; box on; grid on;
% shaded area (hidden from legend)
fill([I, fliplr(I)], [p_base.', fliplr(p_v10mm.')], co(1,:), ...
    'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
% lines
plot(p_base,  '-',  'LineWidth', 1.1, 'Color', base_col, 'DisplayName','Base');
plot(p_v10mm, '--', 'LineWidth', 1.0, 'Color', co(1,:), ...
    'DisplayName', sprintf('m=10 MinMax (MAE=%.3f)', mae(p_v10mm, p_base)));
ylabel('Prediction');
legend('Location','best','Box','off');
title('Base vs Vecchia: m=10, MinMax');

% -------- Panel 2: m=10, Corr -------------------------------------------
nexttile; hold on; box on; grid on;
fill([I, fliplr(I)], [p_base.', fliplr(p_v10c.')], co(2,:), ...
    'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(p_base, '-',  'LineWidth', 1.1, 'Color', base_col, 'DisplayName','Base');
plot(p_v10c, '-.', 'LineWidth', 1.0, 'Color', co(2,:), ...
    'DisplayName', sprintf('m=10 Corr (MAE=%.3f)', mae(p_v10c, p_base)));
ylabel('Prediction');
legend('Location','best','Box','off');
title('Base vs Vecchia: m=10, Corr');

% -------- Panel 3: m=50, Corr -------------------------------------------
nexttile; hold on; box on; grid on;
fill([I, fliplr(I)], [p_base.', fliplr(p_v50c.')], co(3,:), ...
    'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(p_base, '-', 'LineWidth', 1.1, 'Color', base_col, 'DisplayName','Base');
plot(p_v50c, ':', 'LineWidth', 1.0, 'Color', co(3,:), ...
    'DisplayName', sprintf('m=50 Corr (MAE=%.3f)', mae(p_v50c, p_base)));
ylabel('Prediction'); xlabel('Index');
legend('Location','best','Box','off');
title('Base vs Vecchia: m=50, Corr');

% Optional: tighten axes without manual limits
% axis tight

