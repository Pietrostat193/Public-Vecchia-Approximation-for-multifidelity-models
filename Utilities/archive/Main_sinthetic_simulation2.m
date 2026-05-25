%% ============================================================
%  Sweep TWO noise levels sigma_d2 = [2,4] for ONE sim condition (is=12)
%  Compute MAE / RMSE / MAPE for FIVE models:
%    GP1, GP2, GP3, Classic(Exact MFGP), Vecchia (nn_size=30)
%  Output: one MATLAB table with 2(noise) x 3(metrics) rows
%% ============================================================

clear; clc;

% ------------ USER SETTINGS ------------
is = 12;                  % sim condition index
R  = 5;                 % replications
trainFrac = 0.3;
doMetrics = true;

simCond = make_sim_conditions();
cfg0 = simCond(is);
cfg0.n_time = 10;

rho_fixed = 0.6;
sigma_list = [2 4];       % <--- requested
noise_names = ["sigma_d2=2","sigma_d2=4"];  % or ["Low","High"] if you prefer

% Vecchia fixed settings (requested nn_size=30)
vecchiaConds = "Corr";    % use your preferred conditioning
nn_size = 30;

% Exact MFGP optimizer settings
max_restarts_exact = 5;
opt = optimoptions('fminunc',...
    'Algorithm','quasi-newton',...
    'Display','off',...
    'MaxFunctionEvaluations', 3000,...
    'MaxIterations', 500);

% ------------ helper for metrics ------------
mae_fun  = @(yhat,y) mean(abs(yhat(:)-y(:)));
rmse_fun = @(yhat,y) sqrt(mean((yhat(:)-y(:)).^2));
mape_fun = @(yhat,y) mean(abs((yhat(:)-y(:)) ./ max(abs(y(:)),1e-12))) * 100; % percent

% ------------ output storage (final summary table rows) ------------
Rows = table();

for sidx = 1:numel(sigma_list)

    cfg = cfg0;
    cfg.rho = rho_fixed;
    cfg.sigma_d2 = sigma_list(sidx);

    fprintf('\n============================================================\n');
    fprintf('SIM CONDITION %d: %s\n', is, cfg.name);
    fprintf('R=%d | n_time=%d | trainFrac=%.2f | rho=%.2f | sigma_d2=%.2f | nn_size=%d | cond=%s\n', ...
        R, cfg.n_time, trainFrac, cfg.rho, cfg.sigma_d2, nn_size, string(vecchiaConds));
    fprintf('============================================================\n');

    % ---- per-rep metric storage (only for this noise level) ----
    ok_all = false(R,1);

    MAE = nan(R,5);  RMSE = nan(R,5);  MAPE = nan(R,5);
    % column order: [GP1 GP2 GP3 Classic Vecchia]

    for r = 1:R
        fprintf('-- Rep %d/%d --\n', r, R);

        try
            seed = 100000*is + r + 1000*sidx; % separate seeds per sigma_d2
            rng(seed);

            out = simulate_data_dynamic(seed, trainFrac, cfg);

            X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
            y_test = out.HF_test.fH(:);

            %% ---------- Build ModelInfo for Exact MFGP ----------
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

            %% ---------- (A) Fit Exact MFGP ----------
            bestF = inf; bestHyp = [];
            for att = 1:max_restarts_exact
                hyp_init = rand(11,1);
                opt2 = optimoptions(opt,'TypicalX',1+abs(hyp_init));
                try
                    [htry, ftry] = fminunc(@likelihood2Dsp, hyp_init, opt2);
                    if isfinite(ftry) && ftry < bestF
                        bestF = ftry; bestHyp = htry;
                    end
                catch
                end
            end
            if isempty(bestHyp)
                warning('Exact fit failed at rep=%d (sigma_d2=%.2f).', r, cfg.sigma_d2);
                continue;
            end

            hyp_base = bestHyp(:);
            ModelInfo.hyp = hyp_base;
            likelihood2Dsp(hyp_base); % populate internals

            pred_exact = predict2Dsp(X_test);

            %% ---------- (C) GP predictors ----------
            ModelInfo2 = struct();
            ModelInfo2.X_L = ModelInfo.X_L;  ModelInfo2.y_L = ModelInfo.y_L;
            ModelInfo2.X_H = ModelInfo.X_H;  ModelInfo2.y_H = ModelInfo.y_H;

            [Y_pred1_all, ~, Y_pred2_all, ~, Y_pred3_all, ~] = train_and_predict_gpr(ModelInfo2);
            Y_pred1 = Y_pred1_all(out.test_row_idx);
            Y_pred2 = Y_pred2_all(out.test_row_idx);
            Y_pred3 = Y_pred3_all(out.test_row_idx);

            %% ---------- (B) Vecchia at fixed hyp_base (nn_size=30) ----------
            ModelInfo.conditioning = vecchiaConds;
            ModelInfo.nn_size = nn_size;
            ModelInfo.cand_mult = max(10, nn_size);

            if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
            if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

            likelihoodVecchia_nonstat_GLS(hyp_base);
            yhat_v = predictVecchia_CM_calibrated2(X_test);

            % ---- metrics (GP1,GP2,GP3,Classic,Vecchia) ----
            y = y_test(:);

            yhat_list = {Y_pred1(:), Y_pred2(:), Y_pred3(:), pred_exact(:), yhat_v(:)};

            for m = 1:5
                yh = yhat_list{m};
                MAE(r,m)  = mae_fun(yh,y);
                RMSE(r,m) = rmse_fun(yh,y);
                MAPE(r,m) = mape_fun(yh,y);
            end

            ok_all(r) = true;

        catch ME
            warning('Rep %d failed (sigma_d2=%.2f): %s', r, cfg.sigma_d2, ME.message);
            continue;
        end
    end

    % ---- reduce to reps where all models succeeded ----
    idx = ok_all;
    nOK = sum(idx);

    if nOK == 0
        warning('No successful reps for sigma_d2=%.2f. Skipping summary.', cfg.sigma_d2);
        continue;
    end

    % Means across successful reps
    mu_MAE  = mean(MAE(idx,:), 1, 'omitnan');
    mu_RMSE = mean(RMSE(idx,:), 1, 'omitnan');
    mu_MAPE = mean(MAPE(idx,:), 1, 'omitnan');

    % Build 3 rows (MAE/RMSE/MAPE) for this noise level
    modelCols = ["GP1","GP2","GP3","Classic","Vecchia30"];

    block = table();
    block.SimID     = repmat(is, 3, 1);
    block.SimName   = repmat(string(cfg.name), 3, 1);
    block.n_time    = repmat(cfg.n_time, 3, 1);
    block.trainFrac = repmat(trainFrac, 3, 1);
    block.rho       = repmat(cfg.rho, 3, 1);
    block.sigma_d2  = repmat(cfg.sigma_d2, 3, 1);
    block.nn_size   = repmat(nn_size, 3, 1);
    block.n_success = repmat(nOK, 3, 1);

    block.NoiseLevel = repmat(noise_names(sidx), 3, 1);
    block.Metric = ["MAE"; "RMSE"; "MAPE"];

    vals = [mu_MAE; mu_RMSE; mu_MAPE]; % 3 x 5

    % Add model columns
    for j = 1:numel(modelCols)
        block.(modelCols(j)) = vals(:,j);
    end

    Rows = [Rows; block]; %#ok<AGROW>
end

%% ------------ FINAL TABLE (like your LaTeX layout but in MATLAB) ------------
PerfTbl = Rows;

% Optional: nice display ordering
PerfTbl = movevars(PerfTbl, ["NoiseLevel","Metric"], "After", "SimName");

disp(' ');
disp('==================== PERFORMANCE TABLE (means over successful reps) ====================');
disp(PerfTbl);

% Optional export:
% writetable(PerfTbl, sprintf('Perf_is%02d_rho%.2f_nn%d_R%d.csv', is, rho_fixed, nn_size, R));
