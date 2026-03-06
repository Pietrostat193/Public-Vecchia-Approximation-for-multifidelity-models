%% ============================================================
%  Sweep TWO noise levels sigma_d2 = [2,4] for ONE sim condition (is=12)
%  Compute Mean AND Std Dev for MAE / RMSE / MAPE for FIVE models
%  Models: GP1, GP2, GP3, Classic (Exact MFGP), Vecchia60
%% ============================================================
clear; clc;

% ---------------- USER SETTINGS ----------------
is        = 12;
R         = 100;
trainFrac = 0.3;

simCond   = make_sim_conditions();
cfg0      = simCond(is);
cfg0.n_time = 10;

rho_fixed   = 0.6;
sigma_list  = [2 4];
noise_names = ["sigma_d2=2","sigma_d2=4"];

% Exact / Vecchia optimizer settings
max_restarts = 5;
opt = optimoptions('fminunc','Algorithm','quasi-newton','Display','off');

% ---------------- METRICS ----------------
mae_fun  = @(yhat,y) mean(abs(yhat(:) - y(:)));
rmse_fun = @(yhat,y) sqrt(mean((yhat(:) - y(:)).^2));
mape_fun = @(yhat,y) mean(abs((yhat(:) - y(:)) ./ max(abs(y(:)),1e-12))) * 100;

% ---------------- OUTPUT TABLE ----------------
Rows = table();

% Model column names (used consistently everywhere)
modelCols = ["GP1","GP2","GP3","Classic","Vecchia60"];

for sidx = 1:numel(sigma_list)
    % --- configure this noise level ---
    cfg = cfg0;
    cfg.rho      = rho_fixed;
    cfg.sigma_d2 = sigma_list(sidx);

    ok_all = false(R,1);

    % Preallocate metric storage: R runs x 5 models
    MAE  = nan(R, numel(modelCols));
    RMSE = nan(R, numel(modelCols));
    MAPE = nan(R, numel(modelCols));

    % Optional: track failures
    nFail = 0;

    for r = 1:R
        try
            seed = 100000*is + r + 1000*sidx;
            rng(seed);

            out = simulate_data_dynamic(seed, trainFrac, cfg);

            % Test inputs / outputs
            X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
            y      = out.HF_test.fH(:);

            % --- sanity checks ---
            if ~isfield(out,'test_row_idx') || isempty(out.test_row_idx)
                error('out.test_row_idx is missing/empty. Required to align Y1/Y2/Y3 to HF_test.');
            end

            % ============================================================
            % 1) Classic Exact MFGP (global ModelInfo used by likelihood/predict)
            % ============================================================
            clear global ModelInfo; global ModelInfo;
            ModelInfo = struct( ...
                'X_H', [out.HF_train.t, out.HF_train.s1, out.HF_train.s2], ...
                'y_H', out.HF_train.fH(:), ...
                'X_L', [out.LF.t,       out.LF.s1,       out.LF.s2], ...
                'y_L', out.LF.fL(:), ...
                'cov_type', "RBF", ...
                'kernel', "RBF", ...
                'combination', "multiplicative", ...
                'jitter', 1e-6, ...
                'MeanFunction', "zero", ...
                'RhoFunction', "constant" ...
            );

            bestF  = inf;
            bestHyp = [];

            for att = 1:max_restarts
                hyp_init = rand(11,1);
                [htry, ftry] = fminunc(@likelihood2Dsp, hyp_init, opt);
                if isfinite(ftry) && ftry < bestF
                    bestF  = ftry;
                    bestHyp = htry;
                end
            end

            if isempty(bestHyp)
                error('Exact MFGP optimization failed: no finite objective found.');
            end

            ModelInfo.hyp = bestHyp(:);
            likelihood2Dsp(ModelInfo.hyp);      % ensure internal state set
            pred_classic = predict2Dsp(X_test); % Classic predictions on HF test

            % ============================================================
            % 2) GP predictors (GP1/GP2/GP3) via train_and_predict_gpr
            %    NOTE: we assume Y1/Y2/Y3 are vectors aligned to some master
            %    indexing where out.test_row_idx selects the HF test rows.
            % ============================================================
            ModelInfo2 = struct( ...
                'X_L', ModelInfo.X_L, 'y_L', ModelInfo.y_L, ...
                'X_H', ModelInfo.X_H, 'y_H', ModelInfo.y_H ...
            );

            [Y1, ~, Y2, ~, Y3, ~] = train_and_predict_gpr(ModelInfo2);

            % Extract the test-aligned predictions
            yhat_gp1 = Y1(out.test_row_idx);
            yhat_gp2 = Y2(out.test_row_idx);
            yhat_gp3 = Y3(out.test_row_idx);

            % Safety: ensure lengths match y
            if numel(yhat_gp1) ~= numel(y) || numel(yhat_gp2) ~= numel(y) || numel(yhat_gp3) ~= numel(y)
                error('Size mismatch: GP predictions selected by out.test_row_idx do not match HF_test length.');
            end

            % ============================================================
            % 3) Vecchia60 (nonstationary GLS likelihood + calibrated predictor)
            % ============================================================
            bestF_v   = inf;
            bestHyp_v = [];
             ModelInfo.conditioning = "Corr";
            ModelInfo.nn_size = 60;
            ModelInfo.cand_mult = max(10, 60);

            for att = 1:max_restarts
                hyp_init = rand(11,1);
                [htry, ftry] = fminunc(@likelihoodVecchia_nonstat_GLS, hyp_init, opt);
                if isfinite(ftry) && ftry < bestF_v
                    bestF_v   = ftry;
                    bestHyp_v = htry;
                end
            end

            if isempty(bestHyp_v)
                error('Vecchia optimization failed: no finite objective found.');
            end
            
            ModelInfo.hyp = bestHyp_v(:);
            likelihoodVecchia_nonstat_GLS(ModelInfo.hyp);      % ensure internal state set
            yhat_vecchia60 = predictVecchia_CM_calibrated2(X_test);

            if numel(yhat_vecchia60) ~= numel(y)
                error('Size mismatch: Vecchia60 predictions do not match HF_test length.');
            end

            % ============================================================
            % Store Metrics (5 models)
            % ============================================================
            yhats = {yhat_gp1, yhat_gp2, yhat_gp3, pred_classic(:), yhat_vecchia60(:)};

            for m = 1:numel(modelCols)
                MAE(r,m)  = mae_fun(yhats{m}, y);
                RMSE(r,m) = rmse_fun(yhats{m}, y);
                MAPE(r,m) = mape_fun(yhats{m}, y);
            end

            ok_all(r) = true;

        catch
            nFail = nFail + 1;
            continue;
        end
    end

    idx = ok_all;
    nOK = sum(idx);
    if nOK == 0
        fprintf('Noise %s: all runs failed (R=%d).\n', noise_names(sidx), R);
        continue;
    end

    % --- Calculate Mean and Std Dev across successful runs ---
    mu_MAE   = mean(MAE(idx,:),  1);  std_MAE  = std(MAE(idx,:),  0, 1);
    mu_RMSE  = mean(RMSE(idx,:), 1);  std_RMSE = std(RMSE(idx,:), 0, 1);
    mu_MAPE  = mean(MAPE(idx,:), 1);  std_MAPE = std(MAPE(idx,:), 0, 1);

    % --- Construct table block: 6 rows x 5 model columns ---
    block = table();
    block.NoiseLevel = repmat(noise_names(sidx), 6, 1);
    block.Metric     = ["MAE"; "MAE"; "RMSE"; "RMSE"; "MAPE"; "MAPE"];
    block.Stat       = ["Mean"; "Std"; "Mean"; "Std"; "Mean"; "Std"];

    vals = [mu_MAE; std_MAE; mu_RMSE; std_RMSE; mu_MAPE; std_MAPE];
    for j = 1:numel(modelCols)
        block.(modelCols(j)) = vals(:,j);
    end

    % Optionally add counts
    block.nOK   = repmat(nOK, 6, 1);
    block.nFail = repmat(nFail, 6, 1);

    Rows = [Rows; block];
end

disp('==================== PERFORMANCE TABLE (Mean & Std Dev) ====================');
disp(Rows);
