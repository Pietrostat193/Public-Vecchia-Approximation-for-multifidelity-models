%% ============================================================
%  Sweep TWO noise levels sigma_d2 = [2,4] for ONE sim condition (is=12)
%  Compute Mean AND Std Dev for MAE / RMSE / MAPE + COVERAGE (90%,95%)
%  for FIVE models:
%    GP1, GP2, GP3, Classic (Exact Dense GLSmean), Vecchia (v4 REML)
%
%  Ordering: TIME-MAJOR for LF/HF training (as you chose)
%  Vecchia: nn_size = 40 (change nn_use if desired)
%
%  Coverage:
%    computed only for models that provide (mu, s2).
%    - Classic uses predict2Dsp_GLSmean (if available) -> coverage computed
%    - Vecchia uses predict_calibratedCM3_fixed or predictVecchia_calibratedCM3_fixed -> coverage computed
%    - GP1/GP2/GP3: mean only here -> coverage left as NaN unless you extend train_and_predict_gpr
%% ============================================================

clear; clc;

% ---------------- USER SETTINGS ----------------
is        = 12;
R         = 100;       % increase for real experiments
trainFrac = 0.3;

simCond   = make_sim_conditions();
cfg0      = simCond(is);
cfg0.n_time = 10;

rho_fixed   = 0.6;
sigma_list  = [2 4];
noise_names = ["sigma_d2=2","sigma_d2=4"];

% Exact / Vecchia optimizer settings
max_restarts = 5;
opt = optimoptions('fminunc','Algorithm','quasi-newton','Display','iter');

% ---------------- ORDERING CHOICE ----------------
orderingName = "time-major";   % {"time-major","space-major","random","time-causal-randspace"}
nn_use = 40;

% ---------------- METRICS ----------------
mae_fun  = @(yhat,y) mean(abs(yhat(:) - y(:)));
rmse_fun = @(yhat,y) sqrt(mean((yhat(:) - y(:)).^2));
mape_fun = @(yhat,y) mean(abs((yhat(:) - y(:)) ./ max(abs(y(:)),1e-12))) * 100;

% ---------------- COVERAGE ----------------
cov_fun = @(mu,s2,y,alpha) mean( abs(y(:) - mu(:)) <= sqrt(max(s2(:),0)) * norminv(1-alpha/2) );
alpha90 = 0.10;
alpha95 = 0.05;

% ---------------- OUTPUT TABLE ----------------
Rows = table();

% Model column names (used consistently everywhere)
modelCols = ["GP1","GP2","GP3","Classic","Vecchia_v4"];

for sidx = 1:numel(sigma_list)

    % --- configure this noise level ---
    cfg = cfg0;
    cfg.rho      = rho_fixed;
    cfg.sigma_d2 = sigma_list(sidx);

    ok_all = false(R,1);

    % Preallocate metric storage: R runs x 5 models
    MAE   = nan(R, numel(modelCols));
    RMSE  = nan(R, numel(modelCols));
    MAPE  = nan(R, numel(modelCols));
    COV90 = nan(R, numel(modelCols));
    COV95 = nan(R, numel(modelCols));

    nFail = 0;

    for r = 1:R
        try
            seed = 100000*is + r + 1000*sidx;
            rng(seed);

            out = simulate_data_dynamic(seed, trainFrac, cfg);

            % Test inputs / outputs
            X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
            y      = out.HF_test.fH(:);

            if ~isfield(out,'test_row_idx') || isempty(out.test_row_idx)
                error('out.test_row_idx is missing/empty. Required to align Y1/Y2/Y3 to HF_test.');
            end

            % ============================================================
            % 0) Apply ordering to TRAINING sets (LF + HF_train)
            % ============================================================
            X_L_raw = [out.LF.t, out.LF.s1, out.LF.s2];
            y_L_raw = out.LF.fL(:);

            X_H_raw = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
            y_H_raw = out.HF_train.fH(:);

            [X_L, y_L] = apply_ordering(X_L_raw, y_L_raw, orderingName, 111);
            [X_H, y_H] = apply_ordering(X_H_raw, y_H_raw, orderingName, 222);

            % ============================================================
            % 1) Classic Dense MF GP with GLS mean adjustment (Exact)
            % ============================================================
            clear global ModelInfo; global ModelInfo;
            ModelInfo = struct( ...
                'X_H', X_H, ...
                'y_H', y_H, ...
                'X_L', X_L, ...
                'y_L', y_L, ...
                'cov_type', "RBF", ...
                'kernel', "RBF", ...
                'combination', "multiplicative", ...
                'jitter', 1e-6, ...
                'MeanFunction', "zero", ...
                'RhoFunction', "constant" ...
            );

            bestF   = inf;
            bestHyp = [];

                hyp_init = rand(11,1);
                [htry, ftry] = fminunc(@likelihood2Dsp_GLSmean, hyp_init, opt);
                    bestF   = ftry;
                    bestHyp = htry;
            
            if isempty(bestHyp)
                error('Classic optimization failed: no finite objective found.');
            end

            ModelInfo.hyp = bestHyp(:);
            likelihood2Dsp_GLSmean(ModelInfo.hyp);  % ensure caches are consistent with GLSmean

            % Classic predictions: prefer (mu,s2) for coverage if available
            if exist('predict2Dsp_GLSmean','file') == 2
                [mu_classic, s2_classic] = predict2Dsp_GLSmean(X_test, true); % include obs noise for coverage
                pred_classic = mu_classic;
            else
                % fallback (mean only)
                pred_classic = predict2Dsp(X_test);
                mu_classic = pred_classic(:);
                s2_classic = nan(size(mu_classic));
            end

            % ============================================================
            % 2) GP predictors (GP1/GP2/GP3) via train_and_predict_gpr
            % ============================================================
            ModelInfo2 = struct('X_L', X_L, 'y_L', y_L, 'X_H', X_H, 'y_H', y_H);
            [Y1, ~, Y2, ~, Y3, ~] = train_and_predict_gpr(ModelInfo2);

            yhat_gp1 = Y1(out.test_row_idx);
            yhat_gp2 = Y2(out.test_row_idx);
            yhat_gp3 = Y3(out.test_row_idx);

            if numel(yhat_gp1) ~= numel(y) || numel(yhat_gp2) ~= numel(y) || numel(yhat_gp3) ~= numel(y)
                error('Size mismatch: GP predictions selected by out.test_row_idx do not match HF_test length.');
            end

            % ============================================================
            % 3) Vecchia MF GP (v4 REML GLS) with precomputed neighbors
            % ============================================================
            bestF_v   = inf;
            bestHyp_v = [];

            ModelInfo.conditioning = "Corr";
            ModelInfo.nn_size      = nn_use;
            ModelInfo.cand_mult    = max(10, nn_use);

            if ~isfield(ModelInfo,'GLSType')
                ModelInfo.GLSType = "constant";
            end

            % Precompute indices ONCE (ordering-dependent!)
            kernelName = ModelInfo.kernel;
            [idxL, idxH] = precompute_vecchia_indices_for_ordering(ModelInfo.X_L, ModelInfo.X_H, ModelInfo.nn_size, kernelName);
            ModelInfo.idxL_precomputed = idxL;
            ModelInfo.idxH_precomputed = idxH;
            ModelInfo.nn_size=40;
            ModelInfo.jitter=1e-06;
            % optional: fixed permutation caching (helps fminunc)
            ModelInfo.perm_fixed = [];

           
                hyp_init = rand(11,1);
                [htry_v, ftry_v] = fminunc(@likelihoodVecchia_nonstat_GLS_v4, hyp_init, opt);
                    bestF_v   = ftry_v;
                    bestHyp_v = htry_v;
            

            if isempty(bestHyp_v)
                error('Vecchia v4 optimization failed: no finite objective found.');
            end

            ModelInfo.hyp = bestHyp_v(:);
            likelihoodVecchia_nonstat_GLS_v4(ModelInfo.hyp);  % ensure internal state set

            % Vecchia predictions: prefer (mu,s2) for coverage
            if exist('predict_calibratedCM3_fixed','file') == 2
                [mu_vecchia, s2_vecchia] = predict_calibratedCM3_fixed(X_test, ModelInfo);
            elseif exist('predictVecchia_calibratedCM3_fixed','file') == 2
                [mu_vecchia, s2_vecchia] = predictVecchia_calibratedCM3_fixed(X_test, ModelInfo);
            else
                % fallback mean-only
                if exist('predictVecchia_CM_calibrated2','file') == 2
                    mu_vecchia = predictVecchia_CM_calibrated2(X_test);
                else
                    error('No Vecchia predictor found (need predict_calibratedCM3_fixed or equivalent).');
                end
                s2_vecchia = nan(size(mu_vecchia));
            end

            if numel(mu_vecchia) ~= numel(y)
                error('Size mismatch: Vecchia predictions do not match HF_test length.');
            end

            % ============================================================
            % Store Metrics (5 models)
            % ============================================================
            yhats = {yhat_gp1, yhat_gp2, yhat_gp3, pred_classic(:), mu_vecchia(:)};

            for m = 1:numel(modelCols)
                MAE(r,m)  = mae_fun(yhats{m}, y);
                RMSE(r,m) = rmse_fun(yhats{m}, y);
                MAPE(r,m) = mape_fun(yhats{m}, y);
            end

            % Coverage only where variance exists
            % GP1/GP2/GP3: left NaN (no variance here)
            % Classic
            COV90(r, modelCols=="Classic") = cov_fun(mu_classic, s2_classic, y, alpha90);
            COV95(r, modelCols=="Classic") = cov_fun(mu_classic, s2_classic, y, alpha95);

            % Vecchia
            COV90(r, modelCols=="Vecchia_v4") = cov_fun(mu_vecchia, s2_vecchia, y, alpha90);
            COV95(r, modelCols=="Vecchia_v4") = cov_fun(mu_vecchia, s2_vecchia, y, alpha95);

            ok_all(r) = true;

        catch ME
            nFail = nFail + 1;
            fprintf('\n[FAIL] noise=%s r=%d: %s\n', noise_names(sidx), r, ME.message);
            if ~isempty(ME.stack)
                fprintf('  at %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
            end
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

    mu_COV90  = mean(COV90(idx,:), 1); std_COV90 = std(COV90(idx,:), 0, 1);
    mu_COV95  = mean(COV95(idx,:), 1); std_COV95 = std(COV95(idx,:), 0, 1);

    % --- Construct table block: 10 rows x 5 model columns ---
    block = table();
    block.NoiseLevel = repmat(noise_names(sidx), 10, 1);
    block.Metric     = ["MAE"; "MAE"; "RMSE"; "RMSE"; "MAPE"; "MAPE"; "COV90"; "COV90"; "COV95"; "COV95"];
    block.Stat       = ["Mean"; "Std"; "Mean"; "Std"; "Mean"; "Std"; "Mean"; "Std"; "Mean"; "Std"];

    vals = [mu_MAE; std_MAE; mu_RMSE; std_RMSE; mu_MAPE; std_MAPE; ...
            mu_COV90; std_COV90; mu_COV95; std_COV95];

    for j = 1:numel(modelCols)
        block.(modelCols(j)) = vals(:,j);
    end

    block.nOK   = repmat(nOK, 10, 1);
    block.nFail = repmat(nFail, 10, 1);

    Rows = [Rows; block]; %#ok<AGROW>
end

disp('==================== PERFORMANCE TABLE (Mean & Std Dev + Coverage) ====================');
disp(Rows);


save('sweep_results.mat', 'Rows', 'MAE', 'RMSE', 'MAPE', 'COV90', 'COV95', ...
     'sigma_list', 'noise_names', 'R', 'trainFrac', 'orderingName', 'nn_use', 'is');

%% ============================================================
%% Local helper: apply ordering to (X,y)
%% ============================================================
function [Xo, yo] = apply_ordering(X, y, orderingName, seed)
    n = size(X,1);
    switch string(orderingName)
        case "time-major"
            [~, p] = sortrows(X, [1 2 3]);
        case "space-major"
            [~, p] = sortrows(X, [2 3 1]);
        case "random"
            rng(seed);
            p = randperm(n)';
        case "time-causal-randspace"
            rng(seed);
            tvals = X(:,1);
            [tuniq, ~] = unique(tvals, 'stable');
            p = zeros(n,1);
            pos = 1;
            for it = 1:numel(tuniq)
                idx_t = find(tvals == tuniq(it));
                idx_t = idx_t(randperm(numel(idx_t)));
                p(pos:pos+numel(idx_t)-1) = idx_t(:);
                pos = pos + numel(idx_t);
            end
        otherwise
            error('Unknown orderingName.');
    end
    Xo = X(p,:);
    yo = y(p,:);
end
