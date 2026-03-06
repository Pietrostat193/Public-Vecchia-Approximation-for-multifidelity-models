%% ============================================================
%  GP-only sweep: sigma_d2 = [2,4], is=12
%  Metrics: MAE/RMSE/MAPE + COVERAGE 95% for GP1/GP2/GP3
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

orderingName = "time-major";

% ---------------- METRICS ----------------
mae_fun  = @(yhat,y) mean(abs(yhat(:) - y(:)));
rmse_fun = @(yhat,y) sqrt(mean((yhat(:) - y(:)).^2));
mape_fun = @(yhat,y) mean(abs((yhat(:) - y(:)) ./ max(abs(y(:)),1e-12))) * 100;

% ---------------- COVERAGE ----------------
cov_fun = @(mu,s2,y,alpha) mean( abs(y(:) - mu(:)) <= sqrt(max(s2(:),0)) * norminv(1-alpha/2) );
alpha95 = 0.05;

% ---------------- OUTPUT TABLE ----------------
Rows = table();
modelCols = ["GP1","GP2","GP3"];

for sidx = 1:numel(sigma_list)

    cfg = cfg0;
    cfg.rho      = rho_fixed;
    cfg.sigma_d2 = sigma_list(sidx);

    ok_all = false(R,1);

    MAE   = nan(R, numel(modelCols));
    RMSE  = nan(R, numel(modelCols));
    MAPE  = nan(R, numel(modelCols));
    COV95 = nan(R, numel(modelCols));

    nFail = 0;

    for r = 1:R
        try
            seed = 100000*is + r + 1000*sidx;
            rng(seed);

            out = simulate_data_dynamic(seed, trainFrac, cfg);

            y = out.HF_test.fH(:);

            if ~isfield(out,'test_row_idx') || isempty(out.test_row_idx)
                error('out.test_row_idx missing/empty.');
            end

            % ---- ordering only affects training sets ----
            X_L_raw = [out.LF.t, out.LF.s1, out.LF.s2];
            y_L_raw = out.LF.fL(:);

            X_H_raw = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
            y_H_raw = out.HF_train.fH(:);

            [X_L, y_L] = apply_ordering(X_L_raw, y_L_raw, orderingName, 111);
            [X_H, y_H] = apply_ordering(X_H_raw, y_H_raw, orderingName, 222);

            % ---- Train/predict GPs ----
            ModelInfo2 = struct('X_L', X_L, 'y_L', y_L, 'X_H', X_H, 'y_H', y_H);

            % IMPORTANT: we now capture the variance outputs too
            [Y1, S1, Y2, S2, Y3, S3] = train_and_predict_gpr(ModelInfo2);

            mu1 = Y1(out.test_row_idx);
            mu2 = Y2(out.test_row_idx);
            mu3 = Y3(out.test_row_idx);

            % Variances aligned to HF_test (same indexing)
            s21 = S1(out.test_row_idx);
            s22 = S2(out.test_row_idx);
            s23 = S3(out.test_row_idx);

            % If S1/S2/S3 are STD DEV (not variance), uncomment:
            % s21 = s21.^2; s22 = s22.^2; s23 = s23.^2;

            % ---- sanity ----
            if any([numel(mu1),numel(mu2),numel(mu3)] ~= numel(y))
                error('Size mismatch between GP predictions and HF_test.');
            end

            % ---- metrics ----
            yhats = {mu1, mu2, mu3};
            s2s   = {s21, s22, s23};

            for m = 1:numel(modelCols)
                MAE(r,m)  = mae_fun(yhats{m}, y);
                RMSE(r,m) = rmse_fun(yhats{m}, y);
                MAPE(r,m) = mape_fun(yhats{m}, y);
                COV95(r,m)= cov_fun(yhats{m}, s2s{m}, y, alpha95);
            end

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

    mu_MAE   = mean(MAE(idx,:),  1);  std_MAE  = std(MAE(idx,:),  0, 1);
    mu_RMSE  = mean(RMSE(idx,:), 1);  std_RMSE = std(RMSE(idx,:), 0, 1);
    mu_MAPE  = mean(MAPE(idx,:), 1);  std_MAPE = std(MAPE(idx,:), 0, 1);

    mu_COV95  = mean(COV95(idx,:), 1); std_COV95 = std(COV95(idx,:), 0, 1);

    % ---- table block: 8 rows x 3 models ----
    block = table();
    block.NoiseLevel = repmat(noise_names(sidx), 8, 1);
    block.Metric     = ["MAE";"MAE";"RMSE";"RMSE";"MAPE";"MAPE";"COV95";"COV95"];
    block.Stat       = ["Mean";"Std";"Mean";"Std";"Mean";"Std";"Mean";"Std"];

    vals = [mu_MAE; std_MAE; mu_RMSE; std_RMSE; mu_MAPE; std_MAPE; ...
            mu_COV95; std_COV95];

    for j = 1:numel(modelCols)
        block.(modelCols(j)) = vals(:,j);
    end

    block.nOK   = repmat(nOK, 8, 1);
    block.nFail = repmat(nFail, 8, 1);

    Rows = [Rows; block]; %#ok<AGROW>
end

disp('==================== GP-ONLY TABLE (Mean/Std + COV95) ====================');
disp(Rows);

save('sweep_results_GPonly.mat','Rows','MAE','RMSE','MAPE','COV95', ...
     'sigma_list','noise_names','R','trainFrac','orderingName','is');

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
