%% run_condition_sweep.m
% Baseline: Vecchia-approximated multi-fidelity model (your likelihoodVecchia_nonstat_GLS
% + predictVecchia_CM_calibrated2) at nn_size=50, conditioning="Corr"
% Comparator: your MFGP (train_and_predict_gpr) using Model 3
%
% NOTE: variable names now reflect what the models actually are.

clear; clc;

R = 5;
trainFrac = 0.3;

conds = make_sim_conditions();
nCond = numel(conds);

rmse_vecchia_mf = nan(R,nCond);   % Vecchia multi-fidelity baseline
rmse_mfgp       = nan(R,nCond);   % Your MFGP (train_and_predict_gpr)
adv             = nan(R,nCond);   % + => MFGP better
ok              = false(R,nCond);

% ---- Optimizer for Vecchia fit ----
opt = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'SpecifyObjectiveGradient', false, ...
    'FiniteDifferenceType','forward', ...      % more stable than central
    'FiniteDifferenceStepSize',1e-6, ...
    'Display','off', ...
    'MaxIterations', 200, ...
    'MaxFunctionEvaluations', 5000, ...
    'FunctionTolerance',1e-8, ...
    'StepTolerance',1e-8);

max_restarts_v = 2;

for ic = 1:nCond
    cfg = conds(ic);
    fprintf('\n=== Cond %d/%d: %s ===\n', ic, nCond, cfg.name);

    for r = 1:R
        seed = 1000*ic + r;

        try
            % ONLY CHANGE: dynamic simulator
            out = simulate_data_dynamic(seed, trainFrac, cfg);

            % ---------------- HF test set ----------------
            X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
            y_test = out.HF_test.fH(:);

            %% ============================================================
            % 1) Vecchia multi-fidelity baseline (your code)
            %% ============================================================
            clear global ModelInfo
            global ModelInfo
            ModelInfo = struct();

            % Joint LF + HF (this is MULTI-FIDELITY by construction)
            ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
            ModelInfo.y_H = out.HF_train.fH(:);

            ModelInfo.X_L = [out.LF.t, out.LF.s1, out.LF.s2];
            ModelInfo.y_L = out.LF.fL(:);

            % Your settings
            ModelInfo.cov_type    = "RBF";
            ModelInfo.kernel      = "RBF";
            ModelInfo.combination = "multiplicative";
            ModelInfo.jitter      = 1e-6;

            ModelInfo.MeanFunction = "zero";
            ModelInfo.RhoFunction  = "constant";

            % Vecchia settings requested
            ModelInfo.nn_size      = 50;
            ModelInfo.cand_mult    = max(10, ModelInfo.nn_size);
            ModelInfo.conditioning = "Corr";

            % Force neighbor cache rebuild each replication (avoid contamination)
            if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
            if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

            % ---- Fit hyp using Vecchia likelihood (with restarts) ----
            bestF = inf; bestHyp = [];
            for att = 1:max_restarts_v
                hyp_init = rand(11,1);
                opt2 = optimoptions(opt, 'TypicalX', 1 + abs(hyp_init));
                try
                    [htry, ftry] = fminunc(@likelihoodVecchia_nonstat_GLS, hyp_init, opt2);
                    if isfinite(ftry) && ftry < bestF
                        bestF = ftry; bestHyp = htry;
                    end
                catch
                    % ignore this restart
                end
            end

            if isempty(bestHyp) || numel(bestHyp) < 11
                warning('Vecchia MF fit failed at cond=%d rep=%d. Skipping.', ic, r);
                continue;
            end

            hyp_vecchia_mf = bestHyp(:);
            ModelInfo.hyp = hyp_vecchia_mf;

            % Evaluate once to populate internals/debug fields used by calibrated prediction
            likelihoodVecchia_nonstat_GLS(hyp_vecchia_mf);

            % Predict HF at X_test using your calibrated Vecchia MF predictor
            yhat_vecchia_mf = predictVecchia_CM_calibrated2(X_test);

            %% ============================================================
            % 2) Your MFGP (train_and_predict_gpr) - unchanged
            %% ============================================================
            ModelInfo2 = struct();
            ModelInfo2.X_L = [out.LF.t, out.LF.s1, out.LF.s2];
            ModelInfo2.y_L = out.LF.fL(:);

            ModelInfo2.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
            ModelInfo2.y_H = out.HF_train.fH(:);

            [Y_pred1_all, Y_std1_all, Y_pred2_all, Y_std2_all, Y_pred3_all, Y_std3_all] = train_and_predict_gpr(ModelInfo2);

            % Your indexing logic exactly (Model 3)
           
             Y_pred1_gp1 = Y_pred1_all;
             Y_pred2_gp2 = Y_pred2_all ;
             Y_pred3_gp3 = Y_pred3_all;


            %% ---------------- RMSE + advantage ----------------
            yhat_vecchia_mf = yhat_vecchia_mf(:);
            y_test          = y_test(:);

            rmse_vecchia_mf(r,ic) = sqrt(mean((yhat_vecchia_mf - y_test).^2));
            rmse_gp(r,ic)       = sqrt(mean((Y_pred1_gp1       - y_test).^2));
            rmse_gp(r,ic)       = sqrt(mean((Y_pred2_gp2       - y_test).^2));
            rmse_gp(r,ic)       = sqrt(mean((Y_pred3_gp3       - y_test).^2));




            adv(r,ic) = rmse_vecchia_mf(r,ic) - rmse_mfgp(r,ic); % + => MFGP better
            ok(r,ic)  = true;

            fprintf(' rep %02d: RMSE_VecchiaMF=%.4f GP=%.4f ADV=%.4f\n', ...
                r, rmse_vecchia_mf(r,ic), rmse_mfgp(r,ic), adv(r,ic));

        catch ME
            warning('cond=%d rep=%d failed: %s', ic, r, ME.message);
            continue;
        end
    end
end

%% ---------------- Summary table ----------------
T = table((1:nCond)', strings(nCond,1), zeros(nCond,1), nan(nCond,1), nan(nCond,1), nan(nCond,1), nan(nCond,1), ...
    'VariableNames', {'cond_id','name','n_ok','mean_rmse_vecchiaMF','mean_rmse_mfgp','mean_adv','pct_mfgp_wins'});

for ic = 1:nCond
    idx = ok(:,ic);

    T.name(ic) = string(conds(ic).name);
    T.n_ok(ic) = sum(idx);

    T.mean_rmse_vecchiaMF(ic) = mean(rmse_vecchia_mf(idx,ic), 'omitnan');
    T.mean_rmse_mfgp(ic)      = mean(rmse_mfgp(idx,ic),       'omitnan');

    T.mean_adv(ic)      = mean(adv(idx,ic), 'omitnan');
    T.pct_mfgp_wins(ic) = 100 * mean(adv(idx,ic) > 0);
end

T = sortrows(T, 'mean_adv', 'descend');
disp(T);

disp('Best condition (largest mean ADV):');
disp(T(1,:));
