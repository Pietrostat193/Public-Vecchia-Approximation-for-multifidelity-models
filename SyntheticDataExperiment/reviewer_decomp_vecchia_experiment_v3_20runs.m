%% reviewer_decomp_vecchia_experiment_v3_20runs.m
% 20-replication wrapper around the v3 reviewer experiment.
% Mirrors v3 settings: MeanFunction="GP_res", hyp_init=0.1*ones(18,1),
% Vecchia likelihood = nlml_vecchia_fullMF (with same fallback as v3).
% Adds m=30 (so the sweep matches the paper table) and reintroduces the
% Vecchia RMSE via predictVecchia_CM_calibrated2.

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
addpath(genpath(repoRoot));

outputDir = fullfile(scriptDir, 'outputs', 'reviewer_decomp_vecchia_experiment_v3_20runs');
if exist(outputDir, 'dir') ~= 7
    mkdir(outputDir);
end

[vecchiaNlmlFcn, vecchiaMethodName] = resolve_vecchia_nlml_function();

baseSeed = 12345;
nRep = 20;
trainFraction = 0.8;
sizes = [10 20 30 40 60];
conds = ["MinMax", "Corr"];

Tall = table();
exactRMSE = nan(nRep, 1);
failedRep = false(nRep, 1);

for rep = 1:nRep
    repSeed = baseSeed + rep - 1;
    fprintf('\n================ REP %d / %d | seed=%d ================\n', rep, nRep, repSeed);

    try
        rng(repSeed);
        seed = rng;
        out = simulate_data(seed, trainFraction);

        [X_test, y_test, hyp_base, alpha_exact, logdet_exact, quad_exact, rmse_base] = fit_exact_baseline(out);
        exactRMSE(rep) = rmse_base;

        for jc = 1:numel(conds)
            conditioning = conds(jc);

            for i = 1:numel(sizes)
                nn = sizes(i);

                row = evaluate_vecchia_setting(out, X_test, y_test, hyp_base, ...
                    alpha_exact, logdet_exact, quad_exact, conditioning, nn, vecchiaNlmlFcn, vecchiaMethodName);
                row.Rep = rep;
                row.Seed = repSeed;
                row.ExactRMSE = rmse_base;

                Tall = [Tall; row]; %#ok<AGROW>

                fprintf('Cond=%s | m=%d | relErr(alpha)=%.3e | relErr(logdet)=%.3e | relErr(quad)=%.3e | RMSE=%.4f\n', ...
                    conditioning, nn, row.relErr_alpha, row.relErr_logdet, row.relErr_quad, row.RMSE);
            end
        end
    catch ME
        failedRep(rep) = true;
        fprintf('[FAIL] rep=%d seed=%d: %s\n', rep, repSeed, ME.message);
        if ~isempty(ME.stack)
            fprintf('  at %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
    end
end

if isempty(Tall)
    error('All %d replications failed. No output table was produced.', nRep);
end

Summary = groupsummary(Tall, {'Conditioning', 'm'}, {'mean', 'std'}, ...
    {'relErr_alpha', 'relErr_logdet', 'relErr_quad', 'RMSE'});

repCount = groupsummary(Tall, {'Conditioning', 'm'});
Summary.n_rep = repCount.GroupCount;

Summary.ConditionLabel = strings(height(Summary), 1);
Summary.ConditionLabel(Summary.Conditioning == "MinMax") = "Nearest-Neighbor";
Summary.ConditionLabel(Summary.Conditioning == "Corr") = "Corr";

Summary = movevars(Summary, 'ConditionLabel', 'Before', 'Conditioning');
Summary = sortrows(Summary, {'Conditioning', 'm'});

PaperTable = table();
PaperTable.NeighbourSelection = Summary.ConditionLabel;
PaperTable.m = Summary.m;
PaperTable.n_rep = Summary.n_rep;
PaperTable.MeanRelKinvy_SD = format_mean_sd(Summary.mean_relErr_alpha, Summary.std_relErr_alpha, 3, 3);
PaperTable.MeanRelLogdet_SD = format_mean_sd(Summary.mean_relErr_logdet, Summary.std_relErr_logdet, 4, 3);
PaperTable.MeanRelQuad_SD = format_mean_sd(Summary.mean_relErr_quad, Summary.std_relErr_quad, 4, 4);
PaperTable.MeanRMSE = round(Summary.mean_RMSE, 3);

avgExactRMSE = mean(exactRMSE(~isnan(exactRMSE)));

disp(' ');
disp('=== 20-RUN REVIEWER SUMMARY (v3) ===');
fprintf('Vecchia likelihood used: %s\n', vecchiaMethodName);
disp(PaperTable);
fprintf('Average exact RMSE across successful replications: %.3f\n', avgExactRMSE);
fprintf('Successful replications: %d / %d\n', sum(~failedRep), nRep);

rawFile     = fullfile(outputDir, 'reviewer_v3_20runs_raw.csv');
summaryFile = fullfile(outputDir, 'reviewer_v3_20runs_summary.csv');
paperFile   = fullfile(outputDir, 'reviewer_v3_20runs_paper_table.csv');
matFile     = fullfile(outputDir, 'reviewer_v3_20runs_results.mat');

writetable(Tall, rawFile);
writetable(Summary, summaryFile);
writetable(PaperTable, paperFile);
save(matFile, 'Tall', 'Summary', 'PaperTable', 'exactRMSE', 'avgExactRMSE', ...
    'nRep', 'baseSeed', 'sizes', 'conds', 'trainFraction', 'failedRep', 'vecchiaMethodName');

fprintf('Saved raw rows to: %s\n', rawFile);
fprintf('Saved aggregated summary to: %s\n', summaryFile);
fprintf('Saved paper-style table to: %s\n', paperFile);
fprintf('Saved MAT results to: %s\n', matFile);

function [vecchiaNlmlFcn, vecchiaMethodName] = resolve_vecchia_nlml_function()
    if exist('nlml_vecchia_fullMF', 'file') == 2
        vecchiaNlmlFcn = @nlml_vecchia_fullMF;
        vecchiaMethodName = "nlml_vecchia_fullMF";
        return;
    end

    if exist('likelihoodVecchia_nonstat_GLS', 'file') == 2
        vecchiaNlmlFcn = @likelihoodVecchia_nonstat_GLS;
        vecchiaMethodName = "likelihoodVecchia_nonstat_GLS";
        warning(['nlml_vecchia_fullMF was not found on the MATLAB path. ', ...
                 'Falling back to likelihoodVecchia_nonstat_GLS.']);
        return;
    end

    error(['No Vecchia likelihood implementation found. Expected either ', ...
           'nlml_vecchia_fullMF or likelihoodVecchia_nonstat_GLS on the MATLAB path.']);
end

function [X_test, y_test, hyp_base, alpha_exact, logdet_exact, quad_exact, rmse_base] = fit_exact_baseline(out)
    X_test = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
    y_test = out.HF_test.fH(:);

    clear global ModelInfo
    global ModelInfo
    ModelInfo = struct();

    ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
    ModelInfo.y_H = out.HF_train.fH(:);
    ModelInfo.X_L = [out.LF.t, out.LF.s1, out.LF.s2];
    ModelInfo.y_L = out.LF.fL(:);

    ModelInfo.cov_type      = "RBF";
    ModelInfo.kernel        = "RBF";
    ModelInfo.combination   = "multiplicative";
    ModelInfo.jitter        = 1e-6;

    ModelInfo.MeanFunction  = "GP_res";
    ModelInfo.RhoFunction   = "constant";
    ModelInfo.usePermutation = true;
    ModelInfo.show_path_diag = false;

    ModelInfo.nn_size       = 20;
    ModelInfo.conditioning  = "Corr";
    ModelInfo.cand_mult     = 50;

    options = optimoptions('fminunc', ...
        'Algorithm','quasi-newton', ...
        'Display','off', ...
        'MaxIterations', 100, ...
        'FunctionTolerance', 1e-8);

    hyp_init = 0.1 * ones(18,1);

    [hyp_base, ~] = fminunc(@likelihood2Dsp, hyp_init, options);
    ModelInfo.hyp = hyp_base;
    likelihood2Dsp(hyp_base);

    alpha_exact  = ModelInfo.alpha;
    y_joint      = [ModelInfo.y_L; ModelInfo.y_H];
    logdet_exact = 2 * ModelInfo.log_det_classic;
    quad_exact   = y_joint' * alpha_exact;

    p_base    = predict2Dsp(X_test);
    rmse_base = sqrt(mean((p_base(:) - y_test).^2));
end

function row = evaluate_vecchia_setting(out, X_test, y_test, hyp_base, ...
        alpha_exact, logdet_exact, quad_exact, conditioning, nn, vecchiaNlmlFcn, vecchiaMethodName)

    clear global ModelInfo
    global ModelInfo
    ModelInfo = struct();

    ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
    ModelInfo.y_H = out.HF_train.fH(:);
    ModelInfo.X_L = [out.LF.t, out.LF.s1, out.LF.s2];
    ModelInfo.y_L = out.LF.fL(:);

    ModelInfo.cov_type      = "RBF";
    ModelInfo.kernel        = "RBF";
    ModelInfo.combination   = "multiplicative";
    ModelInfo.jitter        = 1e-6;

    ModelInfo.MeanFunction  = "GP_res";
    ModelInfo.RhoFunction   = "constant";
    ModelInfo.usePermutation = true;
    ModelInfo.show_path_diag = false;

    ModelInfo.nn_size       = nn;
    ModelInfo.conditioning  = conditioning;
    ModelInfo.cand_mult     = max(10, nn);
    ModelInfo.hyp           = hyp_base;

    nlml_v   = vecchiaNlmlFcn(hyp_base);
    alpha_vec = ModelInfo.SIy;
    y_tilde   = ModelInfo.y_tilde;

    y_joint = [ModelInfo.y_L; ModelInfo.y_H];
    N = numel(y_joint);
    quad_vec   = y_tilde' * alpha_vec;
    logdet_vec = 2 * (nlml_v - 0.5 * quad_vec - 0.5 * N * log(2 * pi));

    try
        yhat = predictVecchia_CM_calibrated2(X_test);
        rmse_v = sqrt(mean((yhat(:) - y_test).^2));
    catch ME_pred
        warning('predictVecchia_CM_calibrated2 failed (cond=%s, m=%d): %s', ...
            conditioning, nn, ME_pred.message);
        rmse_v = NaN;
    end

    row = table();
    row.Method        = string(vecchiaMethodName);
    row.Conditioning  = conditioning;
    row.m             = nn;
    row.NLML          = nlml_v;
    row.logdetK       = logdet_vec;
    row.quad          = quad_vec;
    row.relErr_alpha  = norm(alpha_vec - alpha_exact) / max(norm(alpha_exact), 1e-12);
    row.relErr_logdet = abs(logdet_vec - logdet_exact) / max(abs(logdet_exact), 1e-12);
    row.relErr_quad   = abs(quad_vec - quad_exact) / max(abs(quad_exact), 1e-12);
    row.RMSE          = rmse_v;
end

function out = format_mean_sd(mu, sigma, muDigits, sigmaDigits)
    out = strings(numel(mu), 1);
    for i = 1:numel(mu)
        out(i) = sprintf(['%0.', num2str(muDigits), 'f (%0.', num2str(sigmaDigits), 'f)'], mu(i), sigma(i));
    end
end
