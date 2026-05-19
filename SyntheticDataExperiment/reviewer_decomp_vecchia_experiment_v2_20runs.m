clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
addpath(genpath(repoRoot));

outputDir = fullfile(scriptDir, 'outputs', 'reviewer_decomp_vecchia_experiment_v2_20runs');
if exist(outputDir, 'dir') ~= 7
    mkdir(outputDir);
end

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
        out = simulate_data(repSeed, trainFraction);
        [X_test, y_test, hyp_base, alpha_exact, logdet_exact, quad_exact, rmse_base] = fit_exact_baseline(out);
        exactRMSE(rep) = rmse_base;

        for jc = 1:numel(conds)
            conditioning = conds(jc);

            for i = 1:numel(sizes)
                nn = sizes(i);

                row = evaluate_vecchia_setting(out, X_test, y_test, hyp_base, alpha_exact, logdet_exact, quad_exact, conditioning, nn);
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
    error('All 20 replications failed. No output table was produced.');
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
disp('=== 20-RUN REVIEWER SUMMARY ===');
disp(PaperTable);
fprintf('Average exact RMSE across successful replications: %.3f\n', avgExactRMSE);
fprintf('Successful replications: %d / %d\n', sum(~failedRep), nRep);

rawFile = fullfile(outputDir, 'reviewer_v2_20runs_raw.csv');
summaryFile = fullfile(outputDir, 'reviewer_v2_20runs_summary.csv');
paperFile = fullfile(outputDir, 'reviewer_v2_20runs_paper_table.csv');
matFile = fullfile(outputDir, 'reviewer_v2_20runs_results.mat');

writetable(Tall, rawFile);
writetable(Summary, summaryFile);
writetable(PaperTable, paperFile);
save(matFile, 'Tall', 'Summary', 'PaperTable', 'exactRMSE', 'avgExactRMSE', 'nRep', 'baseSeed', 'sizes', 'conds', 'trainFraction', 'failedRep');

fprintf('Saved raw rows to: %s\n', rawFile);
fprintf('Saved aggregated summary to: %s\n', summaryFile);
fprintf('Saved paper-style table to: %s\n', paperFile);
fprintf('Saved MAT results to: %s\n', matFile);

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
    ModelInfo.cov_type = "RBF";
    ModelInfo.kernel = "RBF";
    ModelInfo.combination = "multiplicative";
    ModelInfo.jitter = 1e-6;
    ModelInfo.MeanFunction = "zero";
    ModelInfo.RhoFunction = "constant";
    ModelInfo.nn_size = 20;
    ModelInfo.conditioning = "Corr";
    ModelInfo.cand_mult = 50;

    options = optimoptions('fminunc', ...
        'Algorithm', 'quasi-newton', ...
        'SpecifyObjectiveGradient', false, ...
        'FiniteDifferenceType', 'central', ...
        'FiniteDifferenceStepSize', 1e-4, ...
        'Display', 'off', ...
        'MaxIterations', 200, ...
        'MaxFunctionEvaluations', 5000, ...
        'FunctionTolerance', 1e-8, ...
        'StepTolerance', 1e-8);

    hyp_init = rand(11,1);
    options = optimoptions(options, 'TypicalX', 1 + abs(hyp_init));

    [hyp_base, ~] = fminunc(@likelihood2Dsp, hyp_init, options);
    ModelInfo.hyp = hyp_base;
    likelihood2Dsp(hyp_base);

    alpha_exact = ModelInfo.alpha;
    y_joint = [ModelInfo.y_L; ModelInfo.y_H];
    logdet_exact = 2 * ModelInfo.log_det_classic;
    quad_exact = y_joint' * alpha_exact;

    p_base = predict2Dsp(X_test);
    rmse_base = sqrt(mean((p_base(:) - y_test).^2));
end

function row = evaluate_vecchia_setting(out, X_test, y_test, hyp_base, alpha_exact, logdet_exact, quad_exact, conditioning, nn)
    clear global ModelInfo
    global ModelInfo
    ModelInfo = struct();
    ModelInfo.X_H = [out.HF_train.t, out.HF_train.s1, out.HF_train.s2];
    ModelInfo.y_H = out.HF_train.fH(:);
    ModelInfo.X_L = [out.LF.t, out.LF.s1, out.LF.s2];
    ModelInfo.y_L = out.LF.fL(:);
    ModelInfo.cov_type = "RBF";
    ModelInfo.kernel = "RBF";
    ModelInfo.combination = "multiplicative";
    ModelInfo.jitter = 1e-6;
    ModelInfo.MeanFunction = "zero";
    ModelInfo.RhoFunction = "constant";
    ModelInfo.nn_size = nn;
    ModelInfo.conditioning = conditioning;
    ModelInfo.cand_mult = max(10, nn);

    nlml_v = likelihoodVecchia_nonstat_GLS(hyp_base);
    dbg = ModelInfo.debug_vecchia;

    y_joint = [ModelInfo.y_L; ModelInfo.y_H];
    N = numel(y_joint);
    alpha_vec = dbg.SIy;
    quad_vec = y_joint' * alpha_vec;
    logdet_vec = 2 * (nlml_v - 0.5 * quad_vec - 0.5 * N * log(2 * pi));

    yhat = predictVecchia_CM_calibrated2(X_test);
    rmse_v = sqrt(mean((yhat(:) - y_test).^2));

    row = table();
    row.Conditioning = conditioning;
    row.m = nn;
    row.NLML = nlml_v;
    row.logdetK = logdet_vec;
    row.quad = quad_vec;
    row.relErr_alpha = norm(alpha_vec - alpha_exact) / max(norm(alpha_exact), 1e-12);
    row.relErr_logdet = abs(logdet_vec - logdet_exact) / max(abs(logdet_exact), 1e-12);
    row.relErr_quad = abs(quad_vec - quad_exact) / max(abs(quad_exact), 1e-12);
    row.RMSE = rmse_v;
end

function out = format_mean_sd(mu, sigma, muDigits, sigmaDigits)
    out = strings(numel(mu), 1);
    for i = 1:numel(mu)
        out(i) = sprintf(['%0.', num2str(muDigits), 'f (%0.', num2str(sigmaDigits), 'f)'], mu(i), sigma(i));
    end
end