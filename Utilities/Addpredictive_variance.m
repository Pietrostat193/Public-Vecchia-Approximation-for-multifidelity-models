% ========================================================================
% Add PI coverage to results by REFITTING likelihood per row (rebuild caches)
% and using predictive variance from predict_calibratedCM3 (with kss fixed).
%
% REQUIREMENTS:
%   - results table has ModelInfo, Xstar, y_true, case_name
%   - likelihoodVecchia_nonstat_GLS on path
%   - predict_calibratedCM3 on path (with patched prior_diag_kss_HF, see below)
%
% OUTPUT columns added to results:
%   PI80_Coverage, PI95_Coverage, PI80_WidthMean, PI95_WidthMean, PredVarMean
% ========================================================================

clc;

assert(exist('results','var')==1 && istable(results), 'Need table "results".');
assert(all(ismember(["ModelInfo","Xstar","y_true"], string(results.Properties.VariableNames))), ...
    'results must contain ModelInfo, Xstar, y_true');

% z-scores (avoid norminv dependency)
z80 = 1.2815515655446004;  % norminv(0.90)
z95 = 1.959963984540054;   % norminv(0.975)

nR = height(results);
results.PI80_Coverage   = nan(nR,1);
results.PI95_Coverage   = nan(nR,1);
results.PI80_WidthMean  = nan(nR,1);
results.PI95_WidthMean  = nan(nR,1);
results.PredVarMean     = nan(nR,1);

global ModelInfo

for i = 1:nR
    Mi = results.ModelInfo{i};
    if isempty(Mi) || ~isstruct(Mi) || ~isfield(Mi,'hyp') || isempty(Mi.hyp)
        warning('Row %d: missing ModelInfo/hyp', i);
        continue;
    end

    Xstar = double(results.Xstar{i});
    ytrue = double(results.y_true{i}(:));

    % 1) Make THIS row's model global
    ModelInfo = Mi;

    % 2) Re-run likelihood at stored hyp to rebuild debug_vecchia + any GP rho models
    try
        likelihoodVecchia_nonstat_GLS(ModelInfo.hyp);
        Mi = ModelInfo;                    % pull back updated caches/models
        results.ModelInfo{i} = Mi;         % store back
    catch ME
        warning('Row %d: likelihood refit failed: %s', i, ME.message);
        continue;
    end

    % 3) Predict mean + variance at Xstar
    try
        [mu, s2] = predict_calibratedCM3(Xstar, Mi);
    catch ME
        warning('Row %d: predict failed: %s', i, ME.message);
        continue;
    end

    mu = double(mu(:));
    s2 = double(s2(:));

    ok = isfinite(mu) & isfinite(s2) & isfinite(ytrue);
    mu = mu(ok);
    s2 = s2(ok);
    y  = ytrue(ok);

    if numel(y) < 2
        continue;
    end

    % numerical safety
    s2 = max(s2, 0);
    sd = sqrt(s2 + 1e-12);

    % 80% PI
    lo80 = mu - z80*sd;
    hi80 = mu + z80*sd;
    results.PI80_Coverage(i)  = mean(y >= lo80 & y <= hi80);
    results.PI80_WidthMean(i) = mean(hi80 - lo80);

    % 95% PI
    lo95 = mu - z95*sd;
    hi95 = mu + z95*sd;
    results.PI95_Coverage(i)  = mean(y >= lo95 & y <= hi95);
    results.PI95_WidthMean(i) = mean(hi95 - lo95);

    results.PredVarMean(i) = mean(s2);

    if mod(i,6)==0 || i==nR
        fprintf('Done %d/%d\n', i, nR);
    end
end

% Summary by case_name
disp('=== PI summary by case_name (mean) ===');
cases = unique(string(results.case_name));
for k = 1:numel(cases)
    idx = string(results.case_name) == cases(k);
    fprintf('\n%s\n', cases(k));
    fprintf('  mean PI80 cov : %.3f\n', mean(results.PI80_Coverage(idx), 'omitnan'));
    fprintf('  mean PI95 cov : %.3f\n', mean(results.PI95_Coverage(idx), 'omitnan'));
    fprintf('  mean PI80 wid : %.3f\n', mean(results.PI80_WidthMean(idx), 'omitnan'));
    fprintf('  mean PI95 wid : %.3f\n', mean(results.PI95_WidthMean(idx), 'omitnan'));
end

disp('=== First rows with PI columns ===');
disp(results(:, {'holdout_station','case_name','RMSE','MAE','CORR', ...
                 'PI80_Coverage','PI95_Coverage','PI80_WidthMean','PI95_WidthMean','PredVarMean'}));
