function compare_full_vs_vecchia_sweep(n_trials)

global ModelInfo

if nargin < 1
    n_trials = 20;
end

fprintf('\n====================================================\n');
fprintf('   FULL vs VECCHIA (2 METHODS) PARAMETER SWEEP\n');
fprintf('====================================================\n\n');

abs_diff_vec_fullK   = zeros(n_trials,1);
rel_diff_vec_fullK   = zeros(n_trials,1);

abs_diff_vec_nested  = zeros(n_trials,1);
rel_diff_vec_nested  = zeros(n_trials,1);

for k = 1:n_trials

    % ---------------------------------------------------------
    % 1) RANDOM HYPERPARAMETERS
    % ---------------------------------------------------------

    hyp = zeros(11,1);

    % log variances
    hyp(1)  = log(0.1 + 2*rand);
    hyp(3)  = log(0.1 + 2*rand);
    hyp(8)  = log(0.1 + 2*rand);
    hyp(10) = log(0.1 + 2*rand);

    % log lengthscales
    hyp(2)  = log(0.1 + 5*rand);
    hyp(4)  = log(0.1 + 5*rand);
    hyp(9)  = log(0.1 + 5*rand);
    hyp(11) = log(0.1 + 5*rand);

    % rho
    hyp(5) = -0.9 + 1.8*rand;

    % log noises
    hyp(6) = log(1e-4 + 0.1*rand);
    hyp(7) = log(1e-4 + 0.1*rand);

    % ---------------------------------------------------------
    % 2) FULL EXACT
    % ---------------------------------------------------------

    try
        NL_full = likelihood2Dsp(hyp);
    catch
        fprintf('Trial %02d: FULL failed (PD issue)\n',k);
        continue
    end

    % ---------------------------------------------------------
    % 3) FULL-K VECCHIA
    % ---------------------------------------------------------

    try
        NL_vec_fullK = nlml_vecchia_fullMF(hyp);
    catch
        fprintf('Trial %02d: Full-K Vecchia failed\n',k);
        continue
    end

    % ---------------------------------------------------------
    % 4) NESTED / GLS VECCHIA
    % ---------------------------------------------------------

    try
        NL_vec_nested = likelihoodVecchia_nonstat_GLS(hyp);
    catch
        fprintf('Trial %02d: Nested Vecchia failed\n',k);
        continue
    end

    % ---------------------------------------------------------
    % 5) DIFFERENCES
    % ---------------------------------------------------------

    abs_diff_vec_fullK(k)  = NL_vec_fullK  - NL_full;
    rel_diff_vec_fullK(k)  = abs_diff_vec_fullK(k) / abs(NL_full);

    abs_diff_vec_nested(k) = NL_vec_nested - NL_full;
    rel_diff_vec_nested(k) = abs_diff_vec_nested(k) / abs(NL_full);

    % ---------------------------------------------------------
    % 6) PRINT PER TRIAL
    % ---------------------------------------------------------

    fprintf(['Trial %02d | Full: %10.4f | ' ...
             'FullK-Vec: %10.4f | Nested-Vec: %10.4f\n'], ...
             k, NL_full, NL_vec_fullK, NL_vec_nested);

    fprintf('         | ΔFullK: %.3e | ΔNested: %.3e\n', ...
             abs_diff_vec_fullK(k), abs_diff_vec_nested(k));

end

% =============================================================
% SUMMARY
% =============================================================

fprintf('\n====================================================\n');
fprintf('SUMMARY STATISTICS\n');
fprintf('----------------------------------------------------\n');

fprintf('FULL-K VECCHIA:\n');
fprintf('Mean abs diff   : %.6e\n', mean(abs_diff_vec_fullK,'omitnan'));
fprintf('Median abs diff : %.6e\n', median(abs_diff_vec_fullK,'omitnan'));
fprintf('Max abs diff    : %.6e\n', max(abs_diff_vec_fullK,'omitnan'));
fprintf('Mean rel diff   : %.6e\n\n', mean(rel_diff_vec_fullK,'omitnan'));

fprintf('NESTED / GLS VECCHIA:\n');
fprintf('Mean abs diff   : %.6e\n', mean(abs_diff_vec_nested,'omitnan'));
fprintf('Median abs diff : %.6e\n', median(abs_diff_vec_nested,'omitnan'));
fprintf('Max abs diff    : %.6e\n', max(abs_diff_vec_nested,'omitnan'));
fprintf('Mean rel diff   : %.6e\n', mean(rel_diff_vec_nested,'omitnan'));

fprintf('====================================================\n\n');

end
