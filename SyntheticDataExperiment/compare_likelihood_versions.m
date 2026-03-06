%% COMPARE_LIKELIHOODS_ROBUST.M
function compare_likelihoods_robust(hyp_to_test)
    global ModelInfo;

    % 1. Validation
    if nargin < 1 || isempty(hyp_to_test)
        if isfield(ModelInfo, 'hyp'), hyp_to_test = ModelInfo.hyp;
        else, error('No hyperparameters provided.'); end
    end

    fprintf('============================================================\n');
    fprintf('   DIAGNOSTIC: OLD vs V3 LIKELIHOOD (ROBUST)\n');
    fprintf('============================================================\n\n');

    % --- RUN OLD VERSION ---
    [nlml_old] = likelihoodVecchia_nonstat_GLS(hyp_to_test);
    % Extract and force to scalar
    quad_old    = full(0.5 * (ModelInfo.y_tilde' * ModelInfo.SIy));
    logdetH_old = full(2 * sum(log(diag(ModelInfo.L))));
    
    % --- RUN V3 VERSION ---
    [nlml_v3] = likelihoodVecchia_nonstat_GLS_v3(hyp_to_test);
    % Extract and force to scalar
    quad_v3    = full(0.5 * (ModelInfo.y_tilde' * ModelInfo.SIy));
    logdetH_v3 = full(2 * sum(log(diag(ModelInfo.R)))); 

    % --- DATA BREAKDOWN TABLE ---
    fprintf('%-30s | %-12s | %-12s | %-12s\n', 'Component', 'Old', 'V3', 'Delta');
    fprintf('%s\n', repmat('-', 1, 75));
    
    % Use the safe_print helper to avoid sparse input errors
    safe_print('FINAL NLML', nlml_old, nlml_v3);
    safe_print('Quadratic (Data Fit)', quad_old, quad_v3);
    safe_print('log|H| (Determinant)', logdetH_old, logdetH_v3);
    
    % Metadata Comparison
    fprintf('\n%-30s | %-12s | %-12s | %-12s\n', 'Metadata', 'Old', 'V3', 'Delta');
    fprintf('%s\n', repmat('-', 1, 75));
    safe_print('Norm of Beta (GLS)', norm(full(ModelInfo.debug_vecchia.m_GLS)), norm(full(ModelInfo.beta_gls)));
    
    % --- Analysis of the 149.72 Gap ---
    diff_total = nlml_old - nlml_v3;
    fprintf('\nANALYSIS:\n');
    fprintf('Total Difference: %.4f\n', diff_total);
    
    if abs(diff_total - 149.72) < 2
        fprintf('-> The ~150 discrepancy is confirmed.\n');
        [~, max_idx] = max([abs(quad_old-quad_v3), abs(logdetH_old-logdetH_v3)]);
        if max_idx == 1
            fprintf('   CAUSE: The Quadratic terms differ. This usually means the GLS \n');
            fprintf('   mean subtraction is significantly more aggressive in V3.\n');
        else
            fprintf('   CAUSE: The Determinant terms differ. Check the sign logic of log_det_W.\n');
        end
    end
end

function safe_print(label, v1, v2)
    % Convert to double and full to prevent fprintf errors
    v1_val = double(full(v1));
    v2_val = double(full(v2));
    fprintf('%-30s | %-12.4f | %-12.4f | %-12.4f\n', label, v1_val, v2_val, v1_val - v2_val);
end