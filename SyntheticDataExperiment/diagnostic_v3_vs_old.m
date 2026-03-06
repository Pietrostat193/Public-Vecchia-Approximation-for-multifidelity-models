%% FIXED_DIAGNOSTIC.M
function fixed_diagnostic()
    global ModelInfo;
    hyp = ModelInfo.hyp; % This is 11 elements
    
    % Save the current mean function to restore it later
    original_mean = ModelInfo.MeanFunction;
    
    try
        fprintf('--- COMPARING EXACT (2Dsp) vs VECCHIA (Approx) ---\n');

        % 1. Run Exact 2Dsp (No Mean subtraction)
        nlml_2dsp = likelihood2Dsp(hyp);
        quad_2dsp = 0.5 * (ModelInfo.y_tilde' * ModelInfo.alpha);
        
        % 2. Run Vecchia (Force Mean to 'zero' to avoid hyp(12) error)
        ModelInfo.MeanFunction = 'zero';
        nlml_vecchia = likelihoodVecchia_nonstat_GLS(hyp);
        quad_vecchia = 0.5 * (ModelInfo.y_tilde' * ModelInfo.SIy);

        fprintf('\n[RESULTS - ZERO MEAN CASE]\n');
        fprintf('Metric         | Exact (2Dsp) | Vecchia (Approx) | Delta\n');
        fprintf('---------------------------------------------------------\n');
        fprintf('Total NLML     | %12.2f | %12.2f     | %12.2f\n', nlml_2dsp, nlml_vecchia, nlml_2dsp - nlml_vecchia);
        fprintf('Quadratic Term | %12.2f | %12.2f     | %12.2f\n', quad_2dsp, quad_vecchia, quad_2dsp - quad_vecchia);

    catch ME
        fprintf('\nError encountered: %s\n', ME.message);
    end
    
    % Restore original setting
    ModelInfo.MeanFunction = original_mean;
end