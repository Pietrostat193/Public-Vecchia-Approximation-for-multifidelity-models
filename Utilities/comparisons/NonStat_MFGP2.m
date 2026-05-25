function [p1, v1, Mdl1] = NonStat_MFGP2(X_L, X_H, y_L, y_H, varargin)
    global ModelInfo;

    %% 1. Parsing Input
    p = inputParser;
    addParameter(p, 'Kernel', 'RBF');
    addParameter(p, 'NNsize', 20);
    addParameter(p, 'MeanFunction', 'zero');
    addParameter(p, 'RhoFunction', 'constant');
    addParameter(p, 'PredictionX', []); 
    addParameter(p, 'HypInit', []);     
    addParameter(p, 'conditioning', 'Corr');
    addParameter(p, 'jitter', 1e-6);
    
    parse(p, varargin{:});
    opts = p.Results;

    %% 2. Inizializzazione ModelInfo (con i campi richiesti dall'assert)
    ModelInfo = struct(); 
    ModelInfo.X_L = X_L;
    ModelInfo.y_L = y_L;
    ModelInfo.X_H = X_H;
    ModelInfo.y_H = y_H;
    ModelInfo.kernel = opts.Kernel;      
    ModelInfo.nn_size = opts.NNsize;    
    ModelInfo.MeanFunction = opts.MeanFunction;
    ModelInfo.RhoFunction = opts.RhoFunction;
    ModelInfo.conditioning = opts.conditioning;
    ModelInfo.jitter = opts.jitter;
    ModelInfo.show_path_diag = false;
    ModelInfo.combination = "multiplicative"; 


    % --- CAMPI MANCANTI RICHIESTI DALL'ASSERT ---
    % Mappiamo 'kernel' su 'cov_type' e definiamo 'combination'
    ModelInfo.cov_type = opts.Kernel; 

    %% 3. Ottimizzazione
    fprintf('Inizio ottimizzazione BFGS...\n');
    optSettings = optimoptions('fminunc', ...
        'Algorithm', 'quasi-newton', ...
        'Display', 'iter', ...
        'MaxIterations', 100);

    [hyp_opt, min_neg_log_lik] = fminunc(@likelihoodVecchia_nonstat_GLS, opts.HypInit, optSettings);

    % Assicuriamoci che l'hyp ottimale sia salvato prima della predizione
    ModelInfo.hyp = hyp_opt;

    %% 4. Predizione Calibrata
    if ~isempty(opts.PredictionX)
        fprintf('Esecuzione predizione...\n');
        
        % Chiamata corretta: passiamo il test set e l'oggetto ModelInfo aggiornato
        [p1] = predictVecchia_CM_calibrated2(opts.PredictionX, ModelInfo);
    else
        p1 = []; v1 = [];
    end

    Mdl1.hyp = hyp_opt;
    Mdl1.min_lik = min_neg_log_lik;
end