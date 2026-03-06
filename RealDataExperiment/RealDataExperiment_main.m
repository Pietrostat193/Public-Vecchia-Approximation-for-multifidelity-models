%% MAIN EXPERIMENT: MFGP Benchmarking (Vecchia v3 + Warping + Multi-Config)
%clear; clc; 
global ModelInfo

% --- CONFIGURAZIONE INIZIALE ---
hold_id_list = unique(S.sorted_data.IDStation); % Aggiungi qui gli ID delle stazioni
capN         = 744;             % Cap dati per alleggerire il calcolo
dataFile     = "C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy\South_Lombardy_sorted_data.mat";
hold_id_list = unique(S.sorted_data.IDStation);
% Caricamento dati
%if ~exist(dataFile, 'file'), error('File dati non trovato!'); end
S = load(dataFile);
data = S.sorted_data;

% Inizializzazione contenitori risultati
all_metrics = table();
ResultsHistory = struct();

% Diary per log su file
diary('Experiment_Log.txt');

for s_idx = 1:length(hold_id_list)
    hold_id = hold_id_list(s_idx);
    s_name = sprintf('Station_%d', hold_id);
    
    fprintf('\n==========================================================\n');
    fprintf(' PROCESSING STATION: %d (%d of %d)\n', hold_id, s_idx, length(hold_id_list));
    fprintf('==========================================================\n');

    %% 1. SPLIT E PREPARAZIONE DATI
    is_hold = (data.IDStation == hold_id);
    if ~any(is_hold), fprintf('Stazione %d non trovata, salto.\n', hold_id); continue; end
    
    train_tbl = cap_times_per_station(data(~is_hold, :), capN, 42);
    test_tbl  = cap_times_per_station(data(is_hold, :),  capN, 42);
    
    X_L = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
    y_L = double(train_tbl.Wind_speed);
    X_H = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
    y_H = double(train_tbl.ws);
    Xstar    = [double(test_tbl.Time), double(test_tbl.Lat_LF), double(test_tbl.Lon_LF)];
    y_true   = double(test_tbl.ws);
    addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\5_Warping\WMFGP-main (1)\WMFGP-main")
    % --- SETUP ModelInfo BASE ---
    ModelInfo = struct();
    ModelInfo.X_L = X_L; ModelInfo.y_L = y_L;
    ModelInfo.X_H = X_H; ModelInfo.y_H = y_H;
    ModelInfo.nn_size = 25; 
    ModelInfo.kernel = "RBF"; 
    ModelInfo.jitter = 1e-5;
    ModelInfo.MeanFunction = "zero"; % Default, verrà cambiato se Adaptive
    ModelInfo.conditioning="Corr";
    ModelInfo.cov_type="RBF"
    % --- FASE 1: ESTRAZIONE INDICI (Una volta per stazione) ---
    fprintf('Fase 1: Pre-calcolo indici dei vicini (Vecchia v3)...\n');
    resL = vecchia_approx_space_time_corr_fast1(X_L, [1,1], [1,1], ModelInfo.nn_size, 1e-6, "RBF", 10, 1, 1, []);
    resH = vecchia_approx_space_time_corr_fast1(X_H, [1,1], [1,1], ModelInfo.nn_size, 1e-6, "RBF", 10, 1, 1, []);
    ModelInfo.idxL_precomputed = extract_vecchia_indices(resL.B, ModelInfo.nn_size);
    ModelInfo.idxH_precomputed = extract_vecchia_indices(resH.B, ModelInfo.nn_size);

    %% 2. LOOP CONFIGURAZIONI MFGP
    % {Tag, GLSType, RhoFunction, UseWarp}
    configs = {
        'Const_RhoC',   'fixed',    'constant',            false;
        'Adap_RhoC',    'adaptive', 'constant',            false;
        'Const_W_RhoC', 'fixed',    'constant',            true;
        'Adap_W_RhoC',  'adaptive', 'constant',            true;
        'Const_RhoA',   'fixed',    'GP_scaled_empirical', false;
        'Adap_RhoA',    'adaptive', 'GP_scaled_empirical', false
    };


load('paramBank_realdata_twoFiles.mat')
    % --- CALCOLO MEDIE REGIONALI PER WARMSTART ---
% Filtriamo per i due casi di RhoFunction
idx11 = strcmp(paramTable.RhoFunction, 'constant');
idx14 = strcmp(paramTable.RhoFunction, 'GP_scaled_empirical');
% Estraiamo i vettori hyp, li trasformiamo in matrici e calcoliamo la media
all_hyp11 = cell2mat(paramTable.hyp(idx11)'); 
hyp0_constant_mean = mean(all_hyp11, 2);
all_hyp14 = cell2mat(paramTable.hyp(idx14)'); 
hyp0_gp_emp_mean = mean(all_hyp14, 2);

fprintf('Medie regionali calcolate: Hyp11 e Hyp14 pronte.\n');

    for c = 1:size(configs, 1)
        conf_tag = configs{c,1};
        ModelInfo.GLSType = configs{c,2};
        ModelInfo.RhoFunction = configs{c,3};
        use_warp = configs{c,4};

        % --- GESTIONE WARPING ---
        if use_warp
            kernelW = "Tria";
            [~, yH_norm, bgk_H, ~] = KCDF_Estim(y_H, kernelW);
            [~, yL_norm, ~, ~]      = KCDF_Estim(y_L, kernelW);
            ModelInfo.y_H = yH_norm; 
            ModelInfo.y_L = yL_norm;
            lookup_y_H = Gen_Lookup(y_H, bgk_H, kernelW);
        else
            ModelInfo.y_H = y_H; 
            ModelInfo.y_L = y_L;
        end

        % --- OTTIMIZZAZIONE ---
        n_params = 11 + 3*(ModelInfo.RhoFunction == "GP_scaled_empirical");
        hyp0 = rand(n_params, 1);

        % --- SELEZIONE HYP0 BASATA SULLA DIMENSIONE ---
        if strcmp(ModelInfo.RhoFunction, 'constant')
            hyp0 = hyp0_constant_mean;
        else
            hyp0 = hyp0_gp_emp_mean;
        end


        opts_opt = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter', 'MaxIterations', 50);
        
        [hyp_hat, fval] = fminunc(@likelihoodVecchia_nonstat_GLS_v3, hyp0, opts_opt);

        % --- PREDIZIONE ---

         % --- PREDIZIONE ---
        % Rieseguiamo la likelihood per aggiornare ModelInfo con i parametri ottimali hyp_hat
        [~] = likelihoodVecchia_nonstat_GLS_v3(hyp_hat); 

        if strcmp(ModelInfo.GLSType, "adaptive")
            % Caso Adattivo: Trend locale [1, Lat, Lon] x 2 fedeltà = 6 beta
            fprintf('      Predizione con GLS Adattivo...\n');
            [mu_p, s2_p] = predict_calibratedCM3_AdaptiveGLS_v4(Xstar, ModelInfo);
        else
            % Caso Fixed: Trend globale [Intercetta_L, Intercetta_H] = 2 beta
            fprintf('      Predizione con GLS Fisso...\n');
            [mu_p, s2_p] = predict_calibratedCM3_fixed(Xstar, ModelInfo);
        end

        % --- INVERSIONE WARP E METRICHE ---
        if use_warp
            y_pred = Kernel_invNS(mu_p, lookup_y_H);
            u_p = Kernel_invNS(mu_p + 1.96*sqrt(s2_p), lookup_y_H);
            l_p = Kernel_invNS(mu_p - 1.96*sqrt(s2_p), lookup_y_H);
        else
            y_pred = mu_p; 
            u_p = mu_p + 1.96*sqrt(s2_p); 
            l_p = mu_p - 1.96*sqrt(s2_p);
        end

        rmse = sqrt(mean((y_true - y_pred).^2));
        mae  = mean(abs(y_true - y_pred));
        cor  = corr(y_true, y_pred);
        picp = mean((y_true >= l_p) & (y_true <= u_p)) * 100;

        % --- PRINT PARZIALE ---
        fprintf('\n   -> Config: %-15s | NLML: %8.2f\n', conf_tag, fval);
        fprintf('      RMSE: %6.4f | MAE: %6.4f | PICP: %5.2f%%\n', rmse, mae, picp);

        % --- SALVATAGGIO ---
        ResultsHistory.(s_name).(conf_tag).y_true = y_true;
        ResultsHistory.(s_name).(conf_tag).y_pred = y_pred;
        ResultsHistory.(s_name).(conf_tag).CI_up  = u_p;
        ResultsHistory.(s_name).(conf_tag).CI_low = l_p;
        ResultsHistory.(s_name).(conf_tag).hyp    = hyp_hat;
        
        res_row = table(hold_id, {conf_tag}, rmse, mae, cor, picp, fval, ...
            'VariableNames', {'Station', 'Config', 'RMSE', 'MAE', 'Corr', 'PICP_95', 'NLML'});
        all_metrics = [all_metrics; res_row];
    end
    
    % Salvataggio di sicurezza dopo ogni stazione
    save('Experiment_Full_Results.mat', 'all_metrics', 'ResultsHistory', '-v7.3');
end

diary off;
fprintf('\n>>> ESPERIMENTO COMPLETATO. Risultati salvati in Experiment_Full_Results.mat\n');

%% --- FUNZIONI HELPER ---
function idx_mat = extract_vecchia_indices(B, nn)
    n = size(B, 1);
    idx_mat = zeros(n, nn);
    for i = 2:n
        cols = find(B(i, 1:i-1));
        if ~isempty(cols)
            len = min(length(cols), nn);
            idx_mat(i, 1:len) = cols(end-len+1:end);
        end
    end
end