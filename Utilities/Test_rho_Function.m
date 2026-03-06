%% ------------------------------------------------------------------------
%  TEST MULTI-STAZIONE (5 HOLD-OUT) + RHO INTERPOLATO
%  ------------------------------------------------------------------------
%clear; clc; close all;

%% 1. IMPOSTAZIONI
all_ids = [110, 114]; % Le 5 stazioni consecutive
capN    = 100; 
dataFile = "C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy\South_Lombardy_sorted_data.mat";

S = load(dataFile);
data = S.sorted_data;

% Inizializzazione metriche
results_table = table();

for h_idx = 1:numel(all_ids)
    hold_id = all_ids(h_idx);
    fprintf('\n--- TEST STAZIONE %d (%d/%d) ---\n', hold_id, h_idx, numel(all_ids));

    %% 2. SPLIT E SETUP
    is_hold = (data.IDStation == hold_id);
    train_tbl = cap_times_per_station(data(~is_hold,:), capN, 42);
    test_tbl  = cap_times_per_station(data(is_hold,:),  capN, 42);

    X_L = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
    y_L = double(train_tbl.Wind_speed);
    X_H = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
    y_H = double(train_tbl.ws);
    Xstar = [double(test_tbl.Time), double(test_tbl.Lat_LF), double(test_tbl.Lon_LF)];
    y_true = double(test_tbl.ws);

    global ModelInfo
    ModelInfo = struct();
    ModelInfo.X_L = X_L; ModelInfo.y_L = y_L;
    ModelInfo.X_H = X_H; ModelInfo.y_H = y_H;
    ModelInfo.cov_type     = "RBF";
    ModelInfo.combination  = "multiplicative";
    ModelInfo.RhoFunction  = "GP_scaled_empirical"; 
    ModelInfo.nn_size      = 25;
    ModelInfo.kernel       = "RBF";
    ModelInfo.conditioning = "Corr";
    ModelInfo.jitter       = 1e-8;
    ModelInfo.MeanFunction = "zero";

    %% 3. OTTIMIZZAZIONE
    hyp0 = default_hyp_init("GP_scaled_empirical"); 
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off', 'MaxIterations', 50);
    
    try
        [hyp_hat, ~] = fminunc(@likelihoodVecchia_nonstat_GLS, hyp0, options);
        ModelInfo.hyp = hyp_hat;
        [~] = likelihoodVecchia_nonstat_GLS(hyp_hat); % Popola ModelInfo.rho_local
        
        %% 4. PREDIZIONE
        [mu_pred, s2_pred] = predict_calibratedCM3_fixed(Xstar, ModelInfo);

        % Metriche
        rmse = sqrt(mean((y_true - mu_pred).^2));
        z_scores = (y_true - mu_pred) ./ sqrt(s2_pred);
        coverage = mean(abs(z_scores) <= 1.96) * 100;
        
        % Salvataggio
        results_table(h_idx,:) = table(hold_id, rmse, coverage, std(z_scores), ...
            'VariableNames', {'ID', 'RMSE', 'Coverage', 'StdZ'});
        
        fprintf('RMSE: %.4f | Coverage: %.1f%%\n', rmse, coverage);
    catch ME
        fprintf('Errore stazione %d: %s\n', hold_id, ME.message);
    end
end

%% 5. REPORT FINALE E PLOT
disp(results_table);

figure('Color','w');
subplot(1,2,1); bar(results_table.RMSE); xticklabels(results_table.ID);
title('RMSE per Stazione'); ylabel('m/s'); grid on;

subplot(1,2,2); bar(results_table.Coverage); hold on;
yline(95, 'r--', '95% Target'); xticklabels(results_table.ID);
title('Coverage 95%'); ylabel('%'); grid on;

%% ============================ HELPERS ============================
function tbl = cap_times_per_station(tbl, capN, seed)
    ids = unique(tbl.IDStation);
    keep = false(height(tbl),1);
    for i = 1:numel(ids)
        idx = find(tbl.IDStation == ids(i));
        if numel(idx) <= capN, keep(idx) = true; 
        else
            rng(seed); pick = round(linspace(1, numel(idx), capN));
            keep(idx(pick)) = true;
        end
    end
    tbl = tbl(keep,:);
end

function hyp0 = default_hyp_init(RhoFunction)
    hyp0 = zeros(14,1);
    hyp0(1:4) = log(1);   % Lengthscales
    hyp0(5) = 0.8;        % Rho base (partiamo da un valore alto)
    hyp0(6:7) = log(0.1); % Noise
    hyp0(12) = log(0.5);  % sigma rho
    hyp0(13) = log(1.0);  % spatial ell rho
end