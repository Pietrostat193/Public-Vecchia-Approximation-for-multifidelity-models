%% 1. SETUP DATI E PARAMETRI
capN = 744; 
dataFile = "C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy\South_Lombardy_sorted_data.mat";
if ~exist(dataFile, 'file')
    error('File dati non trovato al percorso specificato.');
end
S = load(dataFile);
data = S.sorted_data;

% Lista delle stazioni
station_list = [100, 102, 110, 114, 134, 136, 137, 150, 166, 214, 525, 629, 677, 856, 900, 1266, 1297, 1303];
num_stations = length(station_list);

% --- AGGIORNAMENTO: Inizializzazione contenitori ---
results = table(station_list', zeros(num_stations, 1), zeros(num_stations, 1), ...
    zeros(num_stations, 1), zeros(num_stations, 1), ...
    'VariableNames', {'StationID', 'RMSE_SR', 'MAE_SR', 'Corr_SR', 'Coverage_SR'});

% Oggetto per contenere le serie temporali per i plot futuri
ResultsSR = struct();

%% 2. LOOP PER TUTTE LE STAZIONI
for i = 1:num_stations
    hold_id = station_list(i);
    s_name = sprintf('Station_%d', hold_id);
    fprintf('Elaborazione Stazione ID: %d (%d di %d)...\n', hold_id, i, num_stations);
    
    is_hold = (data.IDStation == hold_id);
    if ~any(is_hold), continue; end
    
    train_tbl = cap_times_per_station(data(~is_hold, :), capN, 42);
    test_tbl  = cap_times_per_station(data(is_hold, :),  capN, 42);
    
    X_H    = [double(train_tbl.Time), double(train_tbl.Lat_HF), double(train_tbl.Lon_HF)];
    y_H    = double(train_tbl.ws);
    Xstar  = [double(test_tbl.Time),  double(test_tbl.Lat_HF), double(test_tbl.Lon_HF)];
    y_true = double(test_tbl.ws);
    
    try
        % 1. Training
        gpr_sr = fitrgp(X_H, y_H, ...
            'FitMethod', 'sr', ... 
            'PredictMethod', 'sr', ...
            'KernelFunction', 'ardsquaredexponential', ... 
            'BasisFunction', 'linear', ... 
            'Standardize', true, ...
            'ActiveSetSize', 100); 
        
        % 2. Predizione
        [m_ex, sd_ex] = predict(gpr_sr, Xstar);
        
        % --- AGGIORNAMENTO: Calcolo metriche estese ---
        rmse_ex = sqrt(mean((y_true - m_ex).^2));
        mae_ex  = mean(abs(y_true - m_ex));
        corr_ex = corr(y_true, m_ex);
        
        % Coverage (usando 1.96 per il 95%)
        u_p = m_ex + 1.96*sd_ex;
        l_p = m_ex - 1.96*sd_ex;
        cov_ex  = mean(y_true >= l_p & y_true <= u_p) * 100;
        
        % --- AGGIORNAMENTO: Salvataggio nell'oggetto per Plotting ---
        ResultsSR.(s_name).y_true = y_true;
        ResultsSR.(s_name).y_pred = m_ex;
        ResultsSR.(s_name).CI_up  = u_p;
        ResultsSR.(s_name).CI_low = l_p;
        
        % Salvataggio nella tabella riassuntiva
        results.RMSE_SR(i)     = rmse_ex;
        results.MAE_SR(i)      = mae_ex;
        results.Corr_SR(i)     = corr_ex;
        results.Coverage_SR(i) = cov_ex;
        
        fprintf('   DONE - RMSE: %.4f, MAE: %.4f, Corr: %.3f, Cov: %.2f%%\n', ...
            rmse_ex, mae_ex, corr_ex, cov_ex);
        
    catch ME
        fprintf('   ERRORE nella stazione %d: %s\n', hold_id, ME.message);
    end
end

%% 3. VISUALIZZAZIONE E SALVATAGGIO
fprintf('\n--- SUMMARY RISULTATI FINALI (SR Baseline) ---\n');
disp(results);

% Salvataggio per poter confrontare con MFGP nel dashboard
save('SR_Baseline_Results.mat', 'results', 'ResultsSR');