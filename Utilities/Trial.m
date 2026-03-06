%% RIESTIMA GP (SINGLE FIDELITY) - LEAVE-ONE-OUT STAZIONE 100
clear; clc;

% 1. Caricamento Dati Originali e Risultati MFGP
load('MFGP_Plotting_Data.mat'); % Per i risultati MFGP e metadati
S = load("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy\South_Lombardy_sorted_data.mat");
data = S.sorted_data;

% --- CONFIGURAZIONE ---
hold_id = 100;
capN = 100; 
idx_plot = 1:100; % Primi 100 punti per il grafico
c_mfgp = "Const_RhoA"; 

fprintf('Riestima GP per Stazione %d in corso...\n', hold_id);

%% 2. PREPARAZIONE DATI (Solo High Fidelity)
% Escludiamo la stazione 100 dal training
is_hold = (data.IDStation == hold_id);
train_tbl = cap_times_per_station(data(~is_hold, :), capN, 42);
test_tbl  = cap_times_per_station(data(is_hold, :),  capN, 42);

% Setup Input/Output
X_train = [double(train_tbl.Time), double(train_tbl.Lat_HF), double(train_tbl.Lon_HF)];
y_train = double(train_tbl.ws);
X_test  = [double(test_tbl.Time),  double(test_tbl.Lat_HF), double(test_tbl.Lon_HF)];
y_true  = double(test_tbl.ws);

%% 3. TRAINING GP (RE-ESTIMATION)
% Usiamo lo stesso setup SR (Subset of Regressors) del tuo script precedente
gpr_final = fitrgp(X_train, y_train, ...
    'FitMethod', 'sr', ... 
    'PredictMethod', 'sr', ...
    'KernelFunction', 'ardsquaredexponential', ... 
    'BasisFunction', 'linear', ... 
    'Standardize', true, ...
    'ActiveSetSize', 100); 

% Predizione sulla stazione 100
[m_gp, sd_gp] = predict(gpr_final, X_test);

%% 4. CONFRONTO CON MFGP
mfgp_res = ResultsHistory.(sprintf('Station_%d', hold_id)).(c_mfgp);

% Metriche GP Riestimato
mae_gp = mean(abs(y_true - m_gp));
rmse_gp = sqrt(mean((y_true - m_gp).^2));
corr_gp = corr(y_true, m_gp);

% Metriche MFGP (dal caricamento precedente)
mae_mfgp = mean(abs(y_true - mfgp_res.y_pred));

%% 5. PLOTTING COMPARATIVO (Primi 100 punti)
figure('Color', 'w', 'Position', [100 100 1000 500]);
hold on;

% Intervallo Confidenza MFGP
fill([idx_plot, fliplr(idx_plot)], [mfgp_res.CI_low(idx_plot)', fliplr(mfgp_res.CI_up(idx_plot)')], ...
    [0 0.44 0.74], 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'MFGP 95% CI');

% Plot Linee
p_gp = plot(idx_plot, m_gp(idx_plot), '--', 'Color', [0.85 0.32 0.1], 'LineWidth', 1.5, 'DisplayName', 'GP Re-estimated');
p_mfgp = plot(idx_plot, mfgp_res.y_pred(idx_plot), '-', 'Color', [0 0.44 0.74], 'LineWidth', 2, 'DisplayName', 'Proposed MFGP');
p_obs = plot(idx_plot, y_true(idx_plot), 'k.', 'MarkerSize', 14, 'DisplayName', 'Observed');

% Box Metriche
txt = {['\bfSTAZIONE ', num2str(hold_id)], ...
       ['GP MAE: ', num2str(mae_gp, '%.3f')], ...
       ['MFGP MAE: ', num2str(mae_mfgp, '%.3f')], ...
       ['GP Corr: ', num2str(corr_gp, '%.3f')]};
annotation('textbox', [0.15 0.15 0.2 0.15], 'String', txt, 'BackgroundColor', 'w', 'FaceAlpha', 0.8);

grid on; box on;
xlabel('Time (Hours)'); ylabel('Wind Speed (m/s)');
title(['Leave-One-Out Comparison: Single Fidelity GP vs. MFGP (Station ', num2str(hold_id), ')']);
legend('Location', 'northeastoutside');