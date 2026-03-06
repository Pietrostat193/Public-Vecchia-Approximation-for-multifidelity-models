%% MFGP vs SR BASELINE: COMPARATIVE DASHBOARD (FIRST 100 POINTS)
clear; clc;

% 1. Caricamento dei dati
load('MFGP_Plotting_Data.mat'); 
load('SR_Baseline_Results.mat'); 

% --- CONFIGURAZIONE ---
c_best = "Const_RhoC"; 
sids = [1303, 102];    
idx = 1:100; % <--- DEFINIAMO IL RANGE DI INTERESSE (Primi 100 punti)

colors = struct('mfgp', [0 0.447 0.741], 'sr', [0.466 0.674 0.188]);

% Creazione Figura
fig = figure('Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
tlo = tiledlayout(2, 1, 'TileSpacing', 'loose', 'Padding', 'compact');

for i = 1:length(sids)
    sid = sids(i);
    s_name = sprintf('Station_%d', sid);
    
    % Accesso ai dati
    mfgp_full = ResultsHistory.(s_name).(c_best);
    sr_full   = ResultsSR.(s_name);
    
    % --- SELEZIONE PRIMI 100 PUNTI ---
    % Usiamo l'operatore (idx) per forzare la stessa lunghezza su tutto
    t      = 1:length(idx);
    y_true = mfgp_full.y_true(idx);
    y_mfgp = mfgp_full.y_pred(idx);
    y_sr   = sr_full.y_pred(idx);
    ci_low = mfgp_full.CI_low(idx);
    ci_up  = mfgp_full.CI_up(idx);
    
    nexttile;
    hold on;
    
    % 1. Intervallo di Confidenza MFGP
    fill([t, fliplr(t)], [ci_low', fliplr(ci_up')], colors.mfgp, ...
        'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'MFGP 95% CI');
    
    % 2. Baseline SR (Verde Tratteggiata)
    plot(t, y_sr, '--', 'Color', colors.sr, 'LineWidth', 1.8, 'DisplayName', 'Baseline SR GP');
    
    % 3. Proposto MFGP (Blu Continua)
    plot(t, y_mfgp, '-', 'Color', colors.mfgp, 'LineWidth', 2, 'DisplayName', 'Proposed MFGP');
    
    % 4. Osservazioni (Puntini Neri)
    plot(t, y_true, 'k.', 'MarkerSize', 12, 'DisplayName', 'Observed');
    
    % Estetica
    grid on; box on;
    ylabel('Wind Speed (m/s)', 'FontWeight', 'bold');
    title_str = sprintf('Station %d: Comparison of First %d Hours', sid, length(idx));
    title(title_str, 'FontSize', 13);
    
    if i == 1
        legend('Location', 'northeastoutside');
    end
end

xlabel(tlo, 'Time (Hours)', 'FontSize', 12, 'FontWeight', 'bold');
title(tlo, 'Focus View: MFGP vs. SR GP Baseline', 'FontSize', 16, 'FontWeight', 'bold');