%% MFGP CONSOLIDATED DASHBOARD: SPATIAL + 4-PANEL TIME SERIES
% Questo script genera il grafico finale per il paper.
% Assicurati che 'MFGP_Plotting_Data.mat' sia nella cartella di lavoro.

clear; clc;
if ~exist('MFGP_Plotting_Data.mat', 'file')
    error('File MFGP_Plotting_Data.mat non trovato. Esegui prima lo script principale.');
end
load('MFGP_Plotting_Data.mat');

% --- CONFIGURAZIONE SOGGETTI E MODELLI ---
c_fixed = "Const_RhoC";
c_adap  = "Adap_RhoC";
sid_success = 102; 
sid_failure = 856;

% Impostazioni Estetiche Globali
set(0, 'DefaultAxesFontSize', 11, 'DefaultLineLineWidth', 1.2, 'DefaultAxesFontName', 'Arial');

% Creazione Figura High-Res
fig = figure('Color', 'w', 'Units', 'normalized', 'Position', [0.05 0.05 0.85 0.85]);
tlo = tiledlayout(4, 2, 'TileSpacing', 'compact', 'Padding', 'loose'); 

%% 1. TOP PANEL: SPATIAL DISTRIBUTION (Occupa le prime 2 righe)
nexttile(tlo, [2 2]);

% Filtraggio e Join Robusto
% Usiamo cellstr per gestire eventuali discrepanze di tipo stringa/cell
metrics_sub = all_metrics(strcmp(cellstr(all_metrics.Config), char(c_fixed)), :);
map_data = join(metrics_sub, station_metadata, 'LeftKeys', 'Station', 'RightKeys', 'IDStation');

% Pulizia dati NaN per evitare errori nei limiti degli assi (ylim)
map_data = map_data(~isnan(map_data.Lat) & ~isnan(map_data.Lon), :);

if isempty(map_data)
    error('Errore: La tabella map_data è vuota dopo il join. Controlla IDStation.');
end

% Plot delle stazioni: Dimensione proporzionale all'errore
scatter(map_data.Lon, map_data.Lat, 150 + (map_data.MAE*500), map_data.MAE, 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
hold on;

% --- CALCOLO LIMITI CON MARGINE ASIMMETRICO PER ABBASSARE I PUNTI ---
lat_min = min(map_data.Lat);
lat_max = max(map_data.Lat);
lat_range = lat_max - lat_min;
if lat_range == 0, lat_range = 0.1; end % Fallback per singola stazione

% Margine superiore ampio (45%) per evitare il titolo, inferiore ridotto (10%)
ylim([lat_min - lat_range*0.10, lat_max + lat_range*0.45]);

lon_min = min(map_data.Lon);
lon_max = max(map_data.Lon);
lon_range = lon_max - lon_min;
if lon_range == 0, lon_range = 0.1; end
xlim([lon_min - lon_range*0.15, lon_max + lon_range*0.15]);

% Annotazioni Stazioni Chiave
text(map_data.Lon(map_data.Station==sid_success)+0.01, map_data.Lat(map_data.Station==sid_success), ...
    [' \leftarrow Success (St. ', num2str(sid_success), ')'], 'FontWeight', 'bold', 'FontSize', 10);
text(map_data.Lon(map_data.Station==sid_failure)+0.01, map_data.Lat(map_data.Station==sid_failure), ...
    [' \leftarrow Failure (St. ', num2str(sid_failure), ')'], 'FontWeight', 'bold', 'FontSize', 10);

% Colorbar e Colormap
colormap(turbo); 
cb = colorbar('eastoutside'); 
cb.Label.String = 'Mean Absolute Error (m/s)';
cb.Label.FontWeight = 'bold';

title('A. Geographic Distribution of Prediction Errors (Best Configuration: Const\_RhoC)', 'FontSize', 14);
xlabel('Longitude'); ylabel('Latitude'); grid on;

%% 2. BOTTOM PANELS: TIME SERIES COMPARISON (4 Pannelli)

% --- Row 3: Station 102 (Success Case) ---
nexttile(tlo);
plot_station_panel(ResultsHistory, sid_success, c_fixed, 'B. St. 102: Fixed GLS (Baseline)', [0.2 0.4 0.6]);

nexttile(tlo);
plot_station_panel(ResultsHistory, sid_success, c_adap, 'C. St. 102: Adaptive GLS (Bias Correction)', [0.8 0.3 0.3]);

% --- Row 4: Station 856 (Failure Case) ---
nexttile(tlo);
plot_station_panel(ResultsHistory, sid_failure, c_fixed, 'D. St. 856: Fixed GLS (Stable)', [0.2 0.4 0.6]);

nexttile(tlo);
plot_station_panel(ResultsHistory, sid_failure, c_adap, 'E. St. 856: Adaptive GLS (Edge Instability)', [0.8 0.3 0.3]);

%% TITOLO GLOBALE CON SPAZIATURA
title(tlo, {'Multi-Fidelity GP Performance: Spatial Anchoring vs. Edge Extrapolation', ' '}, ...
    'FontSize', 18, 'FontWeight', 'bold');

%% FUNZIONE HELPER (Inclusa nello script)
function plot_station_panel(history, sid, conf, titleStr, color)
    s_name = sprintf('Station_%d', sid);
    if ~isfield(history, s_name) || ~isfield(history.(s_name), conf)
        text(0.5,0.5, 'Data Missing', 'HorizontalAlignment', 'center');
        return;
    end
    
    res = history.(s_name).(conf);
    t = 1:length(res.y_true);
    
    hold on;
    % Intervallo di Confidenza 95%
    fill([t, fliplr(t)], [res.CI_low', fliplr(res.CI_up')], color, ...
        'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Dati reali e Predizione
    plot(t, res.y_true, 'k.', 'MarkerSize', 6, 'DisplayName', 'Observed');
    plot(t, res.y_pred, 'Color', color, 'LineWidth', 1.3, 'DisplayName', 'MFGP');
    
    title(titleStr, 'FontSize', 11);
    ylabel('Wind (m/s)'); 
    grid on;
    
    % Metriche locali
    mae_val = mean(abs(res.y_true - res.y_pred));
    picp_val = mean(res.y_true >= res.CI_low & res.y_true <= res.CI_up) * 100;
    
    % Posizionamento dinamico del box delle metriche
    yl = ylim;
    text(max(t)*0.02, yl(1) + (yl(2)-yl(1))*0.85, ...
        sprintf('MAE: %.3f | PICP: %.1f%%', mae_val, picp_val), ...
        'FontSize', 9, 'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'none');
end