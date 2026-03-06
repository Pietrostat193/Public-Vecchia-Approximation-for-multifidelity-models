%% SCRIPT DI TEST: VALIDAZIONE PREDIZIONE CALIBRATA FIXED
% Questo script valida la funzione predict_calibratedCM3_fixed 

fprintf('Esecuzione Test su predict_calibratedCM3_fixed...\n');


    % 1. Chiamata alla funzione
    % Assicurati che y_true, Xstar e ModelInfo siano caricati nel workspace
    [mu_f, s2_f] = predict_calibratedCM3_fixed(Xstar, ModelInfo);

    % 2. Calcolo Residui e Deviazione Standard
    residui = y_true - mu_f;
    std_pred = sqrt(s2_f);
    
    % 3. Calcolo Metriche Core
    rmse_val = sqrt(mean(residui.^2));
    avg_std  = mean(std_pred);
    z_scores = residui ./ std_pred;
    std_z    = std(z_scores);
    
    % Coverage al 95% (1.96 deviazioni standard)
    coverage = mean(abs(z_scores) <= 1.96) * 100;

    % 4. Visualizzazione Risultati
    fprintf('\n--- REPORT PERFORMANCE ---\n');
    fprintf('RMSE:               %.4f (Punto di riferimento: 0.3010)\n', rmse_val);
    fprintf('Incertezza Media:   %.4f (Precedente: 0.1628)\n', avg_std);
    fprintf('Rapporto RMSE/Std:  %.4f (Target: 1.0)\n', rmse_val/avg_std);
    fprintf('--------------------------\n');
    fprintf('Std Dev Z-score:    %.4f (Target: 1.0)\n', std_z);
    fprintf('COVERAGE 95%%:       %.2f%% (Target: 95%%)\n', coverage);
    fprintf('--------------------------\n');

    % 5. Grafico di Diagnostica
    figure('Color', 'w', 'Name', 'Validazione Fixed');
    
    % Subplot 1: Predizione vs Realtà
    subplot(2,1,1);
    hold on;
    fill([1:numel(mu_f), fliplr(1:numel(mu_f))], ...
         [(mu_f + 1.96*std_pred)', fliplr((mu_f - 1.96*std_pred)')], ...
         [0.3 0.7 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'IC 95%');
    plot(y_true, 'k.', 'MarkerSize', 7, 'DisplayName', 'Verità');
    plot(mu_f, 'r-', 'LineWidth', 1, 'DisplayName', 'Predizione');
    title(['RMSE: ', num2str(rmse_val, '%.3f'), ' | Coverage: ', num2str(coverage, '%.1f'), '%']);
    grid on; legend('Location', 'best');

    % Subplot 2: Istogramma Z-score
    subplot(2,1,2);
    histogram(z_scores, 'Normalization', 'pdf', 'FaceColor', [0.4 0.4 0.4]);
    hold on;
    x_range = linspace(-4, 4, 100);
    plot(x_range, normpdf(x_range, 0, 1), 'r', 'LineWidth', 2, 'DisplayName', 'Normale Std');
    title('Distribuzione degli Errori Standardizzati (Z-score)');
    xlabel('Z-score'); ylabel('Densità');
    legend; grid on;

