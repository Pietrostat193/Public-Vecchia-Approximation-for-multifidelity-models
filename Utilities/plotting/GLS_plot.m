%% ------------------------------------------------------------------------
%  SCRIPT COMPLETO: STIMA RHO COSTANTE + SCOMPOSIZIONE GRAFICA
%  ------------------------------------------------------------------------
%clear; clc; close all;

%% 1. CARICAMENTO DATI
hold_id = 102; 
capN    = 100; 
dataFile = "C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy\South_Lombardy_sorted_data.mat";

S = load(dataFile);
data = S.sorted_data;

%% 2. SPLIT E PREPARAZIONE (Usando la funzione helper definita sotto)
is_hold = (data.IDStation == hold_id);
if ~any(is_hold), error('Stazione %d non trovata nel dataset!', hold_id); end

% Applichiamo il cap dei dati per alleggerire il calcolo
train_tbl = cap_times_per_station(data(~is_hold, :), capN, 42);
test_tbl  = cap_times_per_station(data(is_hold, :),  capN, 42);

% Setup Array di Input
X_L = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
y_L = double(train_tbl.Wind_speed);
X_H = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
y_H = double(train_tbl.ws);

Xstar    = [double(test_tbl.Time), double(test_tbl.Lat_LF), double(test_tbl.Lon_LF)];
y_true   = double(test_tbl.ws);
y_L_test = double(test_tbl.Wind_speed);



% Assumiamo che X_train_H e y_train_H siano i tuoi dati stazioni
% e Xstar i punti di predizione

% 1. Training di fitrgp con Approssimazione SR (Subset of Regressors)
% Usiamo 'sr' sia per il fit che per la predizione.
% Il kernel 'ardsquaredexponential' gestisce le tue 3 dimensioni.
gpr_sr = fitrgp(X_H, y_H, ...
    'FitMethod', 'sr', ... 
    'PredictMethod', 'sr', ...
    'KernelFunction', 'ardsquaredexponential', ... 
    'BasisFunction', 'linear', ... 
    'Standardize', true, ...
    'ActiveSetSize', 100); % Opzionale: definisce quanti punti usare per l'approssimazione

% 2. Predizione su Xstar
[m_ex, sd_ex] = predict(gpr_sr, Xstar);

% 3. Calcolo metriche per il benchmark (SR)
rmse_ex = sqrt(mean((y_true - m_ex).^2));
in_ex = (y_true >= (m_ex - 2*sd_ex)) & (y_true <= (m_ex + 2*sd_ex));
cov_ex = mean(in_ex) * 100;

% --- Calcolo metriche per il tuo modello Adaptive (per evitare l'errore precedente) ---
rmse_adp = sqrt(mean((y_true - m_adp).^2));
in_adp = (y_true >= (m_adp - 2*sqrt(v_adp))) & (y_true <= (m_adp + 2*sqrt(v_adp)));
cov_adp = mean(in_adp) * 100;

% 4. PLOT DI CONFRONTO FINALE
figure('Color', 'w', 'Position', [100, 100, 1100, 850]);

% --- TOP: Adaptive GLS v4 ---
subplot(2,1,1);
hold on; grid on;
fill([t_star; flipud(t_star)], [m_adp+2*sqrt(v_adp); flipud(m_adp-2*sqrt(v_adp))], ...
     [0.8, 0.9, 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
plot(t_star, y_true, 'k.', 'MarkerSize', 8, 'DisplayName', 'Truth');
plot(t_star, m_adp, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Adaptive GLS');
title(sprintf('Il Tuo Modello: Adaptive GLS v4 (RMSE: %.4f, Cov: %.2f%%)', rmse_adp, cov_adp));
ylabel('Valore'); legend('Location', 'best');

% --- BOTTOM: fitrgp SR ---
subplot(2,1,2);
hold on; grid on;
fill([t_star; flipud(t_star)], [m_ex+2*sd_ex; flipud(m_ex-2*sd_ex)], ...
     [0.9, 0.9, 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
plot(t_star, y_true, 'k.', 'MarkerSize', 8, 'DisplayName', 'Truth');
plot(t_star, m_ex, 'r-', 'LineWidth', 1.5, 'DisplayName', 'fitrgp SR');
title(sprintf('Benchmark: fitrgp SR (RMSE: %.4f, Cov: %.2f%%)', rmse_ex, cov_ex));
ylabel('Valore'); xlabel('Tempo / Indice'); legend('Location', 'best');

% 5. Print a console
fprintf('\n--- VERDETTO FINALE (Benchmark SR) ---\n');
fprintf('Adaptive GLS RMSE: %.4f\n', rmse_adp);
fprintf('fitrgp SR RMSE: %.4f\n', rmse_ex);


%% 3. SETUP MODELINFO (RHO COSTANTE)
global ModelInfo
ModelInfo = struct();
ModelInfo.X_L = X_L; ModelInfo.y_L = y_L;
ModelInfo.X_H = X_H; ModelInfo.y_H = y_H;
ModelInfo.RhoFunction = "constant"; 
ModelInfo.combination = "multiplicative";
ModelInfo.nn_size     = 25;
ModelInfo.kernel      = "RBF";
ModelInfo.jitter      = 1e-5; 
ModelInfo.MeanFunction = "zero";
ModelInfo.conditioning="Corr";
ModelInfo.cov_type="RBF"



%% 4. OTTIMIZZAZIONE
% hyp: [log_ell_t, log_ell_s, log_sigma_delta, log_ell_delta, rho, noise_L, noise_H, beta1, beta2]
hyp0 = rand(11,1); 

options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter', 'MaxIterations', 50);
fprintf('Inizio ottimizzazione GLS con Rho Costante per stazione %d...\n', hold_id);

[hyp_hat, ~] = fminunc(@likelihoodVecchia_nonstat_GLS, hyp0, options);
ModelInfo.hyp = hyp_hat;

% Eseguiamo un'ultima volta per popolare le intercette GLS in ModelInfo
[~] = likelihoodVecchia_nonstat_GLS(hyp_hat); 

%% 5. PREDIZIONE
[mu_pred, s2_pred] = predict_calibratedCM3_fixed(Xstar, ModelInfo);

%% 6. PLOT SCOMPOSIZIONE
beta_H = ModelInfo.debug_vecchia.m_GLS(2);
beta_L = ModelInfo.debug_vecchia.m_GLS(1);
rho_val = ModelInfo.hyp(5);

% Componenti algebriche
trend_hf = ones(size(mu_pred)) * beta_H;
satellite_contribution = rho_val * (y_L_test - beta_L);
trend_mf = trend_hf + satellite_contribution;

figure('Color', 'w', 'Name', 'Analisi GLS-MFGP');

subplot(2,1,1); hold on;
plot(y_true, 'k.', 'DisplayName', 'Osservazioni (HF)');
plot(y_L_test, 'b--', 'DisplayName', 'Satellite (LF)');
yline(beta_H, 'r', 'LineWidth', 2, 'DisplayName', sprintf('\\beta_{HF} (%.2f)', beta_H));
yline(beta_L, 'b', 'LineWidth', 2, 'DisplayName', sprintf('\\beta_{LF} (%.2f)', beta_L));
title('Dati e Intercette Medie GLS'); grid on; legend('Location', 'bestoutside');

subplot(2,1,2); hold on;
plot(trend_mf, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Trend MF (\beta_{HF} + \rho \Delta LF)');
plot(mu_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'Predizione Finale (Trend + GP)');
plot(y_true, 'k.', 'DisplayName', 'Verità');
rmse = sqrt(mean((y_true - mu_pred).^2));
title(sprintf('Costruzione Predizione: \\rho = %.3f | RMSE = %.4f', rho_val, rmse));
grid on; legend('Location', 'bestoutside');



options = optimoptions('fminunc', ...
                'Algorithm','quasi-newton', ...
                'SpecifyObjectiveGradient', false, ...
                'FiniteDifferenceType','central', ...
                'FiniteDifferenceStepSize',1e-4, ...
                'TypicalX', abs(hyp_init)+1, ...
                'Display','iter', ...
                'MaxIterations', 200, ...
                'MaxFunctionEvaluations', 5000, ...
                'FunctionTolerance',1e-5, ...
                'StepTolerance',1e-5);
    
  
 [hyp_opt, ~] = fminunc(@likelihoodVecchia_nonstat, ModelInfo.hyp, options);
 ModelInfo.hyp=hyp_opt;
 likelihoodVecchia_nonstat(ModelInfo.hyp); % refresh caches if any
[m, v, ci_lower_all, ci_upper_all] = predictVecchia_nonstat2(prediction_X);
rmse_old= sqrt(mean((y_true - m).^2));




% === Professional Plot: RMSE Model Comparison ===

% 1. Calculate RMSE for each approach
rmse_final  = sqrt(mean((y_true - mu_pred).^2));
rmse_zero   = sqrt(mean((y_true - m).^2));
rmse_offset = sqrt(mean((y_true - (m - 2.5)).^2));

% Color Palette (Academic/Publication Quality)
color_truth  = [0.2, 0.2, 0.2];      % Dark Charcoal for y_true
color_final  = [0.0, 0.45, 0.74];    % Deep Blue for mu_pred
color_zero   = [0.6, 0.6, 0.6];      % Grey for m
color_offset = [0.85, 0.33, 0.1];    % Rust Orange for m - 2.5

figure('Color', 'w', 'Units', 'pixels', 'Position', [100, 100, 950, 550]);
hold on;

% 2. Ground Truth (HF Observations)
plot(y_true, '.', 'Color', color_truth, 'MarkerSize', 10, ...
    'DisplayName', 'Ground Truth (y\_true)');

% 3. Baseline: Zero-Mean GP (m)
plot(m, '--', 'Color', color_zero, 'LineWidth', 1.2, ...
    'DisplayName', sprintf('Zero-Mean GP (RMSE: %.4f)', rmse_zero));

% 4. Baseline: Constant Offset (m - 2.5)
plot(m - 2.5, ':', 'Color', color_offset, 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Constant Offset (RMSE: %.4f)', rmse_offset));

% 5. Proposed: Adaptive GLS Prediction (mu_pred)
plot(mu_pred, '-', 'Color', color_final, 'LineWidth', 2.2, ...
    'DisplayName', sprintf('Adaptive GLS + GP (RMSE: %.4f)', rmse_final));

% --- Formatting ---
rho_val = ModelInfo.hyp(5); 

title('Multi-Fidelity Systematic Shift Correction', 'FontSize', 14, 'FontWeight', 'bold');
subtitle(sprintf('Model Performance Comparison | Scaling Factor no GLS \\rho = %.3f', rho_val), 'FontSize', 11);

xlabel('Sample Index', 'FontSize', 12);
ylabel('HF Response Value', 'FontSize', 12);

grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.4, 'TickDir', 'out', 'Box', 'on');

% Legend placement (outside to avoid overlap)
legend('Location', 'bestoutside', 'EdgeColor', 'none', 'FontSize', 10);

% Tighten layout and add 15% padding for visual clarity
axis tight;
yl = ylim;
ylim([yl(1) - 0.15*diff(yl), yl(2) + 0.15*diff(yl)]); 

hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adaptive GLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ModelInfo.GLSType = "adaptive";
ModelInfo.RhoFunction = "GP_scaled_empirical";
hyp_init=rand(14,1);

likelihoodVecchia_nonstat_GLS_v2(hyp_init)


options = optimoptions('fminunc', ...
                'Algorithm','quasi-newton', ...
                'SpecifyObjectiveGradient', false, ...
                'FiniteDifferenceType','central', ...
                'FiniteDifferenceStepSize',1e-4, ...
                'TypicalX', abs(hyp_init)+1, ...
                'Display','iter', ...
                'MaxIterations', 200, ...
                'MaxFunctionEvaluations', 5000, ...
                'FunctionTolerance',1e-5, ...
                'StepTolerance',1e-5);
    
  
ModelInfo.RhoFunction="constant"
 [hyp_opt, ~] = fminunc(@likelihoodVecchia_nonstat_GLS_v2,hyp_opt, options);

ModelInfo.hyp=hyp_opt;
likelihoodVecchia_nonstat_GLS_v2(hyp_opt);
[m_adp,v_adp]=predict_calibratedCM3_AdaptiveGLS_v3(Xstar,ModelInfo);
[m_adp,v_adp]=predict_calibratedCM4_AdaptiveGLS_v4(Xstar,ModelInfo);
rmse_adp=sqrt(mean((y_true - (m_adp)).^2));


options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter', 'MaxIterations', 50);

ModelInfo.RhoFunction="constant"
ModelInfo.GLSType = "adaptive"
hyp_init=rand(11,1)
 %[hyp_opt, ~] = fminunc(@likelihoodVecchia_nonstat_GLS_v2, hyp_init, options);
  [hyp_opt, ~] = fminunc(@likelihoodVecchia_nonstat_GLS_v3, hyp_opt, options);

ModelInfo.hyp=hyp_opt;

likelihoodVecchia_nonstat_GLS_v2(hyp_opt)
likelihoodVecchia_nonstat_GLS_v3(hyp_opt)

[m_adp,v_adp]=predict_calibratedCM3_AdaptiveGLS_v4(Xstar,ModelInfo);
rmse_old= sqrt(mean((y_true - m_adp).^2))

%% --- FUNZIONI HELPER (Devono stare in fondo allo script) ---
function tbl = cap_times_per_station(tbl, capN, seed)
    if isempty(capN) || capN <= 0, return; end
    ids = unique(tbl.IDStation);
    keep = false(height(tbl),1);
    for i = 1:numel(ids)
        idx = find(tbl.IDStation == ids(i));
        if numel(idx) <= capN, keep(idx) = true; 
        else
            rng(seed); 
            % Campionamento uniforme nel tempo
            pick = round(linspace(1, numel(idx), capN));
            sub_idx = idx(pick);
            keep(sub_idx) = true;
        end
    end
    tbl = tbl(keep,:);
end