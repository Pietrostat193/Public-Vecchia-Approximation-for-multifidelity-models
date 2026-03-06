%% ============================================================
%  Comparison Plot with Pre-Configured ModelInfo
%  Comparing Actual Data vs GP3, Classic, and MFGP-Vecchia
%% ============================================================
clear; clc; close all;
addpath 'C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\2_SyntheticData\Synthetic_Vecchia_exp'


% ---------------- 1. SETTINGS & DATA ----------------
is          = 12;    % Sim condition
r_seed      = 1;     % Specific run
sidx        = 1;     % Noise level index
trainFrac   = 0.3;
simCond     = make_sim_conditions();
cfg         = simCond(is);
cfg.n_time  = 10;
cfg.rho     = 0.6;
cfg.sigma_d2 = 2; 

opt = optimoptions('fminunc','Algorithm','quasi-newton','Display','off');

seed = 100000*is + r_seed + 1000*sidx;
rng(seed);
out = simulate_data_dynamic(seed, trainFrac, cfg);

% ---------------- 2. CONFIGURE GLOBAL MODELINFO ----------------
% This must be done BEFORE calling prediction or likelihood functions
global ModelInfo; 
ModelInfo = struct( ...
    'X_H', [out.HF_train.t, out.HF_train.s1, out.HF_train.s2], ...
    'y_H', out.HF_train.fH(:), ...
    'X_L', [out.LF.t,       out.LF.s1,       out.LF.s2], ...
    'y_L', out.LF.fL(:), ...
    'cov_type', "RBF", ...
    'kernel', "RBF", ...
    'combination', "multiplicative", ...
    'jitter', 1e-6, ...
    'MeanFunction', "zero", ...
    'RhoFunction', "constant", ...
    'conditioning', "Corr", ...   % Added per your instruction
    'nn_size', 60, ...            % Added per your instruction
    'cand_mult', max(10, 60) ...  % Added per your instruction
);

X_test   = [out.HF_test.t, out.HF_test.s1, out.HF_test.s2];
y_actual = out.HF_test.fH(:);

% ---------------- 3. MODEL FITTING & PREDICTION ----------------

% A) GP-3D (Direct HF Prediction)
ModelInfo2 = struct('X_L', ModelInfo.X_L, 'y_L', ModelInfo.y_L, ...
                    'X_H', ModelInfo.X_H, 'y_H', ModelInfo.y_H);
[~, ~, ~, ~, Y3, ~] = train_and_predict_gpr(ModelInfo2);
pred_gp3 = Y3(out.test_row_idx);

% B) Classic MFGP
fprintf('Fitting Classic MFGP...\n');
hyp_init = rand(11,1);
[bestHyp, ~] = fminunc(@likelihood2Dsp, hyp_init, opt);
ModelInfo.hyp = bestHyp;
likelihood2Dsp(ModelInfo.hyp); 
pred_classic = predict2Dsp(X_test);

% C) MFGP-Vecchia
fprintf('Fitting MFGP-Vecchia...\n');
hyp_init_v = rand(11,1);
[bestHyp_v, ~] = fminunc(@likelihoodVecchia_nonstat_GLS, hyp_init_v, opt);
ModelInfo.hyp = bestHyp_v;
likelihoodVecchia_nonstat_GLS(ModelInfo.hyp);
pred_vecchia = predictVecchia_CM_calibrated2(X_test);

% ---------------- 4. PLOTTING 4 STATIONS ----------------
stations = unique([out.HF_test.s1, out.HF_test.s2], 'rows');
plot_indices = round(linspace(1, size(stations,1), 4));

% ---------------- 4. PLOTTING 4 STATIONS (FIXED) ----------------
stations = unique([out.HF_test.s1, out.HF_test.s2], 'rows');
plot_indices = round(linspace(1, size(stations,1), 4));

%% ============================================================
%  Find and Plot Best Performing Stations (Including GP-3D)
%% ============================================================
% 1. Calculate MAE per station for the Vecchia model
stations = unique([out.HF_test.s1, out.HF_test.s2], 'rows');
num_st   = size(stations, 1);
vecchia_mae_per_st = zeros(num_st, 1);

for i = 1:num_st
    st_loc = stations(i, :);
    idx = (out.HF_test.s1 == st_loc(1)) & (out.HF_test.s2 == st_loc(2));
    vecchia_mae_per_st(i) = mean(abs(y_actual(idx) - pred_vecchia(idx)));
end

% 2. Sort to find the 4 stations with the LOWEST Vecchia error
[sorted_mae, sort_idx] = sort(vecchia_mae_per_st, 'ascend');
best_station_indices = sort_idx(1:4);

% ---------------- 3. TIME-SERIES OF BEST STATIONS ----------------
figure('Color', 'w', 'Position', [100, 100, 1200, 800]);
tl_best = tiledlayout(2,2, 'TileSpacing', 'compact');
title(tl_best, 'Comparison at Best Vecchia MFGP Locations (Inc. GP-3D)', 'FontSize', 14);

for i = 1:4
    target_idx = best_station_indices(i);
    st_loc = stations(target_idx, :);
    
    % Find logical indices for this station
    is_st = (out.HF_test.s1 == st_loc(1)) & (out.HF_test.s2 == st_loc(2));
    
    % Extract subsets
    t_sub   = out.HF_test.t(is_st);
    y_act   = y_actual(is_st);
    y_vec   = pred_vecchia(is_st);
    y_cla   = pred_classic(is_st);
    y_gp3   = pred_gp3(is_st); % Adding GP-3D here
    
    % Sort by time
    [t_sort, s_idx] = sort(t_sub);
    
    nexttile; hold on; grid on; box on;
    % Plotting all 4 lines
    p1 = plot(t_sort, y_act(s_idx), 'k-o',  'LineWidth', 2,   'DisplayName', 'Actual');
    p2 = plot(t_sort, y_gp3(s_idx), 'r--x', 'LineWidth', 1.2, 'DisplayName', 'GP-3D');
    p3 = plot(t_sort, y_cla(s_idx), 'b--s', 'LineWidth', 1.0, 'DisplayName', 'Classic MFGP');
    p4 = plot(t_sort, y_vec(s_idx), 'g-.d', 'LineWidth', 1.5, 'DisplayName', 'MFGP-Vecchia');
    
    title(sprintf('Station: (%.2f, %.2f) | Vecchia MAE: %.4f', st_loc(1), st_loc(2), sorted_mae(i)));
    
    if i == 1
        legend([p1, p2, p3, p4], 'Location', 'best', 'FontSize', 9);
    end
end
xlabel(tl_best, 'Time (t)');
ylabel(tl_best, 'High-Fidelity Value');]


%% ============================================================
%  Spatial Comparison: MFGP-Vecchia vs. GP-3D with Training Data
%  Blue = Vecchia better | Red = GP-3D better
%  Black Dots = HF Training | Grey Crosses = LF Training
%% ============================================================

% 1. Identify unique spatial stations for testing
stations = unique([out.HF_test.s1, out.HF_test.s2], 'rows');
num_st   = size(stations, 1);
mae_vecchia = zeros(num_st, 1);
mae_gp3     = zeros(num_st, 1);

% 2. Calculate MAE for each test station
for i = 1:num_st
    st_loc = stations(i, :);
    idx = (out.HF_test.s1 == st_loc(1)) & (out.HF_test.s2 == st_loc(2));
    mae_vecchia(i) = mean(abs(y_actual(idx) - pred_vecchia(idx)));
    mae_gp3(i)     = mean(abs(y_actual(idx) - pred_gp3(idx)));
end

vecchia_wins = mae_vecchia < mae_gp3;
gp3_wins     = mae_gp3 <= mae_vecchia;

% 3. Extract unique training locations
% HF training (typically sparse)
train_HF_coords = unique([out.HF_train.s1, out.HF_train.s2], 'rows');
% LF training (typically dense)
train_LF_coords = unique([out.LF.s1, out.LF.s2], 'rows');

% 4. Spatial Plotting
figure('Color', 'w', 'Position', [150, 150, 1000, 750]);
hold on; grid on; box on;

% --- Plot Performance Zones ---
s_vec = scatter(stations(vecchia_wins, 1), stations(vecchia_wins, 2), 120, ...
                'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerEdgeColor', 'none', ...
                'MarkerFaceAlpha', 0.5, 'DisplayName', 'MFGP-Vecchia is Better');

s_gp3 = scatter(stations(gp3_wins, 1), stations(gp3_wins, 2), 120, ...
                'MarkerFaceColor', [0.8500 0.3250 0.0980], 'MarkerEdgeColor', 'none', ...
                'MarkerFaceAlpha', 0.5, 'DisplayName', 'GP-3D is Better');

% --- Plot Training Locations ---
% LF data (Grey Crosses)
p_lf = plot(train_LF_coords(:,1), train_LF_coords(:,2), 'x', ...
            'Color', [0.6 0.6 0.6], 'MarkerSize', 5, 'DisplayName', 'LF Training Locs');

% HF data (Black Circles)
p_hf = plot(train_HF_coords(:,1), train_HF_coords(:,2), 'ko', ...
            'MarkerFaceColor', 'k', 'MarkerSize', 6, 'DisplayName', 'HF Training Locs');

% Styling
title('Model Performance vs. Data Distribution', 'FontSize', 14);
xlabel('s_1', 'FontSize', 12);
ylabel('s_2', 'FontSize', 12);
legend('Location', 'northeastoutside');

% Add Annotation
win_pct = (sum(vecchia_wins) / num_st) * 100;
text(min(stations(:,1)), min(stations(:,2))-0.05, ...
    sprintf('Vecchia Advantage: %.1f%% of domain', win_pct), ...
    'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', 'w');

hold off;