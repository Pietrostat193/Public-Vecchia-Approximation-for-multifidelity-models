% ====================================================================
% MULTI-FIDELITY MODEL COMPUTATION TIME ANALYSIS
% ====================================================================

clc; clear; close all;

%% Load Data
D = readtable("south_data_w.csv");

% Extract observations and inputs
full_y_H = D.Wind_speed;  % High-fidelity observations
full_X_H = [D.Lcat_HF, D.Lon_HF, D.Lon_HF]; % High-fidelity inputs
full_y_L = D.ws;  % Low-fidelity observations
full_X_L = [D.Time, D.Lat_LF, D.Lon_LF]; % Low-fidelity inputs

% Set experiment parameters
max_size = min(length(full_y_H), length(full_y_L)); % Ensure max dataset size
step_size = 500;
sizes = 500:step_size:max_size; % Dataset sizes to evaluate

% Initialize storage for computation times
num_sizes = length(sizes);
computation_times = zeros(num_sizes, 3);
computation_times2 = zeros(num_sizes, 1);

%% Model Configuration
ModelInfo.jitter = 1e-6;
ModelInfo.cov_type = "RBF";
ModelInfo.combination = 'additive';
ModelInfo.rho_H = 'constant';
ModelInfo.MeanFunction = 'constant';
ModelInfo.NonStat = 'F';

% Optimization options
options = optimoptions('fminunc', ...
    'Algorithm', 'quasi-newton', ...
    'Display', 'iter', ...
    'TolFun', 1e-12, ...
    'TolX', 1e-12, ...
    'MaxFunctionEvaluations', 2000);

%% Compute Likelihood for Different Dataset Sizes
for i = 1:num_sizes
    dataset_size = sizes(i);
    disp(['Processing dataset size: ', num2str(dataset_size)])

    % Subset data
    ModelInfo.y_H = full_y_H(1:dataset_size);
    ModelInfo.y_L = full_y_L(1:dataset_size);
    ModelInfo.X_H = full_X_H(1:dataset_size, :);
    ModelInfo.X_L = full_X_L(1:dataset_size, :);

    % Initialize hyperparameters (replace with actual initialization)
    hyp_init = rand(12, 1);

    % Likelihood Vecchia NN25 (5 cores)
    ModelInfo.nn_size = 25;
    tic;
    likelihoodVecchia_nonstat(hyp_init);
    computation_times(i, 1) = toc;

    % Likelihood Vecchia NN25 (10 cores)
    tic;
    likelihoodVecchia_nonstat(hyp_init);
    computation_times(i, 3) = toc;

    % Likelihood 2Dsp (Classic)
    try
        tic;
        result = likelihood2Dsp(hyp_init);
        computation_times(i, 2) = toc;

        % Check for convergence failure
        if isnan(result) || isinf(result)
            warning('likelihood2Dsp failed to converge at dataset size %d. Stopping experiment.', dataset_size);
            break;
        end
    catch ME
        warning('Error in likelihood2Dsp at dataset size %d: %s', dataset_size, ME.message);
        break;
    end
end

%% Plot Results
figure;

% ---- Subplot 1: Full Dataset ----
subplot(1, 2, 1);
hold on;
plot(sizes(1:20) * 2, computation_times(1:20, 1), '-o', 'LineWidth', 1.5); % Vecchia NN100 5 cores
plot(sizes(1:20) * 2, computation_times(1:20, 2), '-x', 'LineWidth', 1.5); % Classic likelihood
plot(sizes(1:20) * 2, computation_times(1:20, 3), '-x', 'LineWidth', 1.5, 'Color', 'yellow'); % Vecchia NN25 5 cores
plot(sizes(1:20) * 2, computation_times2(1:20, 1), '-x', 'LineWidth', 1.5, 'Color', 'magenta'); % Vecchia NN25 10 cores
hold off;

xlabel('Dataset Size', 'FontSize', 12);
ylabel('Computation Time (seconds)', 'FontSize', 12);
title('Computation Time vs Dataset Size', 'FontSize', 14);
legend({'Vecchia NN100 (5 cores)', 'Classic Likelihood', 'Vecchia NN25 (5 cores)', 'Vecchia NN25 (10 cores)'}, ...
    'FontSize', 10, 'Location', 'northwest');
grid on;

% ---- Subplot 2: Large Dataset Subset ----
subplot(1, 2, 2);
hold on;
plot(sizes(15:20) * 2, computation_times(15:20, 1), '-x', 'LineWidth', 1.5);
plot(sizes(15:20) * 2, computation_times(15:20, 3), '-x', 'LineWidth', 1.5, 'Color', 'yellow');
plot(sizes(15:20) * 2, computation_times2(15:20, 1), '-x', 'LineWidth', 1.5, 'Color', 'magenta');
hold off;

xlabel('Dataset Size', 'FontSize', 12);
ylabel('Computation Time (seconds)', 'FontSize', 12);
title('Computation Time vs Dataset Size (Subset)', 'FontSize', 14);
legend({'Vecchia NN100 (5 cores)', 'Vecchia NN25 (5 cores)', 'Vecchia NN25 (10 cores)'}, ...
    'FontSize', 10, 'Location', 'northwest');
grid on;

%% Save Figure
outputDir = '/home/staff3/pcolombo/Matlab/3D_Vecchia_Exp';
outputFile = fullfile(outputDir, 'ComputationTime2.png');
saveas(gcf, outputFile);

disp('Plot saved successfully.');
