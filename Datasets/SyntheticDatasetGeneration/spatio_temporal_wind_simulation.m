function windData = spatio_temporal_wind_simulation()
    %% Spatio-Temporal Wind Speed Simulation

    % Add required paths
    %addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\6.VecchiaApproximation\3d_SpaceTimeVecchia\2_SyntheticData\Dati");
    %addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\6.VecchiaApproximation\3d_SpaceTimeVecchia\3_ModelToShare");
    %addpath("C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\6.VecchiaApproximation\3d_SpaceTimeVecchia\3_ModelToShare\ModelToShare2");
   % addpath 'C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\6.VecchiaApproximation\3d_SpaceTimeVecchia\2_SyntheticData\Experiment_on_synthetic'
    %% Define Model and Grid Parameters
    N = 100;
    grid_size = [3, 3]; % 3x3 spatial grid
    locations = combvec(1:grid_size(1), 1:grid_size(2))'; % 9 locations (x, y)
    num_locations = size(locations, 1);
    time = (1:N)';
    time_L = repmat(time, 9, 1);

    % Global Model Info
    global ModelInfo
    [X, Y] = meshgrid(1:3, 1:3);
    ModelInfo.X_L = [time_L, repelem([X(:), Y(:)], 100, 1)];

    %% Load Wind Speed Data
    wind_data = readtable('yearly_wind_speed_timeseries.csv');
    mean_function = wind_data{1:100, 3};

    %% Define Covariance Function Parameters
    ell_space = 3; sigma_space = 1.5;
    ell_time = 50; sigma_time = 0.3;

    K_spatio_temporal = k_space_time(ModelInfo.X_L, ModelInfo.X_L, [sigma_space, ell_space], [sigma_time, ell_time], "RBF");
    K_spatio_temporal = K_spatio_temporal + 1e-6 * eye(size(K_spatio_temporal));

    % Sample from GP for wind speed
    L = chol(K_spatio_temporal, 'lower');
    gp_sample = L * randn(num_locations * N, 1);
    wind_series = reshape(gp_sample, [N, num_locations]) + mean_function + 2;
    noise_std = 0.3;
    W_H = wind_series + noise_std * randn(N, num_locations);

    %% Corrupt Wind Speed
    load("beta_2D.mat"); % Load estimated coefficients
    sinusoidal_model_2D = @(b, xy) b(1) * sin(b(2) * xy(:,1) + b(3)) + ...
                                   b(4) * cos(b(5) * xy(:,2) + b(6)) + b(7);
    high_corr = round(sinusoidal_model_2D(beta_2D / 6, locations), 3);
    W_r = W_H .* high_corr';

    %% Generate Interference Fields
    sigma_time_h = 1;
    sigma_time_m = 2;
    sigma_time_l = 3;
    W_Lh = compute_interference(ModelInfo, sigma_space, ell_space, sigma_time_h, ell_time, W_r);
    W_La = compute_interference(ModelInfo, sigma_space, ell_space, sigma_time_m, ell_time, W_r);
    W_Ll = compute_interference(ModelInfo, sigma_space, ell_space, sigma_time_l, ell_time, W_r);

    %% Store Data in a Structured Variable
    windData = struct();
    windData.W_H = W_H;
    windData.W_Lh = W_Lh;
    windData.W_La = W_La;
    windData.W_Ll = W_Ll;
    windData.X = ModelInfo.X_L;
    windData.mean_function=mean_function;
end
