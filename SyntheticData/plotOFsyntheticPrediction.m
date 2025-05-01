
%% 1. Add Paths to Auxiliary Functions
addpath('C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\6.VecchiaApproximation\3d_SpaceTimeVecchia\2_SyntheticData\Synthetic_Vecchia_exp\AuxiliaryFunctions')
addpath('/home/staff3/pcolombo/Matlab/VecchiaPaper/Synthetic_Vecchia_exp/AuxiliaryFunctions')

%% 2. Generate Synthetic Wind Data and Reshape to Long Format
windData = spatio_temporal_wind_simulation();
fields = fieldnames(windData);
matrix_LF = windData.(fields{2}); 
long_format_W_L = reshape(matrix_LF, [], 1);
long_format_W_H = reshape(windData.W_H, [], 1);

corr(long_format_W_L, long_format_W_H);  % Compare fidelity sources

%% 3. Select Random Spatial Locations
num_selected = 5;
all_spatial_locations = [1,1; 2,1; 3,1; 1,2; 2,2; 3,2; 1,3; 2,3; 3,3];
ri = randperm(size(all_spatial_locations,1), num_selected);
random_spatial_locations = all_spatial_locations(ri, :);
[selected_rows, selected_indices] = select_random_spatial_locations(windData, num_selected);

%% 4. Prepare Training and Test Sets
numPoints = 900;
X_train_L = windData.X;
X_train_H = windData.X(selected_indices,:);
y_train_L = long_format_W_L;
y_train_H = long_format_W_H(selected_indices,:);
test_index = setdiff(1:numPoints, selected_indices);
y_test = long_format_W_H(test_index,:);

%% 5. Train Multi-Fidelity GP Model (Initial Training)
global ModelInfo
ModelInfo = struct();
ModelInfo.X_L = X_train_L;
ModelInfo.X_H = X_train_H;
ModelInfo.y_L = y_train_L;
ModelInfo.y_H = y_train_H;

hyp_classic = rand(1,11);
[p, v, Mdl] = NonStat_MFGP(ModelInfo.X_L, ModelInfo.X_H, ModelInfo.y_L, ModelInfo.y_H, ...
    'Kernel', 'RBF', ...
    'MeanFunction', 'zero', ...
    'RhoFunction', 'constant', ...
    'Conditioning', "MinMax", ...
    'PredictionX', ModelInfo.X_L, ...
    'HypInit', hyp_classic);

%% 6. Generate Predictions from All Models
[Y_pred1, Y_std1, Y_pred2, Y_std2, Y_pred3, Y_std3] = train_and_predict_gpr(ModelInfo);
GP_1D_LF = Y_pred1(test_index);
MFc0 = p(test_index);
y_truth = y_test;
MFc0_var = v(test_index);
MFc0_std = sqrt(MFc0_var);
upper_bound = MFc0 + 2 * MFc0_std;
lower_bound = MFc0 - 2 * MFc0_std;

%% 7. Plot Multi-Panel Prediction Results
figure;
set(gcf, 'Color', 'w');
segments = {1:100, 101:200, 201:300, 301:400};

for i = 1:4
    idx = segments{i};
    subplot(2,2,i); hold on;
    fill([idx, fliplr(idx)], [upper_bound(idx)', fliplr(lower_bound(idx)')], ...
        [0.8 1 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    plot(idx, y_truth(idx), 'ko', 'MarkerSize', 4, 'DisplayName', 'Ground Truth');
    plot(idx, GP_1D_LF(idx), 'b--', 'LineWidth', 1, 'DisplayName', 'High-Fidelity GP');
    plot(idx, MFc0(idx), 'g-', 'LineWidth', 1, 'DisplayName', 'Multi-Fidelity MFc0');
    xlabel('Input'); ylabel('Output');
    title(sprintf('Segment %d: idx %d-%d', i, idx(1), idx(end)));
    grid on; set(gca, 'FontSize', 10);
    if i == 1, legend('Location', 'Best'); end
end

%% 8. Plot 3x3 Grid with Highlighted Spatial Locations
[x, y] = meshgrid(1:3, 1:3);
coords = [x(:), y(:)];
figure;
scatter(coords(:,1), coords(:,2), 100, 'k', 'filled'); hold on;
plot(random_spatial_locations(:,1), random_spatial_locations(:,2), ...
     'p', 'MarkerSize', 14, 'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'g');
xlabel('X'); ylabel('Y'); title('3x3 Grid with Highlighted Points');
axis([0.5 3.5 0.5 3.5]); grid on; axis square;
set(gca, 'xtick', 1:3, 'ytick', 1:3);

%% 9. Analyze Uncertainty vs Distance from Observed Locations
avg_dist = zeros(4,1);
avg_unc = zeros(4,1);
for i = 1:4
    idx = segments{i};
    a = unique(ModelInfo.X_L(test_index(idx),2:3))';  % Unique spatial point
    distances = sqrt(sum((random_spatial_locations - a).^2, 2));
    avg_dist(i) = mean(distances);
    avg_unc(i) = mean(MFc0_var(idx));
end

%% 10. Analyze Variance in Residuals Between Low- and High-Fidelity
residuals = long_format_W_H - long_format_W_L;
var_residual = var(residuals);
var_model=mean(v);
% The residual variance match the model variance, so the estimates is
% correct
