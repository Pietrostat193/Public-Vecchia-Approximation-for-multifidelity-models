function [Y_pred1, Y_std1, Y_pred2, Y_std2, Y_pred3, Y_std3] = train_and_predict_gpr(ModelInfo)
    % Extract training data
    X_train = ModelInfo.y_L(ismember(ModelInfo.X_L, ModelInfo.X_H, 'rows')); % LF outputs at HF locations
    Y_train = ModelInfo.y_H; % HF outputs
    
    % Ensure X_train is a column vector
    X_train = X_train(:);
    % Prepare additional feature matrices
    X_train2 = [X_train, ModelInfo.X_H];
    X_train3 = ModelInfo.X_H;
    
    % Define Gaussian Process Regression model (Model 1)
    gpr_model1 = fitrgp(X_train, Y_train, ...
        'KernelFunction', 'squaredexponential', ...
        'FitMethod', 'exact', ...
        'PredictMethod', 'exact', ...
        'Standardize', true); 
    
    % Make predictions using Model 1
    X_test1 = ModelInfo.y_L; % All LF outputs
    [Y_pred1, Y_std1] = predict(gpr_model1, X_test1);
    
    % Define Gaussian Process Regression model (Model 2)
    gpr_model2 = fitrgp(X_train2, Y_train, ...
        'KernelFunction', 'squaredexponential', ...
        'FitMethod', 'exact', ...
        'PredictMethod', 'exact', ...
        'Standardize', true); 
    
    % Make predictions using Model 2
    X_test2 = [ModelInfo.y_L, ModelInfo.X_L]; % All LF outputs with locations
    [Y_pred2, Y_std2] = predict(gpr_model2, X_test2);
    
    % Define Gaussian Process Regression model (Model 3)
    gpr_model3 = fitrgp(X_train3, Y_train, ...
        'KernelFunction', 'squaredexponential', ...
        'FitMethod', 'exact', ...
        'PredictMethod', 'exact', ...
        'Standardize', true); 
    
    % Make predictions using Model 3
    X_test3 = ModelInfo.X_L; % All LF outputs
    [Y_pred3, Y_std3] = predict(gpr_model3, X_test3);
end