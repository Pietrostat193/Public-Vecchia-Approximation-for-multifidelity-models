function [NLML] = likelihoodVecchia_nonstat(hyp)
    global ModelInfo;

    % === Configuration ===
    usePermutation = true;
    UsePenalty= false;

    % === Extract Model Information ===
    X_L = ModelInfo.X_L;
    X_H = ModelInfo.X_H;
    y_L = ModelInfo.y_L;
    y_H = ModelInfo.y_H;
    y = [y_L; y_H];
    jitter = ModelInfo.jitter;
    N = size(y, 1);

    % === Extract Hyperparameters ===
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    rho = hyp(5);  eps_LF = exp(hyp(6));  eps_HF = exp(hyp(7));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));
    nn_size = ModelInfo.nn_size;

    %n_processor=8;
    %persistent parallelResource
    %if isempty(parallelResource)
    %    totalWorkers = parcluster('local').NumWorkers;
    %    numWorkers = max(1, totalWorkers-n_processor);
    %    pool = gcp('nocreate');
    %    if isempty(pool) || pool.NumWorkers ~= numWorkers
    %        delete(pool);
    %        parpool('local', numWorkers);
    %    end
    %    parallelResource = true;
    %end

  
    % === Covariance Approximations ===
    conditioning = ModelInfo.conditioning;
    kernel = ModelInfo.kernel;

    switch conditioning
        case "MinMax"
            result_LF = vecchia_approx_space_time_optimized(X_L, [s_sig_LF_s, s_ell_LF], [s_sig_LF_t, t_ell_LF], nn_size, jitter, kernel);
        case "Corr"
            result_LF = vecchia_approx_space_time_corr1(X_L, [s_sig_LF_s, s_ell_LF], [s_sig_LF_t, t_ell_LF], nn_size, jitter, kernel);
    end

    Di_L_sparse = result_LF.Di;
    log_det_K_L = sum(log(diag(Di_L_sparse)));
    Ki_L = result_LF.B' * Di_L_sparse * result_LF.B;

    switch conditioning
        case "MinMax"
            result_HF = vecchia_approx_space_time_optimized(X_H, [s_sig_HF_s, s_ell_HF], [s_sig_HF_t, t_ell_HF], nn_size, jitter, kernel);
        case "Corr"
            result_HF = vecchia_approx_space_time_corr1(X_H, [s_sig_HF_s, s_ell_HF], [s_sig_HF_t, t_ell_HF], nn_size, jitter, kernel);
    end

    Di_D_sparse = result_HF.Di;
    log_det_K_D = sum(log(diag(Di_D_sparse)));
    Ki_D = result_HF.B' * Di_D_sparse * result_HF.B;

    log_det_W = -(log_det_K_D + log_det_K_L);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % === Mean Function ===
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    MeanFunction = ModelInfo.MeanFunction;
    switch MeanFunction
        case "zero"
            m_x = 0;
        case "constant"
            m_x = hyp(12) * ones(size(y));
        case "linear"
            X_combined = [X_L; X_H];
            weights = hyp(12:14);
            bias = hyp(15);
            m_x = X_combined * weights + bias;
        case "GP"
    % Extract hyperparameters from hyp
log_sigma = hyp(12);           % log of signal std dev
log_ell   = hyp(13:15);           % log of length scale
% Convert to natural scale for kernel
sigmaF = exp(log_sigma);       % signal std dev
lengthScale = exp(log_ell);    % length scale
% Combine training data
X_combined = [X_L; X_H];
y_combined = [y_L; y_H];       % Make sure this is defined earlier

% Fit GPR model with fixed hyperparameters (no optimization)
gprModel_forH = fitrgp(X_H, y_H, ...
    'KernelFunction', 'ardsquaredexponential', ...
    'KernelParameters', [lengthScale; sigmaF], ...
    'BasisFunction', 'none', ...
    'FitMethod', 'none', ...
    'PredictMethod', 'exact', ...
    'Sigma', 0.01, ...
    'Standardize', false);


ModelInfo.gprModel_mean=gprModel;
% Store and use GP mean prediction
ModelInfo.X_combined = X_combined;
m_x = predict(gprModel, X_combined);   % GP mean as mean function

    case "GP_res"
        
 % === Extract hyperparameters from hyp ===
log_sigma_L = hyp(12);           % log of signal std dev
log_ell_L   = hyp(13:14);        % log of length scale

log_sigma_H = hyp(15);   
log_ell_H   = hyp(16:17);  

% === Convert to natural scale for kernel ===
sigmaF_L = exp(log_sigma_L);       % signal std dev
lengthScale_L = exp(log_ell_L);    % length scale

sigmaF_H = exp(log_sigma_H);       % signal std dev
lengthScale_H = exp(log_ell_H);    % length scale

% === Reduce to unique spatial locations and average target values ===
[unique_X_L, ~, idx_L] = unique(X_L(:,2:3), 'rows');
[unique_X_H, ~, idx_H] = unique(X_H(:,2:3), 'rows');

y_L_avg = accumarray(idx_L, y_L, [], @mean);
y_H_avg = accumarray(idx_H, y_H, [], @mean);

% === Fit GPR models on unique data ===
gprModel_mean_L = fitrgp(unique_X_L, y_L_avg, ...
    'KernelFunction', 'ardsquaredexponential', ...
    'KernelParameters', [lengthScale_L; sigmaF_L], ...
    'BasisFunction', 'none', ...
    'FitMethod', 'none', ...
    'PredictMethod', 'exact', ...
    'Sigma', 0.01, ...
    'Standardize', false);

gprModel_mean_H = fitrgp(unique_X_H, y_H_avg, ...
    'KernelFunction', 'ardsquaredexponential', ...
    'KernelParameters', [lengthScale_H; sigmaF_H], ...
    'BasisFunction', 'none', ...
    'FitMethod', 'none', ...
    'PredictMethod', 'exact', ...
    'Sigma', 0.01, ...
    'Standardize', false);

% === Predict at unique locations ===
m_x_L_unique = predict(gprModel_mean_L, unique_X_L);
m_x_H_unique = predict(gprModel_mean_H, unique_X_H);

% === Expand predictions back to full dataset size ===
m_x_L = m_x_L_unique(idx_L);
m_x_H = m_x_H_unique(idx_H);

% === Save model info ===
ModelInfo.gprModel_mean_L = gprModel_mean_L;
ModelInfo.gprModel_mean_H = gprModel_mean_H;
ModelInfo.X_combined = X_H;  % optional; depends on downstream use
ModelInfo.m_x_L = m_x_L;
ModelInfo.m_x_H = m_x_H;



    case "GP_for_H"
            % Extract hyperparameters from hyp
log_sigma = hyp(12);           % log of signal std dev
log_ell   = hyp(13:15);           % log of length scale
% Convert to natural scale for kernel
sigmaF = exp(log_sigma);       % signal std dev
lengthScale = exp(log_ell);    % length scale
% Combine training data
     % Make sure this is defined earlier
% Fit GPR model with fixed hyperparameters (no optimization)
gprModel_forH = fitrgp(X_H(:,2:3), y_H, ...
    'KernelFunction', 'ardsquaredexponential', ...
    'KernelParameters', [lengthScale; sigmaF], ...
    'BasisFunction', 'none', ...
    'FitMethod', 'none', ...
    'PredictMethod', 'exact', ...
    'Sigma', 0.01, ...
    'Standardize', false);
  

ModelInfo.gprModel_forH=gprModel_forH;
m_x_H = predict(gprModel_forH, X_H(:,2:3));   
ModelIn
        otherwise
            error('Invalid MeanFunction type.');
    end

    if MeanFunction=="GP_for_H"
        res=y_H-m_x_H;
        y_tilde=[y_L;res];
    elseif MeanFunction=="GP_res"
        y_L_tilde=y_L-m_x_L;
        y_H_tilde=y_H-m_x_H;

        y_tilde=[y_L_tilde;y_H_tilde];
    else
    y_tilde = y - m_x;
    ModelInfo.y_tilde=y_tilde;
    ModelInfo.m_x = m_x;
    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % === rho_H Nonstationary Modelling ===
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ===  ===
    RhoFunction = ModelInfo.RhoFunction;
    switch RhoFunction
        case "constant"
            rho_H = rho; NonStat = "F";
        case "linear"
            phi_x = @(x) [ones(size(x, 1), 1), x];
            Phi_H = phi_x(X_H(:, 2:3));
            
            if MeanFunction == "zero"
    beta_rho = exp(hyp(12:end));
elseif MeanFunction == "constant"
    beta_rho = exp(hyp(13:end));
elseif MeanFunction == "linear"
    beta_rho = exp(hyp(14:end)); 
elseif MeanFunction == "GP"
    beta_rho = exp(hyp(15:end)); 
elseif MeanFunction == "GP_for_H"
    beta_rho = exp(hyp(15:end));
  elseif MeanFunction == "GP_res"
    beta_rho = exp(hyp(18:end));
    %beta_rho = exp(hyp(15:end));
else
    error("Unknown MeanFunction: " + MeanFunction);
end

            ModelInfo.beta_rho=beta_rho;
            rho_H = Phi_H * beta_rho; NonStat = "T";
            ModelInfo.rho_H=rho_H;
        case "polynomial"
            lat = X_H(:, 2); lon = X_H(:, 3);
            lat_norm = (lat - min(lat)) / (max(lat) - min(lat));
            lon_norm = (lon - min(lon)) / (max(lon) - min(lon));
            Phi_H = [ones(size(lat)), lat_norm, lon_norm, lat_norm.^2, lon_norm.^2];
            
if MeanFunction == "zero"
    beta_rho = exp(hyp(12:end));
elseif MeanFunction == "constant"
    beta_rho = exp(hyp(13:end));
elseif MeanFunction == "linear"
    beta_rho = exp(hyp(16:end)); 
elseif MeanFunction == "GP"
    beta_rho = exp(hyp(15:end)); 
elseif MeanFunction == "GP_res"
    beta_rho = exp(hyp(18:end));
else
    error("Unknown MeanFunction: " + MeanFunction);
end


            ModelInfo.beta_rho=beta_rho;
            rho_H = Phi_H * beta_rho; NonStat = "T";
                ModelInfo.rho_H=rho_H;
        case "GP_scaled"
            [X_unique, ~, idx_back] = unique(X_H(:, 2:3), 'rows');
            sigma = exp(hyp(end - 1)); ell = exp(hyp(end));
            dists = pdist2(X_unique, X_unique).^2;
            K_unique = sigma^2 * exp(-0.5 * dists / ell^2) + 1e-6 * eye(size(dists));
            rho_unique = abs(mean(K_unique, 2));
            rho_H = rho_unique(idx_back); NonStat = "T";
            ModelInfo.rho_H_unique = rho_unique;
            ModelInfo.X_H_unique = X_unique;
            ModelInfo.K_rho_unique = K_unique;

        case "GP_scaled_empirical"
    
    % Use externally computed fixed kernel parameters
    log_sigma = hyp(end - 2);
    log_ell   = [hyp(end-1);hyp(end)];
   
    % Get spatial coordinates (unique)
    [X_unique, ~, idx_back] = unique(X_H(:, 2:3), 'rows');
    n_locs = size(X_unique, 1);
    rho_local = zeros(n_locs, 1);

    for i = 1:n_locs
        coord = X_unique(i, :);
        idx_H = ismember(X_H(:, 2:3), coord, 'rows');
        idx_L = ismember(X_L(:, 2:3), coord, 'rows');

        if MeanFunction=="GP_res"
        y_H_i=y_H_tilde(idx_H);
        y_L_i=y_L_tilde(idx_L);
        else
        y_H_i = y_H(idx_H);
        y_L_i = y_L(idx_L);
        end
        
       

        t_H = X_H(idx_H, 1);
        t_L = X_L(idx_L, 1);
        [common_t, iH, iL] = intersect(t_H, t_L);

        y_H_aligned = y_H_i(iH);
        y_L_aligned = y_L_i(iL);

        if length(y_L_aligned) >= 2 && var(y_L_aligned, 1) > 0
            C = cov(y_H_aligned, y_L_aligned, 1);
            rho_local(i) = C(1, 2) / var(y_L_aligned, 1);
        else
            rho_local(i) = 0;
        end
    end

theta = exp([log_sigma; log_ell]);

gprModel_rho = fitrgp(X_unique, rho_local, ...
    'KernelFunction', 'ardsquaredexponential', ...
    'KernelParameters', theta, ...
    'BasisFunction', 'none', ...
    'FitMethod', 'none', ...
    'PredictMethod', 'exact', ...
    'Sigma', 0.01, ...
    'Standardize', false);

    % Predict smoothed rho_H at X_H
    rho_H = predict(gprModel_rho, X_H(:, 2:3));

    % Store
    NonStat = "T";
    ModelInfo.gprModel_rho=gprModel_rho;
    ModelInfo.rho_H = rho_H;
    ModelInfo.rho_local = rho_local;
    ModelInfo.rho_H_unique = predict(gprModel_rho, X_unique);
    ModelInfo.X_H_unique = X_unique;
end
   
   

    % === Build Nested Model Matrices ===
    [A, D, D_inv, Z21, Z1] = CM_nested1(X_L, X_H, rho_H, eps_LF, eps_HF, NonStat, y_L, y_H);

    % === Build H and Solve ===
    H = blkdiag(Ki_L, Ki_D) + A' * D_inv * A + speye(size(Ki_L, 1) + size(Ki_D, 1)) * jitter;
    ModelInfo.H = H;

    Dy = D_inv * y_tilde;

    if usePermutation
        perm = symamd(H);
        H_perm = H(perm, perm);
        [R_perm, p] = chol(H_perm);
        if p > 0, error('Permuted H not positive definite'); end
        ModelInfo.perm = perm;
        ModelInfo.L = R_perm;
        log_det_H = 2 * sum(log(diag(R_perm)));
        Dy_perm = Dy(perm);
        AtDy_perm = A' * Dy;
        AtDy_perm = AtDy_perm(perm);
        temp = R_perm' \ AtDy_perm;
        H_ADy_perm = R_perm \ temp;
        H_ADy = zeros(size(H_ADy_perm));
        H_ADy(perm) = H_ADy_perm;
    else
        [R, p] = chol(H);
        if p > 0, error('H not positive definite'); end
        ModelInfo.L = R;
        log_det_H = 2 * sum(log(diag(R)));
        H_ADy = R \ (R' \ (A' * Dy));
    end

    SIy = Dy - D_inv * (A * H_ADy);
    ModelInfo.SIy = SIy;
    ModelInfo.log_det_H = log_det_H;
    ModelInfo.A = A;
    ModelInfo.D_inv = D_inv;

    % === Final Likelihood ===
    log_det_D = sum(log(diag(D)));
    term1 = 0.5 * (y_tilde' * SIy);
    term2 = 0.5 * (log_det_W + log_det_H + log_det_D);
    term3 = 0.5 * N * log(2 * pi);

    NLML = term1 + term2 + term3;

    if UsePenalty
    ModelInfo.hyp=hyp;
    [mean_pred]=predictVecchia_nonstat2(X_H);
   
    % Penalize if any predicted mean is negative
    if any(mean_pred < 0)
        penalty = 1e4 * sum(abs(mean_pred(mean_pred < 0)));
        NLML = NLML + penalty;
    end
   end
end
