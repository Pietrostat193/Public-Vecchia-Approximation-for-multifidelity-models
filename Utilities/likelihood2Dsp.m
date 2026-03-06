function NLML1 = likelihood2Dsp(hyp)
    
    global ModelInfo;
    X_L = ModelInfo.X_L;
    X_H = ModelInfo.X_H;
    y_L = ModelInfo.y_L;
    y_H = ModelInfo.y_H;
    y = [y_L; y_H];
    jitter = ModelInfo.jitter;

    % Low-Fidelity Temporal Parameters
    s_sig_LF_t = exp(hyp(1));  % Temporal signal variance for the LF model
    t_ell_LF = exp(hyp(2));    % Temporal length scale for the LF model
    
    % High-Fidelity Temporal Parameters
    s_sig_HF_t = exp(hyp(3));  % Temporal signal variance for the HF model
    t_ell_HF = exp(hyp(4));    % Temporal length scale for the HF model
    
    % scale-Correlation Parameter
    rho = hyp(5);              % Cross-correlation coefficient between LF and HF
    
    % Noise Parameters
    eps_LF = exp(hyp(6));      % Noise variance for the LF model
    eps_HF = exp(hyp(7));      % Noise variance for the HF model

    % Low-Fidelity Spatial Parameters
    s_sig_LF_s = exp(hyp(8));  % Spatial signal variance for the LF model
    %s_sig_LF_s = 1;  
    s_ell_LF = exp(hyp(9));    % Spatial length scale for the LF model
    
    % High-Fidelity Spatial Parameters
    s_sig_HF_s = exp(hyp(10)); % Spatial signal variance for the HF model
    %s_sig_HF_s = 1;
    s_ell_HF = exp(hyp(11));   % Spatial length scale for the HF model
 
    % Low-Fidelity Covariance Matrices
  
   % Define your choice for the covariance function
cov_type = ModelInfo.cov_type;   % Set this to 'RBF' or 'Matern' as needed

% Initialize the covariance matrices
switch cov_type
    case 'RBF'
        
        K_LL_t = k1(X_L(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        K_LL_s = k1(X_L(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);
        
        K_LH_t = rho * k1(X_L(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]);
        %K_LH_s = rho * k1(X_L(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);
        K_LH_s = k1(X_L(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

       
        K_HL_t = rho * k1(X_H(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        %K_HL_s = rho * k1(X_H(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);
        K_HL_s =  k1(X_H(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);


       % K_HH_t = (rho^2) * k1(X_H(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]) + k1(X_H(:,1), X_H(:,1), [s_sig_HF_t, t_ell_HF]);
       % K_HH_s = (rho^2) * k1(X_H(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]) + k1(X_H(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);
         
       k_t1 = k1(X_H(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]);
       k_s1 = k1(X_H(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);
       k_t2 = k1(X_H(:,1), X_H(:,1), [s_sig_HF_t, t_ell_HF]);
       k_s2 = k1(X_H(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);
       



          
    case 'RBF_separate_rho'
        
        rho_s = hyp(12);
        % Use RBF (Gaussian) kernel
        K_LL_t = k1(X_L(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        K_LL_s = k1(X_L(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);
        
        K_LH_t = rho * k1(X_L(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]);
        K_LH_s = rho_s * k1(X_L(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);
        %K_LH_t =  k1(X_L(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]);
        %K_LH_s =  k1(X_L(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

        K_HL_t = rho * k1(X_H(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        K_HL_s = rho_s * k1(X_H(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);
        %K_HL_t =  k1(X_H(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        %K_HL_s =  k1(X_H(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);


        K_HH_t = (rho^2) * k1(X_H(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]) + k1(X_H(:,1), X_H(:,1), [s_sig_HF_t, t_ell_HF]);
        K_HH_s = (rho_s^2) * k1(X_H(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]) + k1(X_H(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);
      
    case 'Matern'
        % Use Matérn kernel
        K_LL_t = k_matern(X_L(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        K_LL_s = k_matern(X_L(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

        K_LH_t = rho * k_matern(X_L(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]);
        K_LH_s = rho * k_matern(X_L(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

        K_HL_t = rho * k_matern(X_H(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        K_HL_s = rho * k_matern(X_H(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

        K_HH_t = (rho^2) * k_matern(X_H(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]) + k_matern(X_H(:,1), X_H(:,1), [s_sig_HF_t, t_ell_HF]);
        K_HH_s = (rho^2) * k_matern(X_H(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]) + k_matern(X_H(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);
    
    case 'Mix' 
        % Use RBF (Gaussian) kernel
        K_LL_t = k1(X_L(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        K_LL_s = k_matern(X_L(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);
        
        K_LH_t = rho * k1(X_L(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]);
        K_LH_s = rho * k_matern(X_L(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

        K_HL_t = rho * k1(X_H(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        K_HL_s = rho * k_matern(X_H(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

        %K_HH_t = (rho^2) * k1(X_H(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]) + k1(X_H(:,1), X_H(:,1), [s_sig_HF_t, t_ell_HF]);
        %K_HH_s = (rho^2) * k_matern(X_H(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]) + k1(X_H(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);
       
        
    otherwise
        error('Invalid covariance type. Choose either "RBF" or "Matern".');
end

combination = ModelInfo.combination; % Set this to 'additive' or 'multiplicative' as needed

% Initialize the covariance matrices
switch combination
    case 'additive'
        % Combine the covariance matrices additively
        K_LL = K_LL_t + K_LL_s;
        K_LH = K_LH_t + K_LH_s;
        %K_LL =rho*(K_LL_t + K_LL_s);
        %K_LH =rho*(K_LH_t + K_LH_s);
        K_HL = K_HL_t + K_HL_s;
        %K_HH = K_HH_t + K_HH_s;  
        K_HH = rho^4 * (k_t1 + k_s1) + (k_t2 + k_s2);
    case 'multiplicative'
        % Combine the covariance matrices multiplicatively
        K_LL = K_LL_t .* K_LL_s;  % Low-Fidelity: Multiply temporal and spatial covariance matrices
        K_LH = K_LH_t .* K_LH_s;  % Low-High Fidelity: Multiply temporal and spatial covariance matrices
        %K_LL =rho*(K_LL_t .* K_LL_s);  % Low-Fidelity: Multiply temporal and spatial covariance matrices
        %K_LH =rho* (K_LH_t .* K_LH_s);
        K_HL = K_HL_t .* K_HL_s;  % High-Low Fidelity: Multiply temporal and spatial covariance matrices
        %K_HH = K_HH_t .* K_HH_s;  % High-Fidelity: Multiply temporal and spatial covariance matrices
        K_HH = rho^2 * (k_t1 .* k_s1) + (k_t2 .* k_s2);
    otherwise
        error('Invalid combination structure');
end

  % Add noise variance to LF and HF matrices
    K_LL = K_LL + eye(size(X_L, 1)) * eps_LF;
    K_HH = K_HH + eye(size(X_H, 1)) * eps_HF;
    % Full covariance matrix
    K = [K_LL, K_LH;
         K_HL, K_HH];
    
    % Add jitter for numerical stability
    K = K + eye(size(K)) * jitter;
    ModelInfo.K = K;

   % Cholesky factorization
   [L, p] = chol(K, 'lower');
   ModelInfo.L = L;
   L_inv = inv(L);
   % Compute the inverse of W using the Cholesky factor
   W_inv = L_inv' * L_inv;
   ModelInfo.W_inv=W_inv;

    if p > 0
        fprintf(1, 'Covariance matrix is ill-conditioned\n');
    end

    % Compute alpha
    alpha = L' \ (L \ y);
    ModelObjects.alpha = alpha;

    % Compute log determinant
    ModelInfo.log_det_classic = sum(log(diag(L)));
    ModelInfo.alpha = alpha;
    ModelInfo.y=y;

    % Negative Log Marginal Likelihood (NLML)
    NLML1 = 0.5 * y' * alpha + sum(log(diag(L))) + log(2 * pi) * numel(y) / 2;
end


