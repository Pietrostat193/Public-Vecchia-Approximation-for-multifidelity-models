function NLML = likelihood2Dsp_GLSmean_v2(hyp)
% likelihood2Dsp_GLSmean_v2
% Dense multi-fidelity spatio-temporal GP likelihood with GLS mean adjustment,
% where the design matrix is built INTERNALLY (like your Vecchia likelihood).
%
% Mean / design matrix construction matches your Vecchia code:
%   - If ModelInfo.GLSType == "adaptive":
%       G_L = [1, lat, lon] on X_L(:,2:3)
%       G_H = [1, lat, lon] on X_H(:,2:3)
%       Phi = blkdiag(G_L, G_H)
%   - Else (default):
%       Phi = [LF intercept, HF intercept]
%
% Objective:
%   - If ModelInfo.use_reml == true (default): REML
%   - Else: Profile ML with GLS residuals
%
% Requires:
%   global ModelInfo
%   ModelInfo.X_L, X_H, y_L, y_H, jitter
%   ModelInfo.cov_type in {'RBF','RBF_separate_rho','Matern','Mix'}
%   ModelInfo.combination in {'additive','multiplicative'}
%   kernel helpers: k1, k_matern

    global ModelInfo;
    X_L = ModelInfo.X_L;
    X_H = ModelInfo.X_H;
    y_L = ModelInfo.y_L;
    y_H = ModelInfo.y_H;
    y   = [y_L; y_H];
    jitter = ModelInfo.jitter;

    nL = size(X_L,1);
    nH = size(X_H,1);

    % -------------------------
    % 1) Hyperparameters
    % -------------------------
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    rho        = hyp(5);

    eps_LF     = exp(hyp(6));
    eps_HF     = exp(hyp(7));

    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    cov_type = ModelInfo.cov_type;

    % -------------------------
    % 2) Covariance blocks (same logic as your original likelihood2Dsp)
    % -------------------------
    switch cov_type
        case 'RBF'
            K_LL_t = k1(X_L(:,1),   X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_LL_s = k1(X_L(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

            K_LH_t = rho * k1(X_L(:,1),   X_H(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_LH_s =       k1(X_L(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

            K_HL_t = rho * k1(X_H(:,1),   X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_HL_s =       k1(X_H(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

            k_t1 = k1(X_H(:,1),   X_H(:,1),   [s_sig_LF_t, t_ell_LF]);
            k_s1 = k1(X_H(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);
            k_t2 = k1(X_H(:,1),   X_H(:,1),   [s_sig_HF_t, t_ell_HF]);
            k_s2 = k1(X_H(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);

        case 'RBF_separate_rho'
            rho_s = hyp(12);

            K_LL_t = k1(X_L(:,1),   X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_LL_s = k1(X_L(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

            K_LH_t = rho   * k1(X_L(:,1),   X_H(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_LH_s = rho_s * k1(X_L(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

            K_HL_t = rho   * k1(X_H(:,1),   X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_HL_s = rho_s * k1(X_H(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

            K_HH_t = (rho^2)   * k1(X_H(:,1),   X_H(:,1),   [s_sig_LF_t, t_ell_LF]) + k1(X_H(:,1),   X_H(:,1),   [s_sig_HF_t, t_ell_HF]);
            K_HH_s = (rho_s^2) * k1(X_H(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]) + k1(X_H(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);

        case 'Matern'
            K_LL_t = k_matern(X_L(:,1),   X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_LL_s = k_matern(X_L(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

            K_LH_t = rho * k_matern(X_L(:,1),   X_H(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_LH_s = rho * k_matern(X_L(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

            K_HL_t = rho * k_matern(X_H(:,1),   X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_HL_s = rho * k_matern(X_H(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

            K_HH_t = (rho^2) * k_matern(X_H(:,1),   X_H(:,1),   [s_sig_LF_t, t_ell_LF]) + k_matern(X_H(:,1),   X_H(:,1),   [s_sig_HF_t, t_ell_HF]);
            K_HH_s = (rho^2) * k_matern(X_H(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]) + k_matern(X_H(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);

        case 'Mix'
            K_LL_t = k1(X_L(:,1),   X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_LL_s = k_matern(X_L(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

            K_LH_t = rho * k1(X_L(:,1),   X_H(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_LH_s = rho * k_matern(X_L(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

            K_HL_t = rho * k1(X_H(:,1),   X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
            K_HL_s = rho * k_matern(X_H(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

            k_t1 = k1(X_H(:,1),   X_H(:,1),   [s_sig_LF_t, t_ell_LF]);
            k_s1 = k_matern(X_H(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);
            k_t2 = k1(X_H(:,1),   X_H(:,1),   [s_sig_HF_t, t_ell_HF]);
            k_s2 = k_matern(X_H(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);

        otherwise
            error('Invalid covariance type.');
    end

    combination = ModelInfo.combination;
    switch combination
        case 'additive'
            K_LL = K_LL_t + K_LL_s;
            K_LH = K_LH_t + K_LH_s;
            K_HL = K_HL_t + K_HL_s;

            if exist('K_HH_t','var')
                K_HH = K_HH_t + K_HH_s;
            else
                K_HH = rho^4 * (k_t1 + k_s1) + (k_t2 + k_s2);
            end

        case 'multiplicative'
            K_LL = K_LL_t .* K_LL_s;
            K_LH = K_LH_t .* K_LH_s;
            K_HL = K_HL_t .* K_HL_s;

            if exist('K_HH_t','var')
                K_HH = K_HH_t .* K_HH_s;
            else
                K_HH = rho^2 * (k_t1 .* k_s1) + (k_t2 .* k_s2);
            end

        otherwise
            error('Invalid combination structure');
    end

    % Add nugget + jitter
    K_LL = K_LL + eye(nL) * eps_LF;
    K_HH = K_HH + eye(nH) * eps_HF;

    K = [K_LL, K_LH;
         K_HL, K_HH];

    N = size(K,1);
    K = K + eye(N) * jitter;

    ModelInfo.K = K;
    ModelInfo.y = y;

    % -------------------------
    % 3) Build Phi (design matrix) internally like Vecchia
    % -------------------------
    if isfield(ModelInfo,'GLSType') && strcmp(ModelInfo.GLSType, "adaptive")
        G_L = [ones(nL,1), X_L(:,2), X_L(:,3)];
        G_H = [ones(nH,1), X_H(:,2), X_H(:,3)];
        Phi = blkdiag(G_L, G_H);
    else
        Phi = [[ones(nL,1); zeros(nH,1)], ...
               [zeros(nL,1); ones(nH,1)]];
    end
    P = size(Phi,2);
    ModelInfo.Phi = Phi;

    % -------------------------
    % 4) GLS fit using dense Cholesky
    % -------------------------
    [L, p] = chol(K,'lower');
    if p > 0
        error('Covariance matrix ill-conditioned / not SPD. Increase jitter or check kernel.');
    end

    % K^{-1} y and K^{-1} Phi
    Kinv_y   = L' \ (L \ y);
    Kinv_Phi = L' \ (L \ Phi);

    A = Phi' * Kinv_Phi;   % Phi' K^{-1} Phi
    b = Phi' * Kinv_y;     % Phi' K^{-1} y

    beta_hat = A \ b;

    r = y - Phi * beta_hat;
    Kinv_r = L' \ (L \ r);

    logdetK = 2 * sum(log(diag(L)));

    % -------------------------
    % 5) ML or REML
    % -------------------------
    use_reml = true;
    if isfield(ModelInfo,'use_reml')
        use_reml = logical(ModelInfo.use_reml);
    end

    if use_reml
        % REML: add 0.5 log|Phi'K^{-1}Phi| and use (N-P) in constant
        A = 0.5*(A + A'); % symmetrize
        jitterM = 1e-12;
        if isfield(ModelInfo,'jitterM') && ModelInfo.jitterM > 0
            jitterM = ModelInfo.jitterM;
        end
        A = A + jitterM*eye(P);

        [LA, pA] = chol(A,'lower');
        if pA > 0
            error('Phi''K^{-1}Phi not SPD (Phi rank/collinearity or numerical issues).');
        end
        logdetA = 2 * sum(log(diag(LA)));

        NLML = 0.5*(r' * Kinv_r) + 0.5*logdetK + 0.5*logdetA + 0.5*(N-P)*log(2*pi);

        ModelInfo.logdetA = logdetA;
    else
        % Profile ML
        NLML = 0.5*(r' * Kinv_r) + 0.5*logdetK + 0.5*N*log(2*pi);
    end

    % Cache
    ModelInfo.L = L;
    ModelInfo.logdetK = logdetK;
    ModelInfo.beta_hat = beta_hat;
    ModelInfo.residual = r;
    ModelInfo.alpha = Kinv_r; % alpha for residual-based GP (conditional on beta_hat)
end
