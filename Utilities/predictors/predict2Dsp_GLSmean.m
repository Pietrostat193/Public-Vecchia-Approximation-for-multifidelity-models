function [muH, s2H] = predict2Dsp_GLSmean(Xstar, include_noise)
% predict2Dsp_GLSmean
% Dense (exact) prediction for the MULTI-FIDELITY spatio-temporal model
% used by likelihood2Dsp_GLSmean (GLS / Universal Kriging mean adjustment).
%
% Predicts HIGH-FIDELITY (HF) output at test locations Xstar = [t, s1, s2].
%
% Model:
%   y_L(x) = f_L(x) + e_L
%   y_H(x) = rho * f_L(x) + f_D(x) + e_H
%
% Uses the SAME covariance construction as your likelihood2Dsp_GLSmean:
%   - cov_type: 'RBF', 'RBF_separate_rho', 'Matern', 'Mix'
%   - combination: 'additive' or 'multiplicative'
%   - with the same special handling you used in likelihood2Dsp for K_HH
%
% Mean adjustment (GLS / universal kriging):
%   mu(x*) = Phi_*(x*) beta + k_*^T K^{-1} (y - Phi beta)
%   var(x*) = k** - k_*^T K^{-1} k_* + (Phi_* - k_*^T K^{-1} Phi) A^{-1} (Phi_* - k_*^T K^{-1} Phi)^T
%   where A = Phi^T K^{-1} Phi
%
% Inputs:
%   Xstar : [n* x 3] test inputs (t, s1, s2) where you want HF prediction
%   include_noise : (optional) if true adds eps_HF to predictive variance (default false)
%
% Outputs:
%   muH : [n* x 1] predictive mean for HF
%   s2H : [n* x 1] predictive variance for HF
%
% Requires global ModelInfo filled by likelihood2Dsp_GLSmean, including:
%   ModelInfo.X_L, X_H, y_L, y_H
%   ModelInfo.hyp
%   ModelInfo.L (chol of K), ModelInfo.Phi, ModelInfo.beta_hat, ModelInfo.alpha
% If some fields are missing, this function recomputes the needed pieces.

    if nargin < 2, include_noise = false; end
    global ModelInfo;

    % ---- Training data ----
    X_L = ModelInfo.X_L; X_H = ModelInfo.X_H;
    y_L = ModelInfo.y_L; y_H = ModelInfo.y_H;
    y   = [y_L; y_H];
    nL  = size(X_L,1);
    nH  = size(X_H,1);
    N   = nL + nH;

    % ---- Hypers ----
    hyp = ModelInfo.hyp(:);
    rho = hyp(5);
    eps_LF = exp(hyp(6));
    eps_HF = exp(hyp(7));

    % ---- Build / fetch Phi (design) exactly like Vecchia ----
    if isfield(ModelInfo,'Phi') && ~isempty(ModelInfo.Phi) && size(ModelInfo.Phi,1)==N
        Phi = ModelInfo.Phi;
    else
        Phi = build_gls_design_like_vecchia(X_L, X_H);
        ModelInfo.Phi = Phi;
    end
    P = size(Phi,2);

    % ---- Build / fetch chol(K) ----
    if isfield(ModelInfo,'L') && ~isempty(ModelInfo.L) && size(ModelInfo.L,1)==N
        L = ModelInfo.L;
    else
        % If user didn't call likelihood first, build K and factorize now
        K = build_full_K(X_L, X_H, hyp, ModelInfo);
        jitter = ModelInfo.jitter;
        K = K + eye(N)*jitter;
        [L, p] = chol(K,'lower');
        if p>0, error('K not SPD in prediction. Increase jitter or check hypers.'); end
        ModelInfo.K = K;
        ModelInfo.L = L;
    end

    % ---- Compute / fetch beta and alpha = K^{-1}(y - Phi beta) ----
    have_beta  = isfield(ModelInfo,'beta_hat') && ~isempty(ModelInfo.beta_hat);
    have_alpha = isfield(ModelInfo,'alpha') && ~isempty(ModelInfo.alpha);

    if ~(have_beta && have_alpha)
        Kinv_y   = L' \ (L \ y);
        Kinv_Phi = L' \ (L \ Phi);
        A = Phi' * Kinv_Phi;
        b = Phi' * Kinv_y;
        beta_hat = A \ b;
        r = y - Phi*beta_hat;
        alpha = L' \ (L \ r);
        ModelInfo.beta_hat = beta_hat;
        ModelInfo.alpha = alpha;
        ModelInfo.A_gls = A; % optional cache
        ModelInfo.Kinv_Phi = Kinv_Phi; % optional cache
    end

    beta_hat = ModelInfo.beta_hat(:);
    alpha    = ModelInfo.alpha(:);

    % ---- Cross-covariances: k_* between HF(x*) and training [LF; HF] ----
    % k_star is [n* x N]
    k_star_L = cov_Hstar_Ltrain(Xstar, X_L, hyp, ModelInfo);    % [n* x nL]
    k_star_H = cov_Hstar_Htrain(Xstar, X_H, hyp, ModelInfo);    % [n* x nH]
    k_star   = [k_star_L, k_star_H];

    % ---- Mean design at test for HF ----
    Phi_star = build_gls_design_test_HF(Xstar, X_L, X_H); % [n* x P], consistent with training Phi

    % ---- Predictive mean ----
    muH = Phi_star * beta_hat + k_star * alpha;

    % ---- Predictive variance ----
    % base reduction: k_*^T K^{-1} k_*
    V = (L \ k_star.').^2;              % [N x n*], elementwise square
    red = sum(V, 1).';                  % [n* x 1]

    % prior variance of HF at x*
    k_ss = prior_var_HF(Xstar, hyp, ModelInfo); % [n* x 1]

    s2H = max(k_ss - red, 0);

    % universal kriging/GLS variance term:
    % w = Phi_* - k_* K^{-1} Phi
    if isfield(ModelInfo,'Kinv_Phi') && ~isempty(ModelInfo.Kinv_Phi) && size(ModelInfo.Kinv_Phi,1)==N
        Kinv_Phi = ModelInfo.Kinv_Phi;
    else
        Kinv_Phi = L' \ (L \ Phi);
        ModelInfo.Kinv_Phi = Kinv_Phi;
    end
    w = Phi_star - k_star * Kinv_Phi; % [n* x P]

    if isfield(ModelInfo,'A_gls') && ~isempty(ModelInfo.A_gls) && all(size(ModelInfo.A_gls)==[P,P])
        A = ModelInfo.A_gls;
    else
        A = Phi' * Kinv_Phi;
        ModelInfo.A_gls = A;
    end

    A = 0.5*(A + A'); % symmetrize
    jitterM = 1e-12;
    if isfield(ModelInfo,'jitterM') && ModelInfo.jitterM > 0
        jitterM = ModelInfo.jitterM;
    end
    A = A + jitterM*eye(P);

    [LA, pA] = chol(A,'lower');
    if pA>0
        error('Phi''K^{-1}Phi not SPD in prediction. Check Phi rank or increase jitterM.');
    end

    tmp = LA \ w.';           % [P x n*]
    s2_gls = sum(tmp.^2, 1).'; % [n* x 1]

    s2H = s2H + s2_gls;

    if include_noise
        s2H = s2H + eps_HF;
    end
end

% ======================================================================
% Design matrices (same structure as Vecchia GLS)
% ======================================================================

function Phi = build_gls_design_like_vecchia(X_L, X_H)
    global ModelInfo
    nL = size(X_L,1);
    nH = size(X_H,1);

    if isfield(ModelInfo,'GLSType') && strcmp(ModelInfo.GLSType, "adaptive")
        G_L = [ones(nL,1), X_L(:,2), X_L(:,3)];
        G_H = [ones(nH,1), X_H(:,2), X_H(:,3)];
        Phi = blkdiag(G_L, G_H);
    else
        Phi = [[ones(nL,1); zeros(nH,1)], ...
               [zeros(nL,1); ones(nH,1)]];
    end
end

function Phi_star = build_gls_design_test_HF(Xstar, X_L, X_H)
    global ModelInfo
    nstar = size(Xstar,1);
    nL = size(X_L,1);
    nH = size(X_H,1); %#ok<NASGU>

    if isfield(ModelInfo,'GLSType') && strcmp(ModelInfo.GLSType, "adaptive")
        % Training Phi = blkdiag([1,lat,lon]_L, [1,lat,lon]_H) => P=6
        % For HF prediction, LF block columns are zeros, HF block is [1,lat,lon]
        Phi_star = [zeros(nstar,3), ones(nstar,1), Xstar(:,2), Xstar(:,3)];
    else
        % Training Phi has two intercepts [LF, HF] => for HF prediction: [0, 1]
        Phi_star = [zeros(nstar,1), ones(nstar,1)];
    end
end

% ======================================================================
% Full K builder (only used if L not already in ModelInfo)
% ======================================================================

function K = build_full_K(X_L, X_H, hyp, M)
    % This mirrors your likelihood2Dsp block logic.
    nL = size(X_L,1); nH = size(X_H,1);
    rho = hyp(5);
    eps_LF = exp(hyp(6));
    eps_HF = exp(hyp(7));

    % unpack
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    cov_type = M.cov_type;
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
            error('Invalid cov_type in ModelInfo.');
    end

    combination = M.combination;
    switch combination
        case 'additive'
            K_LL = K_LL_t + K_LL_s;
            K_LH = K_LH_t + K_LH_s;
            K_HL = K_HL_t + K_HL_s;

            if exist('K_HH_t','var')
                K_HH = K_HH_t + K_HH_s;
            else
                % match your original RBF/Mix special HH handling
                K_HH = rho^4 * (k_t1 + k_s1) + (k_t2 + k_s2);
            end

        case 'multiplicative'
            K_LL = K_LL_t .* K_LL_s;
            K_LH = K_LH_t .* K_LH_s;
            K_HL = K_HL_t .* K_HL_s;

            if exist('K_HH_t','var')
                K_HH = K_HH_t .* K_HH_s;
            else
                % match your original RBF/Mix special HH handling
                K_HH = rho^2 * (k_t1 .* k_s1) + (k_t2 .* k_s2);
            end
        otherwise
            error('Invalid combination in ModelInfo.');
    end

    K_LL = K_LL + eye(nL)*eps_LF;
    K_HH = K_HH + eye(nH)*eps_HF;

    K = [K_LL, K_LH;
         K_HL, K_HH];
end

% ======================================================================
% Cross-covariances needed for prediction
% ======================================================================

function KHL = cov_Hstar_Ltrain(Xstar, X_L, hyp, M)
    % Cov( y_H(x*), y_L(x) ) = Cov( rho f_L(x*) + f_D(x*), f_L(x) ) = rho * Cov(f_L(x*), f_L(x))
    rho = hyp(5);

    % LF kernels
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));

    cov_type = M.cov_type;
    if strcmp(cov_type,'RBF') || strcmp(cov_type,'RBF_separate_rho') || strcmp(cov_type,'Mix')
        kt = k1(Xstar(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        if strcmp(cov_type,'Mix')
            ks = k_matern(Xstar(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);
        else
            ks = k1(Xstar(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);
        end
    else
        kt = k_matern(Xstar(:,1), X_L(:,1), [s_sig_LF_t, t_ell_LF]);
        ks = k_matern(Xstar(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);
    end

    switch M.combination
        case 'additive'
            base = kt + ks;
        case 'multiplicative'
            base = kt .* ks;
        otherwise
            error('Invalid combination.');
    end

    % In your likelihood, some cases scale spatial cross differently;
    % for LF-HF cross we follow your K_LH_t and K_LH_s construction:
    if strcmp(cov_type,'RBF')
        % K_LH_t scaled by rho, K_LH_s not scaled, then combined
        % => base = (rho*kt) (+/*) (ks)
        switch M.combination
            case 'additive'
                KHL = (rho*kt) + ks;
            case 'multiplicative'
                KHL = (rho*kt) .* ks;
        end
    elseif strcmp(cov_type,'RBF_separate_rho')
        rho_s = hyp(12);
        switch M.combination
            case 'additive'
                KHL = (rho*kt) + (rho_s*ks);
            case 'multiplicative'
                KHL = (rho*kt) .* (rho_s*ks);
        end
    else
        % Matern and Mix use rho on both blocks in your code (for Mix you used rho on LH_s too)
        switch M.combination
            case 'additive'
                KHL = rho*(kt + ks);
            case 'multiplicative'
                KHL = rho*(kt .* ks);
        end
    end
end

function KHH = cov_Hstar_Htrain(Xstar, X_H, hyp, M)
    % Cov( y_H(x*), y_H(x) ) = rho^2 Cov(f_L(x*),f_L(x)) + Cov(f_D(x*), f_D(x))
    rho = hyp(5);

    % LF kernels
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));

    % HF discrepancy kernels
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    cov_type = M.cov_type;

    % pick kernel handles for LF and HF spatial parts (Mix differs)
    if contains(string(cov_type),'RBF')
        k_t = @k1;
    else
        k_t = @k_matern;
    end

    if strcmp(cov_type,'Mix')
        k_s_LF = @k_matern;
        k_s_HF = @k_matern; % discrepancy spatial you used k1 in some places; keep matern for consistency with your Mix block above
    else
        if contains(string(cov_type),'RBF')
            k_s_LF = @k1;
            k_s_HF = @k1;
        else
            k_s_LF = @k_matern;
            k_s_HF = @k_matern;
        end
    end

    kt1 = k_t(Xstar(:,1), X_H(:,1), [s_sig_LF_t, t_ell_LF]);
    ks1 = k_s_LF(Xstar(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

    kt2 = k_t(Xstar(:,1), X_H(:,1), [s_sig_HF_t, t_ell_HF]);
    ks2 = k_s_HF(Xstar(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);

    switch M.combination
        case 'additive'
            % match your special HH handling for RBF/Mix:
            if strcmp(cov_type,'RBF') || strcmp(cov_type,'Mix')
                KHH = rho^4 * (kt1 + ks1) + (kt2 + ks2);
            else
                KHH = (rho^2)*(kt1 + ks1) + (kt2 + ks2);
            end
        case 'multiplicative'
            if strcmp(cov_type,'RBF') || strcmp(cov_type,'Mix')
                KHH = rho^2 * (kt1 .* ks1) + (kt2 .* ks2);
            else
                KHH = (rho^2)*(kt1 .* ks1) + (kt2 .* ks2);
            end
        otherwise
            error('Invalid combination.');
    end
end

function kss = prior_var_HF(Xstar, hyp, M)
    rho = hyp(5);

    % LF diagonal
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));

    % HF diag
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    cov_type = M.cov_type;

    if contains(string(cov_type),'RBF')
        k_t = @k1;
    else
        k_t = @k_matern;
    end

    if strcmp(cov_type,'Mix')
        k_s_LF = @k_matern;
        k_s_HF = @k_matern;
    else
        if contains(string(cov_type),'RBF')
            k_s_LF = @k1;
            k_s_HF = @k1;
        else
            k_s_LF = @k_matern;
            k_s_HF = @k_matern;
        end
    end

    kL_t = diag(k_t(Xstar(:,1), Xstar(:,1), [s_sig_LF_t, t_ell_LF]));
    kL_s = diag(k_s_LF(Xstar(:,2:3), Xstar(:,2:3), [s_sig_LF_s, s_ell_LF]));
    kD_t = diag(k_t(Xstar(:,1), Xstar(:,1), [s_sig_HF_t, t_ell_HF]));
    kD_s = diag(k_s_HF(Xstar(:,2:3), Xstar(:,2:3), [s_sig_HF_s, s_ell_HF]));

    switch M.combination
        case 'additive'
            if strcmp(cov_type,'RBF') || strcmp(cov_type,'Mix')
                kss = rho^4 * (kL_t + kL_s) + (kD_t + kD_s);
            else
                kss = (rho^2) * (kL_t + kL_s) + (kD_t + kD_s);
            end
        case 'multiplicative'
            kss = (rho^2) * (kL_t .* kL_s) + (kD_t .* kD_s);
        otherwise
            error('Invalid combination.');
    end
end
