function [muH, s2H] = predict_calibratedCM3(Xstar, arg2)
% predict_calibratedCM3
% HF prediction for nested MF-GP (CM) using Vecchia K^{-1} with calibration.
% Mean is intended to be IDENTICAL to your predictVecchia_CM_calibrated2.
% Adds predictive variance s2H (on the SAME scale as muH).
%
% USAGE:
%   [muH,s2H] = predict_calibratedCM3(Xstar)               % uses global ModelInfo
%   [muH,s2H] = predict_calibratedCM3(Xstar, ModelInfo)    % explicit model
%   [muH,s2H] = predict_calibratedCM3(Xstar, opts)         % opts + global ModelInfo
%
% Requires (available on path):
%   compute_rho_star(...), k1(...), k_matern(...), sq_dist(...)
%
% NOTE:
%   This file is SELF-CONTAINED regarding helper functions:
%   build_q_blocks_HF_rho, build_q_on_XH_rho, prior_diag_kss_HF,
%   pick_kernels, apply_Kinv_local are included below.

    % ---------- parse inputs like before ----------
    if nargin < 2 || isempty(arg2)
        opts = struct();
        global ModelInfo
        M = ModelInfo;
    else
        if isstruct(arg2) && all(isfield(arg2, {'X_L','X_H','y_L','y_H','hyp','cov_type'}))
            M = arg2;                 % ModelInfo passed
            opts = struct();
        else
            opts = arg2;              % opts passed
            global ModelInfo
            M = ModelInfo;
        end
    end

    if ~isfield(opts,'calib_mode'),    opts.calib_mode   = 'global_affine'; end
    if ~isfield(opts,'gamma_clip'),    opts.gamma_clip   = [0.25, 4.0];     end
    if ~isfield(opts,'lambda_ridge'),  opts.lambda_ridge = 1e-8;            end
    if ~isfield(opts,'gamma_subset'),  opts.gamma_subset = [];              end
    if ~isfield(opts,'seed'),          opts.seed         = 42;              end
    if ~isfield(opts,'bin_Kt'),        opts.bin_Kt       = 4;               end
    if ~isfield(opts,'bin_Ks'),        opts.bin_Ks       = 4;               end
    if ~isfield(opts,'bin_min_pts'),   opts.bin_min_pts  = 15;              end

    % ---------- pull model ----------
    assert(isfield(M,'debug_vecchia'), 'Run Vecchia likelihood first (debug_vecchia missing).');
    V = M.debug_vecchia;

    needed = {'A','D_inv','R','perm'};
    for f = needed
        assert(isfield(V,f{1}) && ~isempty(V.(f{1})), 'debug_vecchia.%s missing', f{1});
    end
    assert(all(isfield(M, {'X_L','X_H','y_L','y_H','hyp','cov_type','combination','RhoFunction'})), ...
        'ModelInfo must contain X_L, X_H, y_L, y_H, hyp, cov_type, combination, RhoFunction.');

    X_L = M.X_L; X_H = M.X_H;
    y   = [M.y_L; M.y_H];
    nL  = size(X_L,1); nH = size(X_H,1);  N = nL + nH;

    % ---------- K^{-1} apply ----------
    applyKinv = @(v) apply_Kinv_local(v, V.A, V.D_inv, V.R, V.perm);

    % alpha = K^{-1} y  (use cached SIy if available)
    if isfield(V,'SIy') && ~isempty(V.SIy) && numel(V.SIy)==N
        alpha = V.SIy;
    else
        alpha = applyKinv(y);
    end

    % ---------- GLS mean (two intercepts: LF & HF) ----------
    if ~isfield(V,'Z') || ~isfield(V,'m_GLS') || ~isfield(V,'KinvZ') ...
            || isempty(V.Z) || isempty(V.m_GLS) || isempty(V.KinvZ)
        Z = [ [ones(nL,1); zeros(nH,1)], [zeros(nL,1); ones(nH,1)] ];  % N x 2
        KinvZ = [applyKinv(Z(:,1)), applyKinv(Z(:,2))];               % N x 2
        m_GLS = (Z.'*KinvZ) \ (Z.'*alpha);                             % 2x1
    else
        Z = V.Z; KinvZ = V.KinvZ; m_GLS = V.m_GLS;
    end

    resid   = alpha - KinvZ * m_GLS;   % N x 1
    m_H_gls = m_GLS(2);

    % ---------- rho(x) (match base) ----------
    rho_star = compute_rho_star(Xstar, X_H, M.hyp, M.RhoFunction);   % n* x 1
    if isrow(rho_star), rho_star = rho_star(:); end

    rho_H    = compute_rho_star(X_H, X_H, M.hyp, M.RhoFunction);     % nH x 1
    if isrow(rho_H), rho_H = rho_H(:); end

    % ---------- prediction blocks q_* with rho(x) (IDENTICAL to base) ----------
    [qLstar, qHstar] = build_q_blocks_HF_rho(Xstar, rho_star, M);  % [n*,nL], [n*,nH]
    qstar = [qLstar, qHstar];                                      % n* x N
    mu0   = qstar * resid;                                         % n* x 1

    % ===================== PREDICTIVE VARIANCE =====================
    % IMPORTANT:
    % You observed kss - reduction < 0 everywhere. That usually indicates:
    %  - mismatch between "prior diag" and the linear form q*K^{-1}*q'
    %  - or missing noise on HF observations in the marginal variance
    %
    % In the CM nested model, the HF marginal includes:
    %   Var(y_H*) = rho^2 Var(y_L*) + Var(delta*) + eps_HF
    % and LF has eps_LF, etc.
    %
    % Your hyperparameters store eps_LF=exp(hyp(6)), eps_HF=exp(hyp(7)).
    % We add eps_HF to the predictive variance on HF.
    %
    % Additionally, for safety, we compute kss using the SAME kernel calls
    % as in q blocks (so the diag is consistent with your implementation).

    % --- compute consistent prior diag for HF at Xstar:
    kss = prior_diag_kss_HF_consistent(Xstar, rho_star, M);  % n* x 1

    % --- reduction term diag(q K^{-1} q')
    Kinv_qT   = applyKinv(qstar');                 % N x n*
    reduction = sum(qstar' .* Kinv_qT, 1)';        % n* x 1

    % --- base var
    s2_0 = kss - reduction;

    % --- clamp tiny negatives due to numerics (but keep diagnostics meaningful)
    % If your model is consistent, s2_0 should be >= 0 (up to small numeric error).
    s2_0 = max(s2_0, 0);

    % ---------- calibration fit on HF (optionally subset) (IDENTICAL mean logic) ----------
    if ~isempty(opts.gamma_subset) && opts.gamma_subset < nH
        rng(opts.seed);
        pickH = randsample(nH, opts.gamma_subset);
        [qL_H, qH_H] = build_q_on_XH_rho(M, pickH, rho_H(pickH));
        mu0_H = [qL_H, qH_H] * resid;
        yH    = M.y_H(pickH);
        XHsub = M.X_H(pickH,:);
    else
        [qL_H, qH_H] = build_q_on_XH_rho(M, [], rho_H);
        mu0_H = [qL_H, qH_H] * resid;
        yH    = M.y_H;
        XHsub = M.X_H;
    end

    % ---------- apply calibration (mean identical; variance propagated) ----------
    switch lower(string(opts.calib_mode))
        case 'global_gain'
            rH = yH - m_H_gls;
            num = (mu0_H.' * rH);
            den = (mu0_H.' * mu0_H) + opts.lambda_ridge;
            gamma = num / den;
            gamma = max(opts.gamma_clip(1), min(opts.gamma_clip(2), gamma));

            muH = m_H_gls + gamma * mu0;
            s2H = (gamma^2) * s2_0;

        case 'global_affine'
            Xcal = [ones(size(mu0_H)), mu0_H];
            G = Xcal.'*Xcal + diag([0, opts.lambda_ridge]);   % ridge on slope only
            ab = G \ (Xcal.' * yH);
            ab(2) = max(opts.gamma_clip(1), min(opts.gamma_clip(2), ab(2)));

            muH = ab(1) + ab(2) * mu0;
            s2H = (ab(2)^2) * s2_0;

        case 'per_bin_affine'
            % global fallback
            Xcal_g = [ones(size(mu0_H)), mu0_H];
            Gg = Xcal_g.'*Xcal_g + diag([0, opts.lambda_ridge]);
            ab_g = Gg \ (Xcal_g.' * yH);
            ab_g(2) = max(opts.gamma_clip(1), min(opts.gamma_clip(2), ab_g(2)));

            % bins
            tH = XHsub(:,1);  sH = XHsub(:,2:3);
            t_edges  = quantile(tH,   linspace(0,1,opts.bin_Kt+1));
            s1_edges = quantile(sH(:,1), linspace(0,1,opts.bin_Ks+1));
            s2_edges = quantile(sH(:,2), linspace(0,1,opts.bin_Ks+1));
            [~, bt]  = histc(tH, t_edges);
            [~, bs1] = histc(sH(:,1), s1_edges);
            [~, bs2] = histc(sH(:,2), s2_edges);
            bt  = max(1, min(opts.bin_Kt, bt));
            bs1 = max(1, min(opts.bin_Ks, bs1));
            bs2 = max(1, min(opts.bin_Ks, bs2));
            Ktot = opts.bin_Kt * opts.bin_Ks * opts.bin_Ks;
            ab_bins = nan(2, Ktot);
            bin_id = @(bt,bs1,bs2) (bt-1)*opts.bin_Ks*opts.bin_Ks + (bs1-1)*opts.bin_Ks + bs2;

            for kt = 1:opts.bin_Kt
                for k1i = 1:opts.bin_Ks
                    for k2i = 1:opts.bin_Ks
                        k = bin_id(kt,k1i,k2i);
                        idx = (bt==kt) & (bs1==k1i) & (bs2==k2i);
                        if sum(idx) >= opts.bin_min_pts
                            Xc = [ones(sum(idx),1), mu0_H(idx)];
                            Gk = Xc.'*Xc + diag([0, opts.lambda_ridge]);
                            abk = Gk \ (Xc.' * yH(idx));
                            abk(2) = max(opts.gamma_clip(1), min(opts.gamma_clip(2), abk(2)));
                            ab_bins(:,k) = abk;
                        else
                            ab_bins(:,k) = ab_g;
                        end
                    end
                end
            end

            % apply to test points
            tS = Xstar(:,1); sS = Xstar(:,2:3);
            [~, btt]  = histc(tS, t_edges);  btt = max(1, min(opts.bin_Kt, btt));
            [~, bs1t] = histc(sS(:,1), s1_edges); bs1t = max(1, min(opts.bin_Ks, bs1t));
            [~, bs2t] = histc(sS(:,2), s2_edges); bs2t = max(1, min(opts.bin_Ks, bs2t));

            muH = zeros(size(mu0));
            s2H = zeros(size(mu0));
            for i = 1:numel(mu0)
                k = bin_id(btt(i), bs1t(i), bs2t(i));
                abk = ab_bins(:,k);
                muH(i) = abk(1) + abk(2) * mu0(i);
                s2H(i) = (abk(2)^2) * s2_0(i);
            end

        case 'none'
            muH = m_H_gls + mu0;
            s2H = s2_0;

        otherwise
            error('Unknown opts.calib_mode: %s', string(opts.calib_mode));
    end
end

% ===================== helpers =====================

function [qLstar, qHstar] = build_q_blocks_HF_rho(Xstar, rho_star, M)
% Same structure as your base predictor: qL = rho*kL ; qH = rho^2*kL_on_H + kD
    X_L = M.X_L; X_H = M.X_H; hyp = M.hyp;
    [kt, ks] = pick_kernels(M.cov_type);
    comb = lower(string(M.combination));

    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    Kt_L  = kt(Xstar(:,1),   X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
    Ks_L  = ks(Xstar(:,2:3), X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

    Kt_HL = kt(Xstar(:,1),   X_H(:,1),   [s_sig_LF_t, t_ell_LF]);
    Ks_HL = ks(Xstar(:,2:3), X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

    Kt_d  = kt(Xstar(:,1),   X_H(:,1),   [s_sig_HF_t, t_ell_HF]);
    Ks_d  = ks(Xstar(:,2:3), X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);

    switch comb
        case 'additive'
            qL_base = (Kt_L  + Ks_L);
            qH_base = (Kt_HL + Ks_HL);
            qd_base = (Kt_d  + Ks_d);
        case 'multiplicative'
            qL_base = (Kt_L  .* Ks_L);
            qH_base = (Kt_HL .* Ks_HL);
            qd_base = (Kt_d  .* Ks_d);
        otherwise
            error('Invalid combination: %s', string(M.combination));
    end

    rho = rho_star(:);
    qLstar = qL_base .* rho;                   % n* x nL
    qHstar = qH_base .* (rho.^2) + qd_base;    % n* x nH
end

function [qL_H, qH_H] = build_q_on_XH_rho(M, pick, rho_H)
% q blocks but evaluated at X_H (for calibration fit)
    X_L = M.X_L; X_H = M.X_H; hyp = M.hyp;
    [kt, ks] = pick_kernels(M.cov_type);
    comb = lower(string(M.combination));

    if nargin < 2 || isempty(pick)
        XH_t = X_H(:,1); XH_s = X_H(:,2:3);
        rho = rho_H(:);
    else
        XH_t = X_H(pick,1); XH_s = X_H(pick,2:3);
        rho = rho_H(:);
    end

    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    Kt_L  = kt(XH_t, X_L(:,1),   [s_sig_LF_t, t_ell_LF]);
    Ks_L  = ks(XH_s, X_L(:,2:3), [s_sig_LF_s, s_ell_LF]);

    Kt_HL = kt(XH_t, X_H(:,1),   [s_sig_LF_t, t_ell_LF]);
    Ks_HL = ks(XH_s, X_H(:,2:3), [s_sig_LF_s, s_ell_LF]);

    Kt_d  = kt(XH_t, X_H(:,1),   [s_sig_HF_t, t_ell_HF]);
    Ks_d  = ks(XH_s, X_H(:,2:3), [s_sig_HF_s, s_ell_HF]);

    switch comb
        case 'additive'
            qL_base = (Kt_L  + Ks_L);
            qH_base = (Kt_HL + Ks_HL);
            qd_base = (Kt_d  + Ks_d);
        case 'multiplicative'
            qL_base = (Kt_L  .* Ks_L);
            qH_base = (Kt_HL .* Ks_HL);
            qd_base = (Kt_d  .* Ks_d);
        otherwise
            error('Invalid combination: %s', string(M.combination));
    end

    qL_H = qL_base .* rho;                  % nH x nL (or subset)
    qH_H = qH_base .* (rho.^2) + qd_base;   % nH x nH (or subset)
end

function kss = prior_diag_kss_HF_consistent(Xstar, rho_star, M)
% Consistent HF marginal diag using the SAME kernel building blocks.
% HF marginal: rho^2 * kL(x*,x*) + kD(x*,x*) + eps_HF
%
% IMPORTANT: Your kernels appear to use amplitude = exp(hyp(i)) directly
% (not squared). We mimic your existing conventions by calling kt/ks at
% identical points and taking the diagonal.
    hyp = M.hyp;
    [kt, ks] = pick_kernels(M.cov_type);
    comb = lower(string(M.combination));

    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    eps_HF     = exp(hyp(7));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    % Diagonals via self-kernel calls (robust even if kernel convention differs)
    KtL_self = diag( kt(Xstar(:,1),   Xstar(:,1),   [s_sig_LF_t, t_ell_LF]) );
    KsL_self = diag( ks(Xstar(:,2:3), Xstar(:,2:3), [s_sig_LF_s, s_ell_LF]) );

    KtD_self = diag( kt(Xstar(:,1),   Xstar(:,1),   [s_sig_HF_t, t_ell_HF]) );
    KsD_self = diag( ks(Xstar(:,2:3), Xstar(:,2:3), [s_sig_HF_s, s_ell_HF]) );

    switch comb
        case 'additive'
            kL_ss = KtL_self + KsL_self;
            kD_ss = KtD_self + KsD_self;
        case 'multiplicative'
            kL_ss = KtL_self .* KsL_self;
            kD_ss = KtD_self .* KsD_self;
        otherwise
            error('Invalid combination: %s', string(M.combination));
    end

    rho = rho_star(:);
    kss = (rho.^2) .* kL_ss + kD_ss + eps_HF;
end

function [kt, ks] = pick_kernels(cov_type)
% Same mapping as your base
    switch string(cov_type)
        case {'RBF','RBF_separate_rho'}
            kt = @k1;       ks = @k1;
        case 'Matern'
            kt = @k_matern; ks = @k_matern;
        case 'Mix'
            kt = @k1;       ks = @k_matern;
        otherwise
            error('Unknown cov_type: %s', string(cov_type));
    end
end

function x = apply_Kinv_local(v, A, Dinv, R, p)
% MATRIX-SAFE version
% K^{-1}v = D^{-1}v - D^{-1} A z,
% with H z = A' D^{-1} v and H = R'R (AMD perm).

    Dy   = Dinv * v;          % N x m
    rhs  = A.' * Dy;          % nA x m

    if ~isempty(p)
        rhsP = rhs(p, :);
        tmp  = R' \ rhsP;
        zP   = R  \ tmp;

        z = zeros(size(rhs), 'like', rhs);
        z(p, :) = zP;
    else
        tmp = R' \ rhs;
        z   = R  \ tmp;
    end

    x = Dy - Dinv * (A * z);  % N x m
end
