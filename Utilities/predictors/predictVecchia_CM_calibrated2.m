function muH = predictVecchia_CM_calibrated2(Xstar, opts)
% HF prediction for nested MF-GP (CM) using Vecchia K^{-1} with calibration.
% This version matches base predictor by using spatially varying rho(x).
%
% Requires ModelInfo fields from your Vecchia likelihood run:
%   X_L, X_H, y_L, y_H, hyp, cov_type, combination, RhoFunction
%   debug_vecchia.A, .D_inv, .R, .perm  (optionally .SIy, .Z, .KinvZ, .m_GLS cached)
%
% OPTIONS (all optional):
%   opts.calib_mode     = 'global_affine'   % 'global_gain' | 'global_affine' | 'per_bin_affine'
%   opts.gamma_clip     = [0.25, 4.0]
%   opts.lambda_ridge   = 1e-8
%   opts.gamma_subset   = []                % sample this many HF pts for fit
%   opts.seed           = 42
%   % per-bin settings (only for 'per_bin_affine'):
%   opts.bin_Kt         = 4
%   opts.bin_Ks         = 4
%   opts.bin_min_pts    = 15

    if nargin < 2, opts = struct(); end
    if ~isfield(opts,'calib_mode'),    opts.calib_mode   = 'global_affine'; end
    if ~isfield(opts,'gamma_clip'),    opts.gamma_clip   = [0.25, 4.0];     end
    if ~isfield(opts,'lambda_ridge'),  opts.lambda_ridge = 1e-8;            end
    if ~isfield(opts,'gamma_subset'),  opts.gamma_subset = [];              end
    if ~isfield(opts,'seed'),          opts.seed         = 42;              end
    if ~isfield(opts,'bin_Kt'),        opts.bin_Kt       = 4;               end
    if ~isfield(opts,'bin_Ks'),        opts.bin_Ks       = 4;               end
    if ~isfield(opts,'bin_min_pts'),   opts.bin_min_pts  = 15;              end

    % ---------- pull model ----------
    global ModelInfo
    M = ModelInfo;
    assert(isfield(M,'debug_vecchia'), 'Run Vecchia likelihood first (debug_vecchia missing).');
    V = M.debug_vecchia;
    needed = {'A','D_inv','R','perm'};
    for f = needed, assert(isfield(V,f{1}) && ~isempty(V.(f{1})), 'debug_vecchia.%s missing', f{1}); end
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
        V.SIy = alpha; M.debug_vecchia = V; ModelInfo = M; % cache
    end

    % ---------- GLS mean (two intercepts: LF & HF) ----------
    if ~isfield(V,'Z') || ~isfield(V,'m_GLS') || ~isfield(V,'KinvZ') ...
            || isempty(V.Z) || isempty(V.m_GLS) || isempty(V.KinvZ)
        Z = [ [ones(nL,1); zeros(nH,1)], [zeros(nL,1); ones(nH,1)] ];  % N x 2
        KinvZ = [applyKinv(Z(:,1)), applyKinv(Z(:,2))];                 % N x 2
        m_GLS = (Z.'*KinvZ) \ (Z.'*alpha);                              % 2x1
        V.Z = Z; V.KinvZ = KinvZ; V.m_GLS = m_GLS;
        M.debug_vecchia = V; ModelInfo = M;  
    else
        Z = V.Z; KinvZ = V.KinvZ; m_GLS = V.m_GLS;
    end
    resid = alpha - KinvZ * m_GLS;    % N x 1
    m_H_gls = m_GLS(2);

    % ---------- rho(x) (match your base model) ----------
    rho_star = compute_rho_vec(Xstar, X_H, M.hyp, M.RhoFunction);   % n* x 1
    % For calibration on HF training points, we also need rho at X_H:
    rho_H    = compute_rho_vec(X_H,  X_H, M.hyp, M.RhoFunction);    % nH x 1

    % ---------- prediction blocks q_* with rho(x) ----------
    [qLstar, qHstar] = build_q_blocks_HF_rho(Xstar, rho_star, M);  % [n*,nL], [n*,nH]
    qstar = [qLstar, qHstar];                                      % n* x N
    mu0   = qstar * resid;                                         % n* x 1

    % ---------- calibration fit on HF (optionally subset) ----------
    idxH = (nL+1):N;
    if ~isempty(opts.gamma_subset) && opts.gamma_subset < nH
        rng(opts.seed);
        pickH = randsample(nH, opts.gamma_subset);
        [qL_H, qH_H] = build_q_on_XH_rho(M, pickH, rho_H(pickH));  % (sub) nHs x nL / nHs x nH
        mu0_H = [qL_H, qH_H] * resid;                              % nHs x 1
        yH    = M.y_H(pickH);
        XHsub = M.X_H(pickH,:);
    else
        [qL_H, qH_H] = build_q_on_XH_rho(M, [], rho_H);            % nH x nL / nH x nH
        mu0_H = [qL_H, qH_H] * resid;                              % nH x 1
        yH    = M.y_H;
        XHsub = M.X_H;
    end

    switch lower(string(opts.calib_mode))
        case 'global_gain'
            rH = yH - m_H_gls;
            num = (mu0_H.' * rH);
            den = (mu0_H.' * mu0_H) + opts.lambda_ridge;
            gamma = num / den;
            gamma = max(opts.gamma_clip(1), min(opts.gamma_clip(2), gamma));
            muH = m_H_gls + gamma * mu0;

        case 'global_affine'
            Xcal = [ones(size(mu0_H)), mu0_H];
            G = Xcal.'*Xcal + diag([0, opts.lambda_ridge]);   % ridge on slope only
            ab = G \ (Xcal.' * yH);
            ab(2) = max(opts.gamma_clip(1), min(opts.gamma_clip(2), ab(2)));
            muH = ab(1) + ab(2) * mu0;
            M.pred_cal.ab = ab; M.pred_cal.mode = 'global_affine'; ModelInfo = M;

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
                for k1 = 1:opts.bin_Ks
                    for k2 = 1:opts.bin_Ks
                        k = bin_id(kt,k1,k2);
                        idx = (bt==kt) & (bs1==k1) & (bs2==k2);
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
            for i = 1:numel(mu0)
                k = bin_id(btt(i), bs1t(i), bs2t(i));
                abk = ab_bins(:,k);
                muH(i) = abk(1) + abk(2) * mu0(i);
            end

            M.pred_cal.ab_bins = ab_bins; 
            M.pred_cal.bin_edges = struct('t',t_edges,'s1',s1_edges,'s2',s2_edges);
            M.pred_cal.mode = 'per_bin_affine'; ModelInfo = M;

        otherwise
            error('Unknown opts.calib_mode: %s', string(opts.calib_mode));
    end
end

% ===================== helpers =====================

function rho_vec = compute_rho_vec(Xa, XH_all, hyp, RhoFunction)
% Wrapper to match your base call signature. Should return a column vector rho(x) for rows of Xa.
% If you already have compute_rho_star(x, X_H, hyp, RhoFunction) in scope, we just call it.
    rho_vec = compute_rho_star(Xa, XH_all, hyp, RhoFunction);
    if isrow(rho_vec), rho_vec = rho_vec(:); end
end

function [qLstar, qHstar] = build_q_blocks_HF_rho(Xstar, rho_star, M)
% Test-time cross blocks with spatially varying rho(x*):
%   qLstar(i,:) = rho(x*_i) * Cov(fL(x*_i), fL(XL))
%   qHstar(i,:) = rho(x*_i)^2 * Cov(fL(x*_i), fL(XH)) + Cov(delta(x*_i), delta(XH))
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
% Training-HF cross blocks for calibration; uses rho(x) at X_H rows.
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

function [kt, ks] = pick_kernels(cov_type)
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
% K^{-1}v = D^{-1}v - D^{-1} A z, with H z = A' D^{-1} v and H = R'R (AMD perm).
    Dy   = Dinv * v;
    rhs  = A.' * Dy;
    rhsP = rhs(p);
    tmp  = R' \ rhsP;
    zP   = R  \ tmp;
    z    = zeros(size(rhs)); z(p) = zP;
    x    = Dy - Dinv * (A * z);
end
