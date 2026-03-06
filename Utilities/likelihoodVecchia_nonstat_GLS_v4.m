function [NLML] = likelihoodVecchia_nonstat_GLS_v4(hyp)
% likelihoodVecchia_nonstat_GLS_v4
% v4 = v3 + REML correction term for GLS mean adjustment:
%   + 0.5 * log | G' K^{-1} G |  and constant uses (N-P)/2 * log(2*pi)
%
% Full version (corrected for optimization stability):
% - Uses LIGHT Vecchia indices ONLY if present AND valid for current (n,nn)
% - If indices are present but stale/mismatched, safely falls back
% - Optional: caches a fixed AMD permutation (perm_fixed) to reduce non-smoothness
% - Keeps EVERYTHING else from v3/v4:
%   * Nonstationary rho option
%   * Nested CM + Woodbury apply_Kinv_local
%   * Adaptive GLS with blkdiag option
%   * Full caching for prediction and debug_vecchia structure
%
% Expected external precompute (recommended):
%   [idxL, idxH] = precompute_vecchia_indices_for_ordering(ModelInfo.X_L,ModelInfo.X_H,ModelInfo.nn_size,ModelInfo.kernel);
%   ModelInfo.idxL_precomputed = idxL; ModelInfo.idxH_precomputed = idxH;

    global ModelInfo;

    % === 1. Configurazione ===
    X_L = ModelInfo.X_L; X_H = ModelInfo.X_H;
    y_L = ModelInfo.y_L; y_H = ModelInfo.y_H;
    y = [y_L; y_H]; N = size(y, 1);
    nL = size(X_L, 1); nH = size(X_H, 1);

    nn_size = ModelInfo.nn_size; jitter = ModelInfo.jitter;
    kernel = ModelInfo.kernel;
    MeanFunction = ModelInfo.MeanFunction; RhoFunction = ModelInfo.RhoFunction;

    % === 2. Iperparametri ===
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    rho_const  = hyp(5);
    eps_LF     = exp(hyp(6));  eps_HF = exp(hyp(7));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    % === 3. Vecchia Approximation (Light, if available and valid) ===
    canUseLight = false;
    idxL = []; idxH = [];

    if isfield(ModelInfo,'idxL_precomputed') && isfield(ModelInfo,'idxH_precomputed') ...
            && ~isempty(ModelInfo.idxL_precomputed) && ~isempty(ModelInfo.idxH_precomputed)

        idxL0 = ModelInfo.idxL_precomputed;
        idxH0 = ModelInfo.idxH_precomputed;

        % Validity checks: size and nn columns
        sizeOK = (size(idxL0,1) == nL) && (size(idxH0,1) == nH);
        colOK  = (size(idxL0,2) >= nn_size) && (size(idxH0,2) >= nn_size);

        if sizeOK && colOK
            canUseLight = true;
            idxL = idxL0(:, 1:nn_size);
            idxH = idxH0(:, 1:nn_size);
        end
    end

    if canUseLight
        result_LF = vecchia_approx_LIGHT_final(X_L, [s_sig_LF_s, s_ell_LF], [s_sig_LF_t, t_ell_LF], kernel, idxL);
        result_HF = vecchia_approx_LIGHT_final(X_H, [s_sig_HF_s, s_ell_HF], [s_sig_HF_t, t_ell_HF], kernel, idxH);
    else
        % Fallback (lento, ricalcola i vicini ad ogni iterazione)
        result_LF = vecchia_approx_space_time_corr_fast1(X_L, [s_sig_LF_s, s_ell_LF], [s_sig_LF_t, t_ell_LF], nn_size, 1e-6, kernel, 10, t_ell_LF, s_ell_LF, []);
        result_HF = vecchia_approx_space_time_corr_fast1(X_H, [s_sig_HF_s, s_ell_HF], [s_sig_HF_t, t_ell_HF], nn_size, 1e-6, kernel, 10, t_ell_HF, s_ell_HF, []);
    end

    Ki_L = result_LF.B' * result_LF.Di * result_LF.B;
    Ki_D = result_HF.B' * result_HF.Di * result_HF.B;

    % log_det_W in v3 (kept exactly)
    log_det_W = -(sum(log(diag(result_LF.Di))) + sum(log(diag(result_HF.Di))));

    % === 4. Modellazione rho_H ===
    if RhoFunction == "GP_scaled_empirical"
        [X_unique, ~, idx_back] = unique(X_H(:, 2:3), 'rows');
        n_locs = size(X_unique, 1);

        base_L = y_L; base_H = y_H;
        if MeanFunction == "GP_res"
            base_L = y_L - ModelInfo.m_x_L;
            base_H = y_H - ModelInfo.m_x_H;
        end

        rho_local = zeros(n_locs, 1);
        for i = 1:n_locs
            mask_H = (idx_back == i);
            mask_L = (X_L(:,2) == X_unique(i,1) & X_L(:,3) == X_unique(i,2));
            [~, iH, iL] = intersect(X_H(mask_H, 1), X_L(mask_L, 1));
            if length(iH) >= 2
                y_H_sub = base_H(mask_H); y_L_sub = base_L(mask_L);
                C = cov(y_H_sub(iH), y_L_sub(iL));
                rho_local(i) = C(1, 2) / (var(y_L_sub(iL)) + 1e-9);
            end
        end

        % Parametri GP rho: hyp(end-2:end)
        gprModel_rho = fitrgp(X_unique, rho_local, 'KernelFunction','ardsquaredexponential', ...
            'KernelParameters',exp(hyp(end-2:end)), 'FitMethod','none', 'Sigma',0.01);

        rho_H = predict(gprModel_rho, X_H(:, 2:3));
        rho_input = rho_H; NonStat = "T";
        ModelInfo.gprModel_rho = gprModel_rho; % Cache per predizione
    else
        rho_H = rho_const * ones(nH, 1);
        rho_input = rho_const; NonStat = "F";
    end

    % === 5. CM_nested e H matrix ===
    [A, D, D_inv] = CM_nested1(X_L, X_H, rho_input, eps_LF, eps_HF, NonStat, y_L, y_H);

    H = blkdiag(Ki_L, Ki_D) + A' * D_inv * A + speye(size(A,2)) * jitter;

    % ---- Permutation: cache fixed ordering if provided to improve smoothness ----
    if isfield(ModelInfo,'perm_fixed') && ~isempty(ModelInfo.perm_fixed) && numel(ModelInfo.perm_fixed)==size(H,1)
        perm = ModelInfo.perm_fixed;
    else
        perm = symamd(H);
        ModelInfo.perm_fixed = perm; % cache once (safe if dataset/order doesn't change)
    end

    [R, pchol] = chol(H(perm, perm));
    if pchol > 0
        error('Matrice H non definita positiva.');
    end
    log_det_H = 2 * sum(log(diag(R)));

    % === 6. GLS ADATTIVO (blkdiag) ===
    if isfield(ModelInfo, 'GLSType') && strcmp(ModelInfo.GLSType, "adaptive")
        % Trend spaziale separato: [1, Lat, Lon] per L e per H
        G_L = [ones(nL,1), X_L(:,2), X_L(:,3)];
        G_H = [ones(nH,1), X_H(:,2), X_H(:,3)];
        G_gls = blkdiag(G_L, G_H);
    else
        % GLS Standard (due intercette globali)
        G_gls = [[ones(nL,1); zeros(nH,1)], [zeros(nL,1); ones(nH,1)]];
    end

    % Risoluzione GLS Vettorizzata
    Kinv_yG = apply_Kinv_local([y, G_gls], A, D_inv, R, perm);
    Kinv_y = Kinv_yG(:, 1);
    Kinv_G = Kinv_yG(:, 2:end);

    beta_gls = (G_gls' * Kinv_G) \ (G_gls' * Kinv_y);
    SIy_tilde = Kinv_y - Kinv_G * beta_gls;
    y_tilde = y - G_gls * beta_gls;

    % === 7. NLML (REML) ===
    term1 = 0.5 * (y_tilde' * SIy_tilde);

    term2 = 0.5 * (log_det_W + log_det_H + sum(log(diag(D))));

    % REML correction: + 0.5 * log|G' K^{-1} G|
    Mmat = G_gls' * Kinv_G;
    Mmat = 0.5*(Mmat + Mmat'); % symmetrize

    % Jitter for M (configurable)
    P = size(G_gls, 2);
    jitterM = 1e-12;
    if isfield(ModelInfo, 'jitterM') && ModelInfo.jitterM > 0
        jitterM = ModelInfo.jitterM;
    end
    Mmat = Mmat + jitterM * eye(P);

    [LM, pM] = chol(Mmat, 'lower');
    if pM > 0
        error('M = G''K^{-1}G non definita positiva (Phi/G collineare o problemi numerici).');
    end
    log_det_M = 2 * sum(log(diag(LM)));
    termREML = 0.5 * log_det_M;

    term3 = 0.5 * (N - P) * log(2 * pi);

    NLML = term1 + term2 + termREML + term3;

    % === 8. CACHE COMPLETA PER PREDIZIONE ===
    ModelInfo.hyp = hyp; % Fondamentale per predict_calibratedCM3...
    ModelInfo.beta_gls = beta_gls;
    ModelInfo.SIy = SIy_tilde;
    ModelInfo.rho_H = rho_H;
    ModelInfo.A = A;
    ModelInfo.D_inv = D_inv;
    ModelInfo.R = R; % Cholesky di H(perm,perm)
    ModelInfo.perm = perm;
    ModelInfo.G_gls = G_gls;

    % Extra cache for REML
    ModelInfo.log_det_M = log_det_M;
    ModelInfo.M_GKinvG = Mmat;

    % Debug structure (used by your predictors)
    dbg.m_GLS = beta_gls;
    dbg.A = A;
    dbg.D_inv = D_inv;
    dbg.R = R;
    dbg.perm = perm;
    dbg.G_gls = G_gls;
    dbg.log_det_W = log_det_W;
    dbg.log_det_H = log_det_H;
    dbg.log_det_M = log_det_M;
    dbg.term1 = term1;
    dbg.term2 = term2;
    dbg.termREML = termREML;
    dbg.term3 = term3;
    ModelInfo.debug_vecchia = dbg;
end

% --- FUNZIONI LOCALI ---

function X = apply_Kinv_local(V, A, Dinv, R, p)
    % Risolve K^-1 * V usando Woodbury identity su sistema annidato
    DY = Dinv * V;
    RHS = A' * DY;
    RHSP = RHS(p, :);
    ZP = R \ (R' \ RHSP);
    Z = zeros(size(RHS));
    Z(p, :) = ZP;
    X = DY - Dinv * (A * Z);
end

function result = vecchia_approx_LIGHT_final(locations, hyp_s, hyp_t, kernel, n_ind_mat)
    % Versione veloce che usa indici pre-calcolati
    [n, ~] = size(locations);
    eps_val = 1e-7;
    B_rows_c = cell(n,1); B_cols_c = cell(n,1); B_vals_c = cell(n,1);
    Di_vals = zeros(n,1);

    % Diagonale della covarianza (vettorizzata)
    var_self = max(k_space_time_v2(locations, [], hyp_s, hyp_t, kernel), eps_val);

    for i = 1:n
        n_ind = n_ind_mat(i, :);
        n_ind = n_ind(n_ind > 0);

        if ~isempty(n_ind)
            xi = locations(i,:);
            Xnbrs = locations(n_ind,:);
            K_nn = k_space_time_v2(Xnbrs, Xnbrs, hyp_s, hyp_t, kernel);
            K_nn = 0.5*(K_nn + K_nn') + eps_val*eye(length(n_ind));
            K_i_n = k_space_time_v2(Xnbrs, xi, hyp_s, hyp_t, kernel);

            Ai = K_nn \ K_i_n(:);
            cond_var = max(var_self(i) - K_i_n(:)' * Ai, eps_val);

            B_rows_c{i} = repmat(i, length(n_ind), 1);
            B_cols_c{i} = n_ind(:);
            B_vals_c{i} = -Ai;
            Di_vals(i) = 1 / cond_var;
        else
            Di_vals(i) = 1 / var_self(i);
        end
    end
    result.B = sparse(vertcat(B_rows_c{:}), vertcat(B_cols_c{:}), vertcat(B_vals_c{:}), n, n) + speye(n);
    result.Di = spdiags(Di_vals, 0, n, n);
end
