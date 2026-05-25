function [muH, s2H] = predict_calibratedCM3_AdaptiveGLS_v3(Xstar, ModelInfo, opts)
    % predict_calibratedCM3_AdaptiveGLS_v3
    % Combina GLS Adattivo (6 parametri) + Calibrazione Affine + MFGP
    
    if nargin < 3, opts = struct(); end
    if ~isfield(opts,'gamma_clip'), opts.gamma_clip = [0.1, 10.0]; end % Clip per stabilità
    if ~isfield(opts,'lambda_ridge'), opts.lambda_ridge = 1e-6; end
    
    M = ModelInfo; 
    V = M.debug_vecchia; 
    hyp = M.hyp;
    nL = size(M.X_L,1); nH = size(M.X_H,1);
    nStar = size(Xstar,1);
    y_full = [M.y_L; M.y_H];
    eps_HF = exp(hyp(7)); 

    % 1. Setup K^-1 (Woodbury-Vecchia)
    applyKinv = @(v) apply_Kinv_local_internal(v, V.A, V.D_inv, V.R, V.perm);
    
    % 2. Gestione Trend GLS Adattivo (6 parametri)
    % beta = [L_intercetta, L_lat, L_lon, H_intercetta, H_lat, H_lon]
    Z_train_L = [ones(nL,1), M.X_L(:,2:3)];
    Z_train_H = [ones(nH,1), M.X_H(:,2:3)];
    Z_train   = blkdiag(Z_train_L, Z_train_H);
    
    % Trend per Xstar (solo parte High-Fidelity)
    Zstar_H   = [ones(nStar,1), Xstar(:,2:3)];
    Zstar_full = [zeros(nStar,3), Zstar_H]; 
    
    m_GLS_H = V.m_GLS(4:6); % Coefficienti del trend delle stazioni (H)

    % 3. Calcolo Residui e Predizione GP base (mu0)
    % resid_alpha = K^-1 * (y - Z*beta)
    if isfield(M, 'SIy')
        resid_alpha = M.SIy;
    else
        resid_alpha = applyKinv(y_full - Z_train * V.m_GLS);
    end
    
    % Predizione rho (stazionaria o non stazionaria)
    if isfield(M, 'gprModel_rho') && M.RhoFunction == "GP_scaled_empirical"
        rho_star = predict(M.gprModel_rho, Xstar(:, 2:3));
        rho_H_train = predict(M.gprModel_rho, M.X_H(:, 2:3));
    else
        rho_star = hyp(5) * ones(nStar, 1);
        rho_H_train = hyp(5) * ones(nH, 1);
    end

    % GP pura su Xstar
    [qLstar, qHstar] = build_q_local_internal(Xstar, rho_star, M);
    mu0 = [qLstar, qHstar] * resid_alpha;

    % 4. CALIBRAZIONE AFFINE (Il "Segreto" per l'RMSE basso)
    % Dobbiamo trovare a, b tali che: y_H ~ Trend_H + a + b * mu0_H
    [qL_H, qH_H] = build_q_local_internal(M.X_H, rho_H_train, M);
    mu0_H_train = [qL_H, qH_H] * resid_alpha;
    
    % Target: Quello che rimane dopo aver tolto il trend spaziale dai dati reali
    target_H = M.y_H - Z_train_H * m_GLS_H;
    
    % Regressione per a (offset) e b (scala)
    Xcal = [ones(nH,1), mu0_H_train];
    G_cal = Xcal'*Xcal + diag([1e-8, opts.lambda_ridge]);
    ab = G_cal \ (Xcal' * target_H);
    
    % Clipping di b per evitare esplosioni
    ab(2) = max(opts.gamma_clip(1), min(opts.gamma_clip(2), ab(2)));
    
    % Varianza residua della calibrazione (rumore non spiegato)
    s2_cal_res = mean((target_H - Xcal*ab).^2);

    % 5. MEDIA FINALE
    muH = (Zstar_H * m_GLS_H) + ab(1) + ab(2) * mu0;

    % 6. VARIANZA FINALE
    % Incertezza GP
    kss = prior_diag_local_internal(Xstar, rho_star, M);
    qstar = [qLstar, qHstar];
    
    % Riduzione varianza (approssimazione efficiente)
    reduction = zeros(nStar, 1);
    for i = 1:min(nStar, 500) % Loop limitato per velocità se nStar è enorme
        qi = qstar(i,:)';
        reduction(i) = qi' * applyKinv(qi);
    end
    if nStar > 500, reduction(501:end) = mean(reduction(1:500)); end
    
    s2_gp = max(kss - reduction, 0);

    % Incertezza stima Trend (GLS)
    KinvZ = zeros(size(Z_train));
    for j = 1:size(Z_train,2), KinvZ(:,j) = applyKinv(Z_train(:,j)); end
    C_m = inv(Z_train' * KinvZ);
    h = Zstar_full - qstar * KinvZ;
    s2_gls = sum((h * C_m) .* h, 2);

    % Varianza combinata con i pesi della calibrazione
    s2H = (ab(2)^2) * (s2_gp + s2_gls) + s2_cal_res + eps_HF;
end

% --- HELPER INTERNI ---
function x = apply_Kinv_local_internal(v, A, Dinv, R, p)
    Dy = Dinv * v;
    rhs = A' * Dy;
    rhsP = rhs(p, :);
    zP = R \ (R' \ rhsP);
    z = zeros(size(rhs)); 
    z(p, :) = zP;
    x = Dy - Dinv * (A * z);
end

function [qL, qH] = build_q_local_internal(X, rho, M)
    h = M.hyp;
    tLF = [exp(h(1)), exp(h(2))]; tHF = [exp(h(3)), exp(h(4))];
    pLF = [exp(h(8)), exp(h(9))]; pHF = [exp(h(10)), exp(h(11))];
    % Cross-covarianza
    qL_base = k_space_time(X, M.X_L, pLF, tLF, M.kernel);
    qH_base = k_space_time(X, M.X_H, pLF, tLF, M.kernel);
    qd_base = k_space_time(X, M.X_H, pHF, tHF, M.kernel);
    qL = qL_base .* rho(:); 
    qH = qH_base .* (rho(:).^2) + qd_base;
end

function kss = prior_diag_local_internal(X, rho, M)
    h = M.hyp;
    pLF = exp(h(8)); tLF = exp(h(1));
    pHF = exp(h(10)); tHF = exp(h(3));
    kL = (pLF^2) * (tLF^2);
    kD = (pHF^2) * (tHF^2);
    kss = (rho(:).^2) .* kL + kD;
end