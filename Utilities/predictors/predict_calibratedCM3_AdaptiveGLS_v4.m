function [muH, s2H] = predict_calibratedCM3_AdaptiveGLS_v4(Xstar, ModelInfo, opts)
    % Versione V4: Ottimizzata per Likelihood Adattiva (6 beta)
    if nargin < 3, opts = struct(); end
    if ~isfield(opts,'gamma_clip'), opts.gamma_clip = [0.05, 20.0]; end
    if ~isfield(opts,'lambda_ridge'), opts.lambda_ridge = 1e-5; end
    global ModelInfo
    M = ModelInfo; 
    V = M.debug_vecchia; 
    hyp = M.hyp;
    nL = size(M.X_L,1); nH = size(M.X_H,1);
    nStar = size(Xstar,1);
    y_full = [M.y_L; M.y_H];
    eps_HF = exp(hyp(7)); 

    % 1. Setup K^-1 (Cruciale usare la struttura salvata nel debug)
    applyKinv = @(v) apply_Kinv_local_internal(v, V.A, V.D_inv, V.R, V.perm);
    
    % 2. Mapping Trend Adattivo (6 parametri)
    % beta(1:3) -> Satellite (LF) | beta(4:6) -> Stazioni (HF)
    Z_train_L = [ones(nL,1), M.X_L(:,2:3)];
    Z_train_H = [ones(nH,1), M.X_H(:,2:3)];
    Z_train   = blkdiag(Z_train_L, Z_train_H);
    
    m_GLS_H = V.m_GLS(4:6); 
    Zstar_H = [ones(nStar,1), Xstar(:,2:3)]; % Base per la predizione HF
    
    % 3. Calcolo Residui Pesati (alpha)
    % Usiamo SIy calcolato nella likelihood (K^-1 * residui)
    if isfield(M, 'SIy')
        resid_alpha = M.SIy;
    else
        resid_alpha = applyKinv(y_full - Z_train * V.m_GLS);
    end
    
    % 4. Rho e Predizione GP (mu0)
    if isfield(M, 'gprModel_rho') && M.RhoFunction == "GP_scaled_empirical"
        rho_star = predict(M.gprModel_rho, Xstar(:, 2:3));
        rho_train_H = predict(M.gprModel_rho, M.X_H(:, 2:3));
    else
        rho_star = hyp(5) * ones(nStar, 1);
        rho_train_H = hyp(5) * ones(nH, 1);
    end
    ModelInfo.rho_star=rho_star;

    % Calcolo mu0 (parte GP pura) su Xstar e sui punti di training HF
    [qLstar, qHstar] = build_q_local_internal(Xstar, rho_star, M);
    mu0_star = [qLstar, qHstar] * resid_alpha;

    [qL_H, qH_H] = build_q_local_internal(M.X_H, rho_train_H, M);
    mu0_H_train = [qL_H, qH_H] * resid_alpha;

    % 5. Calibrazione Affine (a, b)
    % Cerchiamo di correggere i residui del trend spaziale
    target_H = M.y_H - Z_train_H * m_GLS_H;
    Xcal = [ones(nH,1), mu0_H_train];
    G_cal = Xcal'*Xcal + diag([1e-6, opts.lambda_ridge]);
    ab = G_cal \ (Xcal' * target_H);
    
    % Clip del fattore di scala per evitare instabilità
    ab(2) = max(opts.gamma_clip(1), min(opts.gamma_clip(2), ab(2)));
    
    % Errore residuo della calibrazione
    s2_cal_res = mean((target_H - Xcal*ab).^2);

    % 6. Media e Varianza Finale
    % mu = Trend_HF + Calibrazione_Offset + Calibrazione_Scale * GP
    muH = (Zstar_H * m_GLS_H) + ab(1) + ab(2) * mu0_star;

    % Varianza (GP + Incertezza stima Trend)
    kss = prior_diag_local_internal(Xstar, rho_star, M);
    qstar = [qLstar, qHstar];
    
    % Approssimazione riduzione varianza (Vecchia)
    reduction = zeros(nStar, 1);
    for i = 1:min(nStar, 1000)
        qi = qstar(i,:)';
        reduction(i) = qi' * applyKinv(qi);
    end
    if nStar > 1000, reduction(1001:end) = median(reduction(1:1000)); end
    
    % Componente incertezza sui coefficienti beta
    KinvZ = zeros(size(Z_train));
    for j = 1:size(Z_train,2), KinvZ(:,j) = applyKinv(Z_train(:,j)); end
    C_m = inv(Z_train' * KinvZ + 1e-8*eye(6));
    
    Zstar_full = [zeros(nStar,3), Zstar_H];
    h = Zstar_full - qstar * KinvZ;
    s2_gls = sum((h * C_m) .* h, 2);

    s2H = (ab(2)^2) * (max(kss - reduction, 0) + s2_gls) + s2_cal_res + eps_HF;
end

% --- Helper Functions ---
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
    % Uso k_space_time per coerenza totale con likelihood
    qL = k_space_time(X, M.X_L, pLF, tLF, M.kernel) .* rho(:); 
    qH = k_space_time(X, M.X_H, pLF, tLF, M.kernel) .* (rho(:).^2) + ...
         k_space_time(X, M.X_H, pHF, tHF, M.kernel);
end

function kss = prior_diag_local_internal(X, rho, M)
    h = M.hyp;
    % sigma_s * sigma_t per ogni fedeltà
    kL = exp(h(8))^2 * exp(h(1))^2;
    kD = exp(h(10))^2 * exp(h(3))^2;
    kss = (rho(:).^2) .* kL + kD;
end