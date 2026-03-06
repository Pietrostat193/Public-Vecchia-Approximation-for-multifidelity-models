function [muH, s2H] = predict_calibratedCM3_fixed(Xstar, ModelInfo)
    V = ModelInfo.debug_vecchia;
    M = ModelInfo;
    hyp = M.hyp;
    
    % 1. Estrazione pezzi Woodbury e intercette GLS
    A = V.A; Dinv = V.D_inv; R = V.R; perm = V.perm;
    m_GLS = V.m_GLS; % [m_LF; m_HF] stimati nella likelihood
    
    % 2. Setup Helper per K^-1 (Matrix-safe per Woodbury)
    applyKinv = @(v) apply_Kinv_local(v, A, Dinv, R, perm);
    
    % 3. Rho e Covarianze q*
    rho_star = compute_rho_star(Xstar, M.X_H, hyp, M.RhoFunction);
    [qLstar, qHstar] = build_q_blocks_HF_rho(Xstar, rho_star, M); 
    qstar = [qLstar, qHstar]; % [n* x N]
    
    % 4. Predizione Media (Residui + Intercetta HF)
    % alpha_resid sono i residui (K^-1 * (y - Z*m_GLS)) già calcolati
    % Se non li hai, li ricalcoliamo: 
    Z_train = [ [ones(size(M.X_L,1),1); zeros(size(M.X_H,1),1)], ...
                [zeros(size(M.X_L,1),1); ones(size(M.X_H,1),1)] ];
    resid_alpha = applyKinv([M.y_L; M.y_H] - Z_train * m_GLS);
    
    mu0 = qstar * resid_alpha;
    muH = m_GLS(2) + mu0; % Media finale (Intercetta HF + correlazione residui)

    % 5. Predizione Varianza con termine GLS
    % --- Parte 1: s2_GP (Varianza del processo)
    kss = prior_diag_kss_HF_consistent(Xstar, rho_star, M);
    Kinv_qT = applyKinv(qstar');
    reduction = sum(qstar' .* Kinv_qT, 1)';
    s2_GP = max(kss - reduction, 0);
    
    % --- Parte 2: Incertezza GLS (L'anello mancante)
    % C_m = (Z' * K^-1 * Z)^-1  (Incertezza delle intercette)
    % Già calcolabile dai pezzi che hai
    Z_train_inv = [applyKinv(Z_train(:,1)), applyKinv(Z_train(:,2))];
    C_m = inv(Z_train' * Z_train_inv); 
    
    % z_star per HF è sempre [0, 1]
    z_star = repmat([0, 1], size(Xstar,1), 1); 
    % h = z_star - q* * K^-1 * Z
    qKinvZ = qstar * Z_train_inv;
    h = z_star - qKinvZ;
    
    % s2_GLS = diag(h * C_m * h')
    s2_GLS = sum((h * C_m) .* h, 2);
    
    % Varianza Totale
    s2H = s2_GP + s2_GLS;
end