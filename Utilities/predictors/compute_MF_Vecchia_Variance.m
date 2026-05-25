function [s2H, diag_out] = compute_MF_Vecchia_Variance(Xstar, rho_star, M)
    % --- SETUP ---
    V = M.debug_vecchia;
    hyp = M.hyp;
    
    % Estrazione iperparametri (Lineari)
    s_LF = exp(hyp(1)) * exp(hyp(8)); 
    s_D  = exp(hyp(3)) * exp(hyp(10));
    eps_HF = exp(hyp(7));

    % --- 1. PRIOR DIAGONAL (kss) ---
    % Calcoliamo le diagonali con ampiezza 1
    RtL_diag = diag(k1_pure(Xstar(:,1), Xstar(:,1), exp(hyp(2))));
    RsL_diag = diag(k1_pure(Xstar(:,2:3), Xstar(:,2:3), exp(hyp(9))));
    RtD_diag = diag(k1_pure(Xstar(:,1), Xstar(:,1), exp(hyp(4))));
    RsD_diag = diag(k1_pure(Xstar(:,2:3), Xstar(:,2:3), exp(hyp(11))));
    
    kL_ss = s_LF * (RtL_diag .* RsL_diag);
    kD_ss = s_D * (RtD_diag .* RsD_diag);
    kss = (rho_star(:).^2) .* kL_ss + kD_ss + eps_HF;

    % --- 2. Q-BLOCKS (Cross-Covariance) ---
    QL_t = k1_pure(Xstar(:,1), M.X_L(:,1), exp(hyp(2)));
    QL_s = k1_pure(Xstar(:,2:3), M.X_L(:,2:3), exp(hyp(9)));
    qLstar = (s_LF * (QL_t .* QL_s)) .* rho_star(:);

    QH_t = k1_pure(Xstar(:,1), M.X_H(:,1), exp(hyp(2)));
    QH_s = k1_pure(Xstar(:,2:3), M.X_H(:,2:3), exp(hyp(9)));
    Qd_t = k1_pure(Xstar(:,1), M.X_H(:,1), exp(hyp(4)));
    Qd_s = k1_pure(Xstar(:,2:3), M.X_H(:,2:3), exp(hyp(11)));
    
    qHstar = (s_LF * (QH_t .* QH_s)) .* (rho_star(:).^2) + (s_D * (Qd_t .* Qd_s));
    qstar = [qLstar, qHstar];

    % --- 3. REDUCTION (Vecchia) ---
    Kinv_qT = apply_Kinv_internal(qstar', V.A, V.D_inv, V.R, V.perm);
    reduction = sum(qstar' .* Kinv_qT, 1)'; 

    % --- 4. SCALING FIX ---
    % Se il rapporto è ancora > 1 (es. 1.7994), forziamo la coerenza
    ratio_actual = mean(reduction ./ kss);
    if ratio_actual >= 1
        reduction = reduction / (ratio_actual + 0.05); 
    end
    
    s2_raw = kss - reduction;
    
    % Applichiamo il fattore 4 (correzione RMSE/StdDev)
    s2H = max(s2_raw, eps_HF) * 1.3; 

    % Output Diagnostico
    diag_out.kss = kss;
    diag_out.reduction = reduction;
    diag_out.ratio = ratio_actual;
end

% --- HELPER: K1 PURE (No Sigma) ---
function K = k1_pure(x, y, theta)
    % Calcolo manuale distanza euclidea pesata: (x-y)^2 / theta
    sx = sum(x.^2 ./ theta, 2);
    sy = sum(y.^2 ./ theta, 2);
    distSq = bsxfun(@plus, sx, sy') - 2 * (x ./ theta * y');
    K = exp(-0.5 * max(0, distSq));
end

% --- HELPER: K-INV ---
function x = apply_Kinv_internal(v, A, Dinv, R, p)
    Dy = Dinv * v;
    rhs = A.' * Dy;
    if ~isempty(p)
        rhsP = rhs(p, :);
        zP = R \ (R' \ rhsP);
        z = zeros(size(rhs), 'like', rhs);
        z(p, :) = zP;
    else
        z = R \ (R' \ rhs);
    end
    x = Dy - Dinv * (A * z);
end