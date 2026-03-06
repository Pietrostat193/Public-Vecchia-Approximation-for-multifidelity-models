function [muH, s2H] = predict_calibratedCM3_fixed(Xstar, ModelInfo, opts)
    % predict_calibratedCM3_fixed
    % Versione definitiva con: Media GLS, Calibrazione Affine e Varianza Residua.
    
    if nargin < 3, opts = struct(); end
    if ~isfield(opts,'gamma_clip'), opts.gamma_clip = [0.25, 4.0]; end
    if ~isfield(opts,'lambda_ridge'), opts.lambda_ridge = 1e-8; end

    M = ModelInfo; V = M.debug_vecchia; hyp = M.hyp;
    eps_HF = exp(hyp(7)); 

    % 1. Setup matrici e K^-1
    % Definiamo l'anonima che punta alla funzione locale definita in fondo
    applyKinv = @(v) apply_Kinv_local_internal(v, V.A, V.D_inv, V.R, V.perm);
    
    nL = size(M.X_L,1); nH = size(M.X_H,1);
    y_full = [M.y_L; M.y_H];
    Z_train = [ [ones(nL,1); zeros(nH,1)], [zeros(nL,1); ones(nH,1)] ];

    % 2. Residui e Intercette GLS
    m_GLS = V.m_GLS; 
    KinvZ = [applyKinv(Z_train(:,1)), applyKinv(Z_train(:,2))];
    resid_alpha = applyKinv(y_full - Z_train * m_GLS);

    % 3. Predizione Base (mu0)
    rho_star = compute_rho_star(Xstar, M.X_H, hyp, M.RhoFunction);
    [qLstar, qHstar] = build_q_local_internal(Xstar, rho_star(:), M);
    qstar = [qLstar, qHstar];
    mu0 = qstar * resid_alpha;

    % 4. CALIBRAZIONE AFFINE + CALCOLO VARIANZA RESIDUA
    rho_H = compute_rho_star(M.X_H, M.X_H, hyp, M.RhoFunction);
    [qL_H, qH_H] = build_q_local_internal(M.X_H, rho_H(:), M);
    mu0_H_train = [qL_H, qH_H] * resid_alpha;
    
    % Fitting: (y_H - m_HF) = a + b * mu0_H_train
    target_H = M.y_H - m_GLS(2);
    Xcal = [ones(nH,1), mu0_H_train];
    G = Xcal'*Xcal + diag([0, opts.lambda_ridge]);
    ab = G \ (Xcal' * target_H); 
    ab(2) = max(opts.gamma_clip(1), min(opts.gamma_clip(2), ab(2)));

    % Calcolo della varianza dei residui della calibrazione (il "pezzo" mancante)
    fit_H = Xcal * ab;
    s2_cal_res = mean((target_H - fit_H).^2); 

    % Media Finale (RMSE ~0.3)
    muH = m_GLS(2) + ab(1) + ab(2) * mu0;

    % 5. Varianza Finale (GP + GLS + Calibrazione + Residuo)
    kss = prior_diag_local_internal(Xstar, rho_star(:), M);
    reduction = sum(qstar' .* applyKinv(qstar'), 1)';
    s2_0 = max(kss - reduction, 0);

    C_m = inv(Z_train' * KinvZ);
    h = repmat([0, 1], size(Xstar,1), 1) - qstar * KinvZ;
    s2_GLS = sum((h * C_m) .* h, 2);

    % Somma finale: scaliamo la varianza teorica e aggiungiamo l'errore di fitting
    s2H = (ab(2)^2) * (s2_0 + s2_GLS) + s2_cal_res + eps_HF;
end

% --- FUNZIONI HELPER LOCALI ---

function x = apply_Kinv_local_internal(v, A, Dinv, R, p)
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

function [qL, qH] = build_q_local_internal(X, rho, M)
    h = M.hyp; 
    tLF=[exp(h(1)),exp(h(2))]; pLF=[exp(h(8)),exp(h(9))];
    tHF=[exp(h(3)),exp(h(4))]; pHF=[exp(h(10)),exp(h(11))];
    
    % Selettore Kernel (Matern di default)
    if contains(string(M.cov_type),'RBF'), k=@k1; else, k=@k_matern; end
    
    qL_base = k(X(:,1), M.X_L(:,1), tLF) .* k(X(:,2:3), M.X_L(:,2:3), pLF);
    qH_base = k(X(:,1), M.X_H(:,1), tLF) .* k(X(:,2:3), M.X_H(:,2:3), pLF);
    qd_base = k(X(:,1), M.X_H(:,1), tHF) .* k(X(:,2:3), M.X_H(:,2:3), pHF);
    
    qL = qL_base .* rho; 
    qH = qH_base .* (rho.^2) + qd_base;
end

function kss = prior_diag_local_internal(X, rho, M)
    h = M.hyp;
    tLF=[exp(h(1)),exp(h(2))]; pLF=[exp(h(8)),exp(h(9))];
    tHF=[exp(h(3)),exp(h(4))]; pHF=[exp(h(10)),exp(h(11))];
    if contains(string(M.cov_type),'RBF'), k=@k1; else, k=@k_matern; end
    
    kL = diag(k(X(:,1),X(:,1),tLF)) .* diag(k(X(:,2:3),X(:,2:3),pLF));
    kD = diag(k(X(:,1),X(:,1),tHF)) .* diag(k(X(:,2:3),X(:,2:3),pHF));
    kss = (rho.^2) .* kL + kD;
end