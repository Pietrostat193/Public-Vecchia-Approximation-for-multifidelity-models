function [NLML] = likelihoodVecchia_nonstat_GLS_v2(hyp)
    % Versione Integrale v2: GLS Adattivo + Speculare a v1
    global ModelInfo;

    % === 1. Configurazione e Estrazione Base ===
    usePermutation = true;
    X_L = ModelInfo.X_L; 
    X_H = ModelInfo.X_H;
    y_L = ModelInfo.y_L; 
    y_H = ModelInfo.y_H;
    y   = [y_L; y_H];
    N   = size(y, 1);
    nL  = size(X_L, 1);
    nH  = size(X_H, 1);
    
    nn_size      = ModelInfo.nn_size;
    jitter       = ModelInfo.jitter;
    kernel       = ModelInfo.kernel;
    conditioning = ModelInfo.conditioning;
    MeanFunction = ModelInfo.MeanFunction; % Estratta subito per evitare errore riga 76
    RhoFunction  = ModelInfo.RhoFunction;

    % === 2. Estrazione Iperparametri Covarianza (Mapping v1) ===
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    rho_const  = hyp(5);
    eps_LF     = exp(hyp(6));  eps_HF = exp(hyp(7));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    % === 3. Vecchia Approximation (LF & HF) ===
    idxFixed_L = []; idxFixed_H = [];
    if isfield(ModelInfo,'vecchia_idxL'), idxFixed_L = ModelInfo.vecchia_idxL; end
    if isfield(ModelInfo,'vecchia_idxH'), idxFixed_H = ModelInfo.vecchia_idxH; end

    % Blocco Low-Fidelity (Satellite)
    if conditioning == "Corr"
        result_LF = vecchia_approx_space_time_corr_fast1(X_L, [s_sig_LF_s, s_ell_LF], ...
            [s_sig_LF_t, t_ell_LF], nn_size, 1e-6, kernel, 10, t_ell_LF, s_ell_LF, idxFixed_L);
    else
        result_LF = vecchia_approx_space_time_optimized(X_L, [s_sig_LF_s, s_ell_LF], ...
            [s_sig_LF_t, t_ell_LF], nn_size, 1e-6, kernel);
    end
    Ki_L = result_LF.B' * result_LF.Di * result_LF.B;
    log_det_K_L = sum(log(diag(result_LF.Di)));

    % Blocco High-Fidelity (Stazioni - Discrepanza)
    if conditioning == "Corr"
        result_HF = vecchia_approx_space_time_corr_fast1(X_H, [s_sig_HF_s, s_ell_HF], ...
            [s_sig_HF_t, t_ell_HF], nn_size, 1e-6, kernel, 10, t_ell_HF, s_ell_HF, idxFixed_H);
    else
        result_HF = vecchia_approx_space_time_optimized(X_H, [s_sig_HF_s, s_ell_HF], ...
            [s_sig_HF_t, t_ell_HF], nn_size, 1e-6, kernel);
    end
    Ki_D = result_HF.B' * result_HF.Di * result_HF.B;
    log_det_K_D = sum(log(diag(result_HF.Di)));
    
    log_det_W = -(log_det_K_D + log_det_K_L);

    % === 4. Modellazione rho_H (Nonstazionarietà) ===
    switch RhoFunction
        case "constant"
            rho_input = rho_const; % PASSA SCALARE per CM_nested1
            rho_H = rho_const * ones(nH, 1); 
            NonStat = "F";
        case "linear"
            beta_rho = exp(hyp(12:14)); 
            Phi_H = [ones(nH,1), X_H(:,2:3)];
            rho_H = Phi_H * beta_rho;
            rho_input = rho_H; 
            NonStat = "T";
        case "GP_scaled_empirical"
            log_sigma_rho = hyp(end - 2);
            log_ell_rho   = [hyp(end-1); hyp(end)];
            [X_unique, ~, idx_back] = unique(X_H(:, 2:3), 'rows');
            n_locs = size(X_unique, 1);
            rho_local = zeros(n_locs, 1);
            
            % Check MeanFunction per residui empirici
            if MeanFunction == "GP_res"
                base_L = y_L - (ModelInfo.m_x_L);
                base_H = y_H - (ModelInfo.m_x_H);
            else
                base_L = y_L; base_H = y_H;
            end
            
            for i = 1:n_locs
                coord = X_unique(i, :);
                idx_Hi = ismember(X_H(:, 2:3), coord, 'rows');
                idx_Li = ismember(X_L(:, 2:3), coord, 'rows');
                [common_t, iH, iL] = intersect(X_H(idx_Hi, 1), X_L(idx_Li, 1)); 
                y_H_i = base_H(idx_Hi); y_L_i = base_L(idx_Li);
                y_H_aligned = y_H_i(iH); y_L_aligned = y_L_i(iL);
                if length(y_L_aligned) >= 2 && var(y_L_aligned) > 0
                    C = cov(y_H_aligned, y_L_aligned);
                    rho_local(i) = C(1, 2) / var(y_L_aligned);
                else
                    rho_local(i) = 0;
                end
            end
            
            theta_rho = exp([log_sigma_rho; log_ell_rho]);
            gprModel_rho = fitrgp(X_unique, rho_local, 'KernelFunction','ardsquaredexponential',...
                'KernelParameters',theta_rho,'FitMethod','none','Sigma',0.01);
            rho_H = predict(gprModel_rho, X_H(:, 2:3));
            rho_input = rho_H;
            NonStat = "T";
            
            % Cache Rho Info
            ModelInfo.gprModel_rho = gprModel_rho;
            ModelInfo.rho_local = rho_local;
            ModelInfo.rho_H_unique = predict(gprModel_rho, X_unique);
            ModelInfo.X_H_unique = X_unique;
        otherwise
            rho_input = rho_const; rho_H = rho_const * ones(nH,1); NonStat = "F";
    end

    % === 5. Costruzione Sistema Annidato (CM_nested) ===
    [A, D, D_inv] = CM_nested1(X_L, X_H, rho_input, eps_LF, eps_HF, NonStat, y_L, y_H);
    H = blkdiag(Ki_L, Ki_D) + A' * D_inv * A + speye(size(A,2)) * jitter;

    % Cholesky
    if usePermutation
        perm = symamd(H);
        [R, pchol] = chol(H(perm, perm));
        if pchol > 0, error('H non definita positiva'); end
        log_det_H = 2 * sum(log(diag(R)));
    else
        [R, pchol] = chol(H);
        if pchol > 0, error('H non definita positiva'); end
        log_det_H = 2 * sum(log(diag(R)));
        perm = 1:size(H,1);
    end

    % === 6. ADAPTIVE GLS MEAN REMOVAL ===
    applyKinv = @(v) apply_Kinv_local(v, A, D_inv, R, perm);
    
    GLSType = "constant";
    if isfield(ModelInfo, 'GLSType'), GLSType = ModelInfo.GLSType; end
    
    if GLSType == "adaptive"
        % Design Matrix Spaziale: [1, Lat, Lon] per ogni fedeltà
        G_L = [ones(nL,1), X_L(:,2), X_L(:,3)];
        G_H = [ones(nH,1), X_H(:,2), X_H(:,3)];
        G_gls = blkdiag(G_L, G_H);
    else
        % GLS Standard: due intercette globali
        G_gls = [ [ones(nL,1); zeros(nH,1)], [zeros(nL,1); ones(nH,1)] ];
    end

    % Risoluzione GLS: beta = (G' K^-1 G) \ G' K^-1 y
    Kinv_y = applyKinv(y);
    num_cols_G = size(G_gls, 2);
    Kinv_G = zeros(N, num_cols_G);
    for j = 1:num_cols_G
        Kinv_G(:,j) = applyKinv(G_gls(:,j));
    end
    
    G_Kinv_G = G_gls' * Kinv_G;
    G_Kinv_y = G_gls' * Kinv_y;
    beta_gls = G_Kinv_G \ G_Kinv_y;

    % Residui
    y_tilde = y - G_gls * beta_gls;
    SIy_tilde = Kinv_y - Kinv_G * beta_gls; 

    % === 7. Calcolo Likelihood ===
    log_det_D = sum(log(diag(D)));
    term1 = 0.5 * (y_tilde' * SIy_tilde);
    term2 = 0.5 * (log_det_W + log_det_H + log_det_D);
    term3 = 0.5 * N * log(2 * pi);
    
    NLML = term1 + term2 + term3;

    % === 8. Caching per Predizione ===
    ModelInfo.beta_gls = beta_gls;
    ModelInfo.G_gls = G_gls;
    ModelInfo.SIy = SIy_tilde;
    ModelInfo.L = R;
    ModelInfo.perm = perm;
    ModelInfo.A = A;
    ModelInfo.D_inv = D_inv;
    ModelInfo.rho_H = rho_H;
    
    % Debug info speculare a v1
    dbg.m_GLS = beta_gls;
    dbg.A = A; dbg.D_inv = D_inv; dbg.R = R; dbg.perm = perm;
    ModelInfo.debug_vecchia = dbg;
end

% --- HELPER FUNCTIONS ---
function x = apply_Kinv_local(v, A, Dinv, R, p)
    Dy = Dinv * v;
    rhs = A' * Dy;
    rhsP = rhs(p);
    zP = R \ (R' \ rhsP);
    z = zeros(size(rhs)); 
    z(p) = zP;
    x = Dy - Dinv * (A * z);
end

function result = vecchia_approx_space_time_corr_fast1(locations, hyp_s, hyp_t, nn, eps_val, kernel, cand_mult, ell_t, ell_s, idxAll)
    if nargin < 4 || isempty(nn), nn = 15; end
    if nargin < 5 || isempty(eps_val), eps_val = 1e-7; end
    if nargin < 6 || isempty(kernel), kernel = 'RBF'; end
    [n, ~] = size(locations);
    nn = min(nn, n-1);
    inv_ell_t2 = 1/(ell_t^2 + eps);
    inv_ell_s2 = 1/(ell_s^2 + eps);
    
    var_self = zeros(n,1);
    for j = 1:n
        vj = k_space_time(locations(j,:), locations(j,:), hyp_s, hyp_t, kernel);
        var_self(j) = max(vj, eps_val);
    end
    
    B_rows_c = cell(n,1); B_cols_c = cell(n,1); B_vals_c = cell(n,1);
    Di_vals = zeros(n,1);
    hasIdx = (nargin >= 10) && ~isempty(idxAll);
    
    for i = 1:n
        if i == 1
            Di_vals(1) = 1 / var_self(1);
            continue;
        end
        prev_idx = 1:(i-1);
        xi = locations(i,:);
        
        fixed_list = [];
        if hasIdx
            row = double(idxAll(i, :));
            fixed_list = row(row > 0 & row < i);
        end
        n_ind = fixed_list(1:min(nn, numel(fixed_list)));
        
        k_needed = nn - numel(n_ind);
        if k_needed > 0
            rem_pool = setdiff(prev_idx, n_ind, 'stable');
            if ~isempty(rem_pool)
                dt = (locations(rem_pool,1) - xi(1)).^2 * inv_ell_t2;
                dx = (locations(rem_pool,2) - xi(2)).^2 * inv_ell_s2;
                dy = (locations(rem_pool,3) - xi(3)).^2 * inv_ell_s2;
                [~, Icand] = mink(dt+dx+dy, min(numel(rem_pool), cand_mult*k_needed));
                cand_idx = rem_pool(Icand);
                
                K_cand = k_space_time(locations(cand_idx,:), xi, hyp_s, hyp_t, kernel);
                denom = sqrt(var_self(i) * var_self(cand_idx));
                corr_cand = K_cand(:) ./ max(denom, eps_val);
                [~, pick] = maxk(corr_cand, min(k_needed, numel(cand_idx)));
                n_ind = [n_ind, cand_idx(pick)];
            end
        end
        
        k = numel(n_ind);
        if k > 0
            Xnbrs = locations(n_ind,:);
            K_nn = k_space_time(Xnbrs, Xnbrs, hyp_s, hyp_t, kernel);
            K_nn = 0.5*(K_nn + K_nn') + eps_val*eye(k);
            K_i_n = k_space_time(Xnbrs, xi, hyp_s, hyp_t, kernel);
            Ai = K_nn \ K_i_n(:);
            cond_var = max(var_self(i) - K_i_n(:)' * Ai, eps_val);
            
            B_rows_c{i} = repmat(i, k, 1);
            B_cols_c{i} = n_ind(:);
            B_vals_c{i} = -Ai;
            Di_vals(i) = 1 / cond_var;
        else
            Di_vals(i) = 1 / var_self(i);
        end
    end
    
    B = sparse(double(vertcat(B_rows_c{:})), double(vertcat(B_cols_c{:})), vertcat(B_vals_c{:}), n, n) + speye(n);
    result.B = B; result.Di = spdiags(Di_vals, 0, n, n);
end