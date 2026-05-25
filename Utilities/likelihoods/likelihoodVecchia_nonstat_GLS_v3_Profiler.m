function [NLML] = likelihoodVecchia_nonstat_GLS_v3_Profiler(hyp)
    global ModelInfo;
    T = struct(); 
    main_tic = tic;
    
    % === 1. Configurazione ===
    usePermutation = true;
    X_L = ModelInfo.X_L; X_H = ModelInfo.X_H;
    y_L = ModelInfo.y_L; y_H = ModelInfo.y_H;
    y = [y_L; y_H]; N = size(y, 1);
    nL = size(X_L, 1); nH = size(X_H, 1);
    nn_size = ModelInfo.nn_size; jitter = ModelInfo.jitter;
    kernel = ModelInfo.kernel; conditioning = ModelInfo.conditioning;
    MeanFunction = ModelInfo.MeanFunction; RhoFunction = ModelInfo.RhoFunction;

    % === 2. Iperparametri ===
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    rho_const  = hyp(5);
    eps_LF     = exp(hyp(6));  eps_HF = exp(hyp(7));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    % === 3. Vecchia Approximation ===
    t_v_total = tic;
    idxFixed_L = []; idxFixed_H = [];
    if isfield(ModelInfo,'vecchia_idxL'), idxFixed_L = ModelInfo.vecchia_idxL; end
    if isfield(ModelInfo,'vecchia_idxH'), idxFixed_H = ModelInfo.vecchia_idxH; end
    
    if conditioning == "Corr"
        result_LF = vecchia_approx_corr_PROFILER(X_L, [s_sig_LF_s, s_ell_LF], [s_sig_LF_t, t_ell_LF], nn_size, 1e-6, kernel, 10, t_ell_LF, s_ell_LF, idxFixed_L);
        result_HF = vecchia_approx_corr_PROFILER(X_H, [s_sig_HF_s, s_ell_HF], [s_sig_HF_t, t_ell_HF], nn_size, 1e-6, kernel, 10, t_ell_HF, s_ell_HF, idxFixed_H);
    else
        % Se usi optimized (non-corr), il tempo verrà comunque conteggiato qui
        result_LF = vecchia_approx_space_time_optimized(X_L, [s_sig_LF_s, s_ell_LF], [s_sig_LF_t, t_ell_LF], nn_size, 1e-6, kernel);
        result_HF = vecchia_approx_space_time_optimized(X_H, [s_sig_HF_s, s_ell_HF], [s_sig_HF_t, t_ell_HF], nn_size, 1e-6, kernel);
    end
    T.Vecchia_Core_Logic = toc(t_v_total);

    t_mat_ops = tic;
    Ki_L = result_LF.B' * result_LF.Di * result_LF.B;
    Ki_D = result_HF.B' * result_HF.Di * result_HF.B;
    log_det_W = -(sum(log(diag(result_LF.Di))) + sum(log(diag(result_HF.Di))));
    T.Sparse_Matrix_Assembly = toc(t_mat_ops);

    % === 4. Modellazione rho_H ===
    t_rho = tic;
    if RhoFunction == "GP_scaled_empirical"
        [X_unique, ~, idx_back] = unique(X_H(:, 2:3), 'rows');
        n_locs = size(X_unique, 1);
        base_L = y_L; base_H = y_H;
        if MeanFunction == "GP_res"
            base_L = y_L - ModelInfo.m_x_L; base_H = y_H - ModelInfo.m_x_H;
        end
        rho_local = zeros(n_locs, 1);
        for i = 1:n_locs
            mask_H = (idx_back == i);
            mask_L = (X_L(:,2) == X_unique(i,1) & X_L(:,3) == X_unique(i,2));
            [~, iH, iL] = intersect(X_H(mask_H, 1), X_L(mask_L, 1)); 
            if length(iH) >= 2
                y_H_aligned = base_H(mask_H); y_L_aligned = base_L(mask_L);
                C = cov(y_H_aligned(iH), y_L_aligned(iL));
                rho_local(i) = C(1, 2) / var(y_L_aligned(iL));
            end
        end
        gprModel_rho = fitrgp(X_unique, rho_local, 'KernelFunction','ardsquaredexponential',...
            'KernelParameters',exp([hyp(end-2); hyp(end-1); hyp(end)]),'FitMethod','none','Sigma',0.01);
        rho_H = predict(gprModel_rho, X_H(:, 2:3));
        rho_input = rho_H; NonStat = "T";
    else
        rho_input = rho_const; rho_H = rho_const * ones(nH,1); NonStat = "F";
    end
    T.Rho_Empirical_GP = toc(t_rho);

    % === 5. CM_nested e H matrix ===
    t_nested = tic;
    % Qui chiamiamo la tua funzione esterna CM_nested1
    [A, D, D_inv] = CM_nested1(X_L, X_H, rho_input, eps_LF, eps_HF, NonStat, y_L, y_H);
    H = blkdiag(Ki_L, Ki_D) + A' * D_inv * A + speye(size(A,2)) * jitter;
    T.CM_Nested_Build = toc(t_nested);

    % === 6. Cholesky ===
    t_chol = tic;
    perm = symamd(H);
    [R, pchol] = chol(H(perm, perm));
    if pchol > 0, error('H non definita positiva'); end
    log_det_H = 2 * sum(log(diag(R)));
    T.Cholesky_H = toc(t_chol);

    % === 7. GLS e Likelihood ===
    t_gls = tic;
    if isfield(ModelInfo, 'GLSType') && ModelInfo.GLSType == "adaptive"
        G_gls = [[ones(nL,1), X_L(:,2), X_L(:,3)]; [ones(nH,1), X_H(:,2), X_H(:,3)]];
    else
        G_gls = [[ones(nL,1); zeros(nH,1)], [zeros(nL,1); ones(nH,1)]];
    end
    
    % Testiamo apply_Kinv_local internamente per il profiling
    Kinv_yG = apply_Kinv_profiler([y, G_gls], A, D_inv, R, perm);
    
    beta_gls = (G_gls' * Kinv_yG(:, 2:end)) \ (G_gls' * Kinv_yG(:, 1));
    SIy_tilde = Kinv_yG(:, 1) - Kinv_yG(:, 2:end) * beta_gls;
    y_tilde = y - G_gls * beta_gls;
    
    NLML = 0.5 * (y_tilde' * SIy_tilde + log_det_W + log_det_H + sum(log(diag(D))) + N*log(2*pi));
    T.GLS_and_Likelihood = toc(t_gls);

    T.Total_Time = toc(main_tic);
    ModelInfo.Profiling = T;
end

% --- HELPER FUNCTIONS CON PROFILING ---

function X = apply_Kinv_profiler(V, A, Dinv, R, p)
    % Risolve K^-1 * V usando Woodbury
    DY = Dinv * V;
    RHS = A' * DY;
    RHSP = RHS(p, :);
    ZP = R \ (R' \ RHSP);
    Z = zeros(size(RHS)); 
    Z(p, :) = ZP;
    X = DY - Dinv * (A * Z);
end

function result = vecchia_approx_corr_PROFILER(locations, hyp_s, hyp_t, nn, eps_val, kernel, cand_mult, ell_t, ell_s, idxAll)
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
                % Ricerca vicini (Questo è solitamente il bottleneck)
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