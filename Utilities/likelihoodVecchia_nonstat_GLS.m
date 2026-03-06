function [NLML] = likelihoodVecchia_nonstat_GLS(hyp)
    % Non-destructive: preserves your original structure & caches.
    % Adds only: Corr neighbor widening, cand_mult control, and a path diagnostic.

    global ModelInfo;


    if ~isfield(ModelInfo,'show_path_diag'), ModelInfo.show_path_diag = false; end

    % === Configuration ===
    usePermutation = true;
    UsePenalty     = false;

    % === Extract Model Information ===
    X_L = ModelInfo.X_L;
    X_H = ModelInfo.X_H;
    y_L = ModelInfo.y_L;
    y_H = ModelInfo.y_H;
    y   = [y_L; y_H];
    jitter  = ModelInfo.jitter;
    N       = size(y, 1);
    nn_size = ModelInfo.nn_size;

    % === Extract Hyperparameters ===
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    rho   = hyp(5);
    eps_LF = exp(hyp(6));  eps_HF = exp(hyp(7));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    % === Covariance Approximations (Vecchia) ===
    kernel       = ModelInfo.kernel;
    conditioning = ModelInfo.conditioning;

    % Load any cached fixed-neighbor lists
    idxFixed_L = [];
    idxFixed_H = [];
    if isfield(ModelInfo,'vecchia_idxL'), idxFixed_L = ModelInfo.vecchia_idxL; end
    if isfield(ModelInfo,'vecchia_idxH'), idxFixed_H = ModelInfo.vecchia_idxH; end

    % --- Corr: ensure cached neighbor lists are at least nn_size wide (widen/rebuild)
    if conditioning == "Corr"
        if isempty(idxFixed_L) || size(idxFixed_L,2) < nn_size
            ModelInfo.vecchia_idxL = build_fixed_neighbors_once(X_L, nn_size, t_ell_LF, s_ell_LF);
            idxFixed_L = ModelInfo.vecchia_idxL;
        end
        if isempty(idxFixed_H) || size(idxFixed_H,2) < nn_size
            ModelInfo.vecchia_idxH = build_fixed_neighbors_once(X_H, nn_size, t_ell_HF, s_ell_HF);
            idxFixed_H = ModelInfo.vecchia_idxH;
        end
    end

    % Candidate pool size for Corr neighbor selection (overridable)
    if isfield(ModelInfo,'cand_mult') && ~isempty(ModelInfo.cand_mult)
        cand_mult = ModelInfo.cand_mult;
    else
        cand_mult = max(10, ceil(0.5*nn_size));
    end

    jitter_v = 1e-6;

    % --- LF block
    switch conditioning
        case "MinMax"
            result_LF = vecchia_approx_space_time_optimized( ...
                X_L, [s_sig_LF_s, s_ell_LF], [s_sig_LF_t, t_ell_LF], nn_size, jitter_v, kernel);
        case "Corr"
            result_LF = vecchia_approx_space_time_corr_fast1( ...
                X_L, [s_sig_LF_s, s_ell_LF], [s_sig_LF_t, t_ell_LF], ...
                nn_size, 0.0, kernel, cand_mult, t_ell_LF, s_ell_LF, idxFixed_L);
        otherwise
            error('Unknown conditioning: %s', conditioning);
    end
    Di_L_sparse = result_LF.Di;
    log_det_K_L = sum(log(diag(Di_L_sparse)));
    Ki_L = result_LF.B' * Di_L_sparse * result_LF.B;

    % --- HF block
    switch conditioning
        case "MinMax"
            result_HF = vecchia_approx_space_time_optimized( ...
                X_H, [s_sig_HF_s, s_ell_HF], [s_sig_HF_t, t_ell_HF], nn_size, jitter_v, kernel);
        case "Corr"
            result_HF = vecchia_approx_space_time_corr_fast1( ...
                X_H, [s_sig_HF_s, s_ell_HF], [s_sig_HF_t, t_ell_HF], ...
                nn_size, 0.0, kernel, cand_mult, t_ell_HF, s_ell_HF, idxFixed_H);
    end
    Di_D_sparse = result_HF.Di;
    log_det_K_D = sum(log(diag(Di_D_sparse)));
    Ki_D = result_HF.B' * Di_D_sparse * result_HF.B;

    log_det_W = -(log_det_K_D + log_det_K_L);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % === Mean Function (all your branches kept)                        ===
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    MeanFunction = ModelInfo.MeanFunction;
    switch MeanFunction
        case "zero"
            m_x = 0;

        case "constant"
            m_x = hyp(12) * ones(size(y));

        case "linear"
            X_combined = [X_L; X_H];
            weights = hyp(12:14);
            bias = hyp(15);
            m_x = X_combined * weights + bias;

        case "GP"
            log_sigma = hyp(12);
            log_ell   = hyp(13:15);
            sigmaF = exp(log_sigma);
            lengthScale = exp(log_ell);
            gprModel_forH = fitrgp(X_H, y_H, ...
                'KernelFunction','ardsquaredexponential', ...
                'KernelParameters',[lengthScale; sigmaF], ...
                'BasisFunction','none','FitMethod','none', ...
                'PredictMethod','exact','Sigma',0.01,'Standardize',false);
            ModelInfo.gprModel_mean = gprModel_forH;
            X_combined = [X_L; X_H];
            m_x = predict(gprModel_forH, X_combined);

        case "GP_res"
            log_sigma_L = hyp(12);  log_ell_L = hyp(13:14);
            log_sigma_H = hyp(15);  log_ell_H = hyp(16:17);
            sigmaF_L = exp(log_sigma_L); lengthScale_L = exp(log_ell_L);
            sigmaF_H = exp(log_sigma_H); lengthScale_H = exp(log_ell_H);

            [unique_X_L, ~, idx_L] = unique(X_L(:,2:3), 'rows');
            [unique_X_H, ~, idx_H] = unique(X_H(:,2:3), 'rows');
            y_L_avg = accumarray(idx_L, y_L, [], @mean);
            y_H_avg = accumarray(idx_H, y_H, [], @mean);

            gprModel_mean_L = fitrgp(unique_X_L, y_L_avg, ...
                'KernelFunction','ardsquaredexponential', ...
                'KernelParameters',[lengthScale_L; sigmaF_L], ...
                'BasisFunction','none','FitMethod','none', ...
                'PredictMethod','exact','Sigma',0.01,'Standardize',false);

            gprModel_mean_H = fitrgp(unique_X_H, y_H_avg, ...
                'KernelFunction','ardsquaredexponential', ...
                'KernelParameters',[lengthScale_H; sigmaF_H], ...
                'BasisFunction','none','FitMethod','none', ...
                'PredictMethod','exact','Sigma',0.01,'Standardize',false);

            m_x_L = predict(gprModel_mean_L, unique_X_L); m_x_L = m_x_L(idx_L);
            m_x_H = predict(gprModel_mean_H, unique_X_H); m_x_H = m_x_H(idx_H);

            ModelInfo.gprModel_mean_L = gprModel_mean_L;
            ModelInfo.gprModel_mean_H = gprModel_mean_H;
            ModelInfo.m_x_L = m_x_L; ModelInfo.m_x_H = m_x_H;

        case "GP_for_H"
            log_sigma = hyp(12); log_ell = hyp(13:15);
            sigmaF = exp(log_sigma); lengthScale = exp(log_ell);
            gprModel_forH = fitrgp(X_H(:,2:3), y_H, ...
                'KernelFunction','ardsquaredexponential', ...
                'KernelParameters',[lengthScale; sigmaF], ...
                'BasisFunction','none','FitMethod','none', ...
                'PredictMethod','exact','Sigma',0.01,'Standardize',false);
            ModelInfo.gprModel_forH = gprModel_forH;
            m_x_H = predict(gprModel_forH, X_H(:,2:3));

        otherwise
            error('Invalid MeanFunction type.');
    end

    if MeanFunction=="GP_for_H"
        res = y_H - m_x_H;
        y_tilde = [y_L; res];
    elseif MeanFunction=="GP_res"
        y_L_tilde = y_L - ModelInfo.m_x_L;
        y_H_tilde = y_H - ModelInfo.m_x_H;
        y_tilde   = [y_L_tilde; y_H_tilde];
    else
        y_tilde = y - m_x;
        ModelInfo.m_x = m_x;
    end
    ModelInfo.y_tilde=y_tilde;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % === Nonstationary rho_H (all your branches kept)                  ===
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    RhoFunction = ModelInfo.RhoFunction;
    switch RhoFunction
        case "constant"
            rho_H = rho; NonStat = "F";

        case "linear"
            phi_x = @(x) [ones(size(x, 1), 1), x];
            Phi_H = phi_x(X_H(:, 2:3));
            if MeanFunction == "zero"
                beta_rho = exp(hyp(12:end));
            elseif MeanFunction == "constant"
                beta_rho = exp(hyp(13:end));
            elseif MeanFunction == "linear"
                beta_rho = exp(hyp(14:end));
            elseif MeanFunction == "GP"
                beta_rho = exp(hyp(15:end));
            elseif MeanFunction == "GP_for_H"
                beta_rho = exp(hyp(15:end));
            elseif MeanFunction == "GP_res"
                beta_rho = exp(hyp(18:end));
            else
                error("Unknown MeanFunction: " + MeanFunction);
            end
            ModelInfo.beta_rho = beta_rho;
            rho_H = Phi_H * beta_rho; NonStat = "T";
            ModelInfo.rho_H = rho_H;

        case "polynomial"
            lat = X_H(:, 2); lon = X_H(:, 3);
            lat_norm = (lat - min(lat)) / (max(lat) - min(lat));
            lon_norm = (lon - min(lon)) / (max(lon) - min(lon));
            Phi_H = [ones(size(lat)), lat_norm, lon_norm, lat_norm.^2, lon_norm.^2];
            if MeanFunction == "zero"
                beta_rho = exp(hyp(12:end));
            elseif MeanFunction == "constant"
                beta_rho = exp(hyp(13:end));
            elseif MeanFunction == "linear"
                beta_rho = exp(hyp(16:end));
            elseif MeanFunction == "GP"
                beta_rho = exp(hyp(15:end));
            elseif MeanFunction == "GP_res"
                beta_rho = exp(hyp(18:end));
            else
                error("Unknown MeanFunction: " + MeanFunction);
            end
            ModelInfo.beta_rho = beta_rho;
            rho_H = Phi_H * beta_rho; NonStat = "T";
            ModelInfo.rho_H = rho_H;

        case "GP_scaled"
            [X_unique, ~, idx_back] = unique(X_H(:, 2:3), 'rows');
            sigma = exp(hyp(end - 1)); ell = exp(hyp(end));
            dists = pdist2(X_unique, X_unique).^2;
            K_unique = sigma^2 * exp(-0.5 * dists / ell^2) + 1e-6 * eye(size(dists));
            rho_unique = abs(mean(K_unique, 2));
            rho_H = rho_unique(idx_back); NonStat = "T";
            ModelInfo.rho_H_unique = rho_unique;
            ModelInfo.X_H_unique   = X_unique;
            ModelInfo.K_rho_unique = K_unique;

        case "GP_scaled_empirical"
            log_sigma = hyp(end - 2);
            log_ell   = [hyp(end-1); hyp(end)];
            [X_unique, ~, idx_back] = unique(X_H(:, 2:3), 'rows');
            n_locs = size(X_unique, 1);
            rho_local = zeros(n_locs, 1);
            if MeanFunction=="GP_res"
                base_L = y_L - (ModelInfo.m_x_L);
                base_H = y_H - (ModelInfo.m_x_H);
            else
                base_L = y_L; base_H = y_H;
            end
            for i = 1:n_locs
                coord = X_unique(i, :);
                idx_Hi = ismember(X_H(:, 2:3), coord, 'rows');
                idx_Li = ismember(X_L(:, 2:3), coord, 'rows');
                y_H_i = base_H(idx_Hi);
                y_L_i = base_L(idx_Li);
                t_Hi = X_H(idx_Hi, 1); t_Li = X_L(idx_Li, 1);
                [common_t, iH, iL] = intersect(t_Hi, t_Li); 
                y_H_aligned = y_H_i(iH);
                y_L_aligned = y_L_i(iL);
                if length(y_L_aligned) >= 2 && var(y_L_aligned, 1) > 0
                    C = cov(y_H_aligned, y_L_aligned, 1);
                    rho_local(i) = C(1, 2) / var(y_L_aligned, 1);
                else
                    rho_local(i) = 0;
                end
            end
            theta = exp([log_sigma; log_ell]);
            gprModel_rho = fitrgp(X_unique, rho_local, ...
                'KernelFunction','ardsquaredexponential', ...
                'KernelParameters',theta, ...
                'BasisFunction','none','FitMethod','none', ...
                'PredictMethod','exact','Sigma',0.01,'Standardize',false);
            rho_H = predict(gprModel_rho, X_H(:, 2:3));
            NonStat = "T";
            ModelInfo.gprModel_rho = gprModel_rho;
            ModelInfo.rho_H        = rho_H;
            ModelInfo.rho_local    = rho_local;
            ModelInfo.rho_H_unique = predict(gprModel_rho, X_unique);
            ModelInfo.X_H_unique   = X_unique;
    end

    % === Vecchia pieces to assemble full precision ===
    Di_L_sparse = result_LF.Di;
    Di_D_sparse = result_HF.Di;
    Ki_L = result_LF.B' * Di_L_sparse * result_LF.B;
    Ki_D = result_HF.B' * Di_D_sparse * result_HF.B;
    log_det_K_L = sum(log(diag(Di_L_sparse)));
    log_det_K_D = sum(log(diag(Di_D_sparse)));
    log_det_W = -(log_det_K_D + log_det_K_L); 

    % === Build Nested Model Matrices ===
    [A, D, D_inv, Z21, Z1] = CM_nested1(X_L, X_H, rho_H, eps_LF, eps_HF, NonStat, y_L, y_H); %#ok<ASGLU>

    % === Build H and Solve ===
    H = blkdiag(Ki_L, Ki_D) + A' * D_inv * A + speye(size(Ki_L, 1) + size(Ki_D, 1)) * jitter;
    ModelInfo.H = H;

    % pieces for NLML with your chosen mean:
    if exist('y_tilde','var')==0
        error('Internal: y_tilde not set (mean branch).');
    end
    Dy = D_inv * y_tilde;

    if usePermutation
        perm = symamd(H);
        H_perm = H(perm, perm);
        [R_perm, pchol] = chol(H_perm);
        if pchol > 0, error('Permuted H not positive definite'); end
        ModelInfo.perm = perm;
        ModelInfo.L    = R_perm;
        log_det_H = 2 * sum(log(diag(R_perm)));
        AtDy_perm = (A' * Dy);  AtDy_perm = AtDy_perm(perm);
        tmp  = R_perm' \ AtDy_perm;
        H_ADy_perm = R_perm \ tmp;
        H_ADy = zeros(size(H_ADy_perm)); H_ADy(perm) = H_ADy_perm;
    else
        [R, pchol] = chol(H);
        if pchol > 0, error('H not positive definite'); end
        ModelInfo.L = R;
        log_det_H = 2 * sum(log(diag(R)));
        H_ADy = R \ (R' \ (A' * Dy));
        perm  = [];
    end

    SIy_tilde = Dy - D_inv * (A * H_ADy);  % = K^{-1}(y - m) used in NLML
    ModelInfo.SIy=SIy_tilde;
    % === Final Likelihood ===
    log_det_D = sum(log(diag(D)));
    term1 = 0.5 * (y_tilde' * SIy_tilde);
    term2 = 0.5 * (log_det_W + log_det_H + log_det_D);
    term3 = 0.5 * N * log(2 * pi);
    NLML = term1 + term2 + term3;

    % === Optional penalty (unchanged) ===
    if UsePenalty
        ModelInfo.hyp=hyp;
        mean_pred = predictVecchia_nonstat2(X_H);
        if any(mean_pred < 0)
            penalty = 1e4 * sum(abs(mean_pred(mean_pred < 0)));
            NLML = NLML + penalty;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % === Cache everything needed for calibrated prediction (UNCHANGED) ===
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1) Store core Woodbury/Cholesky pieces
    dbg.A     = A;              % sparse "A" from nested CM
    dbg.D_inv = D_inv;          % block diag inverse
    dbg.perm  = perm;           % AMD permutation used for H
    dbg.R     = ModelInfo.L;    % upper-triangular such that H(perm,perm) = R'*R

    % 2) Also store K^{-1} y with *raw* y (no mean subtraction)
    Dy_raw = D_inv * y;
    if usePermutation
        rhs_perm = (A' * Dy_raw); rhs_perm = rhs_perm(perm);
        tmp  = ModelInfo.L' \ rhs_perm;
        zP   = ModelInfo.L  \ tmp;                          % solves H z = A' D^{-1} y
        z    = zeros(size(rhs_perm)); z(1:numel(zP)) = zP;  % permuted vector
        z_full = zeros(size(A,2),1); z_full(perm) = z;      % un-permute
        SIy_raw = Dy_raw - D_inv * (A * z_full);            % = K^{-1} y
    else
        z = ModelInfo.L \ (ModelInfo.L' \ (A' * Dy_raw));
        SIy_raw = Dy_raw - D_inv * (A * z);
    end
    dbg.SIy = SIy_raw;

    % 3) Precompute GLS pieces (two-intercept design)
    nL = size(X_L,1); nH = size(X_H,1);
    Z  = [ [ones(nL,1); zeros(nH,1)], [zeros(nL,1); ones(nH,1)] ]; % N x 2
    applyKinv = @(v) apply_Kinv_local(v, dbg.A, dbg.D_inv, dbg.R, dbg.perm);
    KinvZ = [applyKinv(Z(:,1)), applyKinv(Z(:,2))];                 % N x 2
    m_GLS = (Z.'*KinvZ) \ (Z.'*dbg.SIy);                            % 2x1
    dbg.Z = Z; dbg.KinvZ = KinvZ; dbg.m_GLS = m_GLS;

    % 4) Save hyperparams + meta
    ModelInfo.hyp = hyp;
    ModelInfo.debug_vecchia = dbg;
    


    % ---- Path diagnostic (as requested) ----
    % ---- Optional path diagnostic ----
if ModelInfo.show_path_diag
    w = which('vecchia_approx_space_time_corr_fast1');
    if isempty(w)
        error('Missing function vecchia_approx_space_time_corr_fast1 on the MATLAB path.');
    else
        disp(['[which] vecchia_approx_space_time_corr_fast1 -> ', w]);
    end
else
    if isempty(which('vecchia_approx_space_time_corr_fast1'))
        error('Missing function vecchia_approx_space_time_corr_fast1 on the MATLAB path.');
    end
end

end
% -------------------------------------------------------------------------
% Helper: build a fixed Vecchia neighbor list with width nn (time/space scaled)
function idxAll = build_fixed_neighbors_once(X, nn, ell_t0, ell_s0)
    if nargin < 3 || isempty(ell_t0) || ~isfinite(ell_t0) || ell_t0<=0
        ell_t0 = std(X(:,1)); if ~isfinite(ell_t0) || ell_t0<=0, ell_t0 = 1; end
    end
    if nargin < 4 || isempty(ell_s0) || ~isfinite(ell_s0) || ell_s0<=0
        sx = std(X(:,2)); sy = std(X(:,3));
        ell_s0 = mean([sx, sy]); if ~isfinite(ell_s0) || ell_s0<=0, ell_s0 = 1; end
    end
    n  = size(X,1);
    nn = min(nn, max(0,n-1));
    idxAll = zeros(n, nn, 'uint32');

    inv_t2 = 1/(ell_t0^2 + eps);
    inv_s2 = 1/(ell_s0^2 + eps);

    for i = 2:n
        prev = 1:(i-1);
        dt = (X(prev,1) - X(i,1)).^2 * inv_t2;
        dx = (X(prev,2) - X(i,2)).^2 * inv_s2;
        dy = (X(prev,3) - X(i,3)).^2 * inv_s2;
        r2 = dt + dx + dy;
        k = min(nn, numel(prev));
        if k > 0
            [~, I] = mink(r2, k);
            idxAll(i,1:k) = uint32(prev(I(:)'));
        end
    end
end

% -------------------------------------------------------------------------
% Helper: Corr-conditioned Vecchia with supplemental neighbor logic
function result = vecchia_approx_space_time_corr_fast1(locations, hyp_s, hyp_t, nn, eps_val, kernel, cand_mult, ell_t, ell_s, idxAll)
    % Corr-conditioned Vecchia with supplemental neighbor logic
    % (now correctly passes coordinates to k_space_time)

    if nargin < 4 || isempty(nn),         nn = 15; end
    if nargin < 5 || isempty(eps_val),    eps_val = 1e-7; end
    if eps_val <= 0,                       eps_val = 1e-7; end
    if nargin < 6 || isempty(kernel),     kernel = 'RBF'; end
    if nargin < 7 || isempty(cand_mult),  cand_mult = 4; end
    if nargin < 8 || isempty(ell_t),      ell_t = 1; end
    if nargin < 9 || isempty(ell_s),      ell_s = 1; end

    [n, ~] = size(locations);
    if n < 1, error('locations must be non-empty'); end
    if nn > n-1, nn = n-1; end

    use_prefilter = (n > 200 && cand_mult > 1);
    inv_ell_t2 = 1/(ell_t^2 + eps);
    inv_ell_s2 = 1/(ell_s^2 + eps);

    % self variances
    var_self = zeros(n,1);
    for j = 1:n
        vj = k_space_time(locations(j,:), locations(j,:), hyp_s, hyp_t, kernel);
        if ~isscalar(vj), vj = vj(1); end
        var_self(j) = max(vj, eps_val);
    end

    B_rows_c = cell(n,1); B_cols_c = cell(n,1); B_vals_c = cell(n,1);
    Di_vals  = zeros(n,1);

    hasIdx = (nargin >= 10) && ~isempty(idxAll);

    for i = 1:n
        if i == 1
            Di_vals(1)  = 1 / var_self(1);
            B_rows_c{1} = []; B_cols_c{1} = []; B_vals_c{1} = [];
            continue;
        end

        prev_idx = 1:(i-1);
        xi       = locations(i,:);  % 1×3

        % start with any fixed neighbors for this i (truncate to < i)
        fixed_list = [];
        if hasIdx
            row = double(idxAll(i, :));
            row = row(row > 0 & row < i);
            if ~isempty(row)
                fixed_list = unique(row(:).', 'stable');
            end
        end
        n_ind = fixed_list(1:min(nn, numel(fixed_list)));

        % supplement to reach nn using correlation
        k_needed = nn - numel(n_ind);
        if k_needed > 0
            rem_pool = setdiff(prev_idx, n_ind, 'stable');
            if ~isempty(rem_pool)
                if use_prefilter
                    dt = (locations(rem_pool,1) - xi(1)).^2 * inv_ell_t2;
                    dx = (locations(rem_pool,2) - xi(2)).^2 * inv_ell_s2;
                    dy = (locations(rem_pool,3) - xi(3)).^2 * inv_ell_s2;
                    r2 = dt + dx + dy;
                    cand_k = min(numel(rem_pool), max(k_needed, cand_mult * k_needed));
                    [~, Icand] = mink(r2, cand_k);
                    cand_idx = rem_pool(Icand);
                else
                    cand_idx = rem_pool;
                end

                % >>> FIX 1: pass coordinates, not indices
                K_cand = k_space_time(locations(cand_idx,:), xi, hyp_s, hyp_t, kernel);  % (cand × 1)
                if size(K_cand,2) > 1, K_cand = K_cand(:,1); end

                denom = sqrt(var_self(i) * var_self(cand_idx));
                denom(denom < eps_val) = eps_val;

                corr_cand = K_cand ./ denom;
                [~, pick_rel] = maxk(corr_cand, min(k_needed, numel(cand_idx)));
                n_ind = [n_ind, cand_idx(pick_rel)]; %#ok<AGROW>
            end
        end

        k = numel(n_ind);
        if k == 0
            Di_vals(i)  = 1 / var_self(i);
            B_rows_c{i} = []; B_cols_c{i} = []; B_vals_c{i} = [];
            continue;
        end

        Xi    = xi;                         % 1×3
        Xnbrs = locations(n_ind,:);         % k×3

        K_nn = k_space_time(Xnbrs, Xnbrs, hyp_s, hyp_t, kernel);
        if ~ismatrix(K_nn), K_nn = K_nn(:); end
        K_nn = 0.5*(K_nn + K_nn');
        K_nn = K_nn + eps_val*eye(k);

        % >>> FIX 2: pass coordinates, not indices
        K_i_n = k_space_time(Xnbrs, Xi, hyp_s, hyp_t, kernel);  % (k × 1)
        if size(K_i_n,2) > 1, K_i_n = K_i_n(:,1); end

        Ai = K_nn \ K_i_n;                       % k×1
        cond_var = var_self(i) - (K_i_n.') * Ai; % scalar
        cond_var = max(cond_var, eps_val);

        % assemble B (unit lower-triangular: B(i,n_ind) = -Ai)
        B_rows_c{i} = repmat(i, k, 1);
        B_cols_c{i} = n_ind(:);
        B_vals_c{i} = -Ai(:);

        Di_vals(i) = 1 / cond_var;
    end

    % build sparse outputs
    B_rows = double(vertcat(B_rows_c{:}));
    B_cols = double(vertcat(B_cols_c{:}));
    B_vals = vertcat(B_vals_c{:});

    B = sparse(B_rows, B_cols, B_vals, n, n);
    B = B + speye(n);

    Di = spdiags(Di_vals, 0, n, n);

    result.B  = B;
    result.Di = Di;
end

% -------------------------------------------------------------------------
% Helper: Apply K^{-1} using cached Vecchia/Woodbury pieces
function x = apply_Kinv_local(v, A, Dinv, R, p)
    Dy   = Dinv * v;
    rhs  = A.' * Dy;
    if ~isempty(p)
        rhsP = rhs(p);
        tmp  = R' \ rhsP;
        zP   = R  \ tmp;
        z    = zeros(size(rhs)); z(p) = zP;
    else
        z = R \ (R' \ rhs);
    end
    x = Dy - Dinv * (A * z);
end
