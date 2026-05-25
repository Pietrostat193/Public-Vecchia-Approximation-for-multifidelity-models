function result = vecchia_approx_space_time_corr_fast(locations, hyp_s, hyp_t, nn, eps_val, kernel, cand_mult, ell_t, ell_s, idxFixed)
% vecchia_approx_space_time_corr_fast
% Correlation-based Vecchia with optional FIXED neighbors.
% Builds factors (B, Di) so that an approximation to the COVARIANCE precision is:
%     Ki ≈ B' * Di * B,
% where the construction is done in CORRELATION space and then scaled back.
%
% Inputs
%   locations : [n x 3] matrix [t, x, y] (double is safest)
%   hyp_s     : [s_sig, s_ell]
%   hyp_t     : [t_sig, t_ell]
%   nn        : number of neighbors (default 15)
%   eps_val   : covariance nugget to be mimicked in corr space (default 1e-10)
%   kernel    : 'RBF' or 'Matern' (used by k_space_time)
%   cand_mult : candidate multiplier for prefilter (default 4)
%   ell_t     : time lengthscale for neighbor metric (defaults to hyp_t(2))
%   ell_s     : space lengthscale for neighbor metric (defaults to hyp_s(2))
%   idxFixed  : optional [n x nn] fixed neighbor indices (< i), any class
%
% Outputs
%   result.B        : sparse unit-lower + off-diagonal regression (n x n)
%   result.Di       : sparse diagonal (n x n) in COVARIANCE space
%   result.Di_corr  : sparse diagonal in CORRELATION space (diagnostic)
%   result.sigma    : s_sig * t_sig used for corr <-> cov scaling
%   result.eps_corr : corr nugget used internally
%   result.var_self_corr : 1 + eps_corr (corr-space self variance)
%
% Notes
%   - All sparse subscript arrays are forced to DOUBLE to avoid MATLAB type errors
%     when idxFixed is uint32 / int32, etc.
%   - Robustifies Cholesky with a tiny diagonal lift and jitter escalation.

    % ---------------- defaults / guards ----------------
    if nargin < 4 || isempty(nn),        nn = 15;         end
    if nargin < 5 || isempty(eps_val),   eps_val = 1e-10; end
    if nargin < 6 || isempty(kernel),    kernel = 'RBF';  end
    if nargin < 7 || isempty(cand_mult), cand_mult = 4;   end

    haveFixed = (nargin >= 10) && ~isempty(idxFixed);

    % Ensure doubles for numeric stability / sparse indexing
    if ~isa(locations, 'double'), locations = double(locations); end

    [n, d] = size(locations);
    if d ~= 3
        error('locations must be n x 3 = [t, x, y].');
    end
    if nn > n-1, error('nn must be < n'); end

    % ---------------- separable amplitude and corr nugget ----------------
    % σ = s_sig * t_sig
    sigma = hyp_s(1) * hyp_t(1);
    if ~(isfinite(sigma) && sigma > 0), sigma = 1; end

    % eps_corr so that adding eps_val to covariance corresponds to adding
    % eps_corr in correlation space: K_corr = (K_cov / sigma)
    eps_corr = eps_val / max(sigma, 1e-30);
    var_self_corr = 1 + eps_corr;  % diag of corr-space with nugget

    % ---------------- neighbor metric scalings (for non-fixed) -----------
    if ~haveFixed
        if nargin < 8 || isempty(ell_t), ell_t = hyp_t(2); end
        if nargin < 9 || isempty(ell_s), ell_s = hyp_s(2); end
        inv_ell_t2 = 1 / max(ell_t^2, 1e-30);
        inv_ell_s2 = 1 / max(ell_s^2, 1e-30);
    end

    % ---------------- containers ----------------
    B_rows_c = cell(n,1); B_cols_c = cell(n,1); B_vals_c = cell(n,1);
    Di_corr_vals = zeros(n,1);   % diagonal in correlation space

    % ---------------- main loop ----------------
    for i = 1:n
        if i == 1
            % No predecessors
            Di_corr_vals(1) = 1 / var_self_corr;
            B_rows_c{1} = []; B_cols_c{1} = []; B_vals_c{1} = [];
            continue
        end

        % ----- neighbor selection -----
        if haveFixed
            % Accept either double/int/uint indices; cast later to double
            cand_idx = idxFixed(i, :);
            cand_idx = cand_idx(cand_idx < i & cand_idx > 0);
            if isempty(cand_idx)
                n_ind = [];
            else
                n_ind = cand_idx(1:min(nn, numel(cand_idx)));
            end
        else
            prev_idx = 1:(i-1);
            if n > 200 && cand_mult > 1
                % ell-weighted distance prefilter
                dt = (locations(prev_idx,1) - locations(i,1)).^2 * inv_ell_t2;
                dx = (locations(prev_idx,2) - locations(i,2)).^2 * inv_ell_s2;
                dy = (locations(prev_idx,3) - locations(i,3)).^2 * inv_ell_s2;
                r2 = dt + dx + dy;
                cand_k = min(numel(prev_idx), cand_mult * nn);
                [~, cand_rel] = mink(r2, cand_k);
                cand_idx = prev_idx(cand_rel);
            else
                cand_idx = prev_idx;
            end

            % final k-NN in the same ell-weighted metric
            dt = (locations(cand_idx,1) - locations(i,1)).^2 * inv_ell_t2;
            dx = (locations(cand_idx,2) - locations(i,2)).^2 * inv_ell_s2;
            dy = (locations(cand_idx,3) - locations(i,3)).^2 * inv_ell_s2;
            r2 = dt + dx + dy;
            k_nn = min(nn, numel(cand_idx));
            if k_nn > 0
                [~, pick_rel] = mink(r2, k_nn);
                n_ind = cand_idx(pick_rel);
            else
                n_ind = [];
            end
        end

        % force double for sparse indices
        n_ind = double(n_ind);

        k = numel(n_ind);
        if k == 0
            Di_corr_vals(i) = 1 / var_self_corr;
            B_rows_c{i} = []; B_cols_c{i} = []; B_vals_c{i} = [];
            continue
        end

        Xi = locations(i,:);
        XN = locations(n_ind,:);

        % ----- correlation blocks (divide covariance by σ) -----
        K_nn_cov = k_space_time(XN, XN, hyp_s, hyp_t, kernel);
        K_nn     = K_nn_cov / sigma;

        % robust diagonal lift in corr space
        % ensure at least eps_corr on the diagonal; tiny trace term for stability
        lift = eps_corr + 1e-12 * trace(K_nn) / max(k,1);
        K_nn = (K_nn + K_nn')/2 + lift * eye(k);

        K_n_i_cov = k_space_time(XN, Xi, hyp_s, hyp_t, kernel);
        K_n_i     = K_n_i_cov(:) / sigma;   % k x 1

        % ----- stable solve via Cholesky (with escalation if needed) -----
        add = 0;
        for tries = 1:5
            [R, pflag] = chol(K_nn + add*eye(k), 'lower');
            if pflag == 0, break; end
            add = max(1e-10, 10*max(add, lift));
        end
        if pflag ~= 0
            % last resort: make it diagonal-dominant
            R = chol(K_nn + (add + 1e-6)*eye(k), 'lower');
        end

        v  = R \ K_n_i;
        Ai = R' \ v;  % K_nn^{-1} * K_n_i

        denom = var_self_corr - (K_n_i.' * Ai);
        if ~isfinite(denom) || denom <= 0
            denom = denom + max(eps_corr, 1e-12);
        end
        Di_corr_vals(i) = 1 / denom;

        % store -Ai into row i of B (ensure double subscripts and values)
        B_rows_c{i} = double(repmat(i, k, 1));
        B_cols_c{i} = double(n_ind(:));
        B_vals_c{i} = double(-Ai);
    end

    % ---------------- assemble sparse B & Di ----------------
    B_rows = double(vertcat(B_rows_c{:}));
    B_cols = double(vertcat(B_cols_c{:}));
    B_vals = double(vertcat(B_vals_c{:}));
    B  = sparse(B_rows, B_cols, B_vals, n, n);
    B  = B + speye(n);

    % Scale back from CORR to COV: Ki = (1/σ) * Ki_corr
    Di_cov_vals = (1 / sigma) * Di_corr_vals;
    Di  = spdiags(Di_cov_vals, 0, n, n);

    % ---------------- pack result ----------------
    result.B              = B;
    result.Di             = Di;
    % diagnostics
    result.Di_corr        = spdiags(Di_corr_vals, 0, n, n);
    result.sigma          = sigma;
    result.eps_corr       = eps_corr;
    result.var_self_corr  = var_self_corr;
end
