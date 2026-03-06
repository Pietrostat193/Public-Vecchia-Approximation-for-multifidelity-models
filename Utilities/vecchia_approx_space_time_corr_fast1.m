function result = vecchia_approx_space_time_corr_fast1(locations, hyp_s, hyp_t, nn, eps_val, kernel, cand_mult, ell_t, ell_s, idxAll)
    % (Tua funzione originale, mantenuta per compatibilità e pre-calcolo)
    [n, ~] = size(locations);
    nn = min(nn, n-1);
    inv_ell_t2 = 1/(ell_t^2 + eps); inv_ell_s2 = 1/(ell_s^2 + eps);
    var_self = zeros(n,1);
    for j = 1:n
        var_self(j) = max(k_space_time(locations(j,:), locations(j,:), hyp_s, hyp_t, kernel), eps_val);
    end
    B_rows_c = cell(n,1); B_cols_c = cell(n,1); B_vals_c = cell(n,1);
    Di_vals = zeros(n,1);
    hasIdx = (nargin >= 10) && ~isempty(idxAll);
    for i = 1:n
        if i == 1, Di_vals(1) = 1 / var_self(1); continue; end
        prev_idx = 1:(i-1); xi = locations(i,:);
        fixed_list = [];
        if hasIdx, row = double(idxAll(i, :)); fixed_list = row(row > 0 & row < i); end
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
                corr_cand = K_cand(:) ./ max(sqrt(var_self(i)*var_self(cand_idx)), eps_val);
                [~, pick] = maxk(corr_cand, min(k_needed, numel(cand_idx)));
                n_ind = [n_ind, cand_idx(pick)];
            end
        end
        if ~isempty(n_ind)
            Xnbrs = locations(n_ind,:);
            K_nn = k_space_time(Xnbrs, Xnbrs, hyp_s, hyp_t, kernel);
            K_nn = 0.5*(K_nn + K_nn') + eps_val*eye(length(n_ind));
            K_i_n = k_space_time(Xnbrs, xi, hyp_s, hyp_t, kernel);
            Ai = K_nn \ K_i_n(:);
            Di_vals(i) = 1 / max(var_self(i) - K_i_n(:)' * Ai, eps_val);
            B_rows_c{i} = repmat(i, length(n_ind), 1); B_cols_c{i} = n_ind(:); B_vals_c{i} = -Ai;
        else
            Di_vals(i) = 1 / var_self(i);
        end
    end
    result.B = sparse(vertcat(B_rows_c{:}), vertcat(B_cols_c{:}), vertcat(B_vals_c{:}), n, n) + speye(n);
    result.Di = spdiags(Di_vals, 0, n, n);
end