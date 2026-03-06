function run_best4_fminunc_and_oos()
% RUN_BEST4_FMINUNC_AND_OOS
% One-click:
% - data split HF train/test
% - 4 best configs (ordering, nn)
% - precompute neighbors ONCE (causal KNN on previous points, scaled metric)
% - fminunc optimization of hyp for each config (stable because neighbors fixed)
% - OOS HF predictions + plots (scatter, residuals vs time, calibration)
%
% Uses internal optimized likelihood: likelihoodVecchia_nonstat_GLS_v3_FAST
% (same as your v3, but: fixed neighbors + cached AMD permutation)
%
% Requires in path:
%   simulate_data, CM_nested1, k_space_time_v2
%   predict_calibratedCM3_fixed

    rng(123);
    kernelName = "RBF";

    %% ---------------- USER SETTINGS ----------------
    DO_OPT   = true;     % set false to skip fminunc and use hyp0
    maxIters = 60;       % fminunc iterations per config
    test_frac = 0.2;     % HF out-of-sample fraction
    nn_list_best4 = [40, 40, 20, 20]; % fixed by your previous experiment

    % Prediction opts (your function)
    optsPred = struct();
    optsPred.gamma_clip = [0.25, 4.0];
    optsPred.lambda_ridge = 1e-8;

    %% ---------------- DATA ----------------
    out = simulate_data(1, 1.0);
    X_L0 = [out.LF.t, out.LF.s1, out.LF.s2];  y_L0 = out.LF.fL(:);
    X_H0 = [out.HF.t, out.HF.s1, out.HF.s2];  y_H0 = out.HF.fH(:);

    nL = size(X_L0,1);
    nH = size(X_H0,1);
    fprintf('Dataset: nL=%d, nH=%d\n', nL, nH);

    %% ---------------- HF HOLDOUT (fixed) ----------------
    nTest = max(1, round(test_frac*nH));
    p = randperm(nH);
    te = p(1:nTest);
    tr = p(nTest+1:end);

    X_H_test = X_H0(te,:);  y_H_test = y_H0(te);
    X_H_tr0  = X_H0(tr,:);  y_H_tr0  = y_H0(tr);

    fprintf('HF split: nTrainH=%d, nTestH=%d\n', numel(tr), numel(te));

    %% ---------------- hyp0 (your default) ----------------
    hyp0 = zeros(11,1);
    hyp0(1)  = log(1.0);
    hyp0(2)  = log(0.20);
    hyp0(3)  = log(1.0);
    hyp0(4)  = log(0.20);
    hyp0(5)  = 0.6;
    hyp0(6)  = log(0.10);
    hyp0(7)  = log(0.10);
    hyp0(8)  = log(1.0);
    hyp0(9)  = log(1.00);
    hyp0(10) = log(1.0);
    hyp0(11) = log(1.00);

    %% ---------------- ORDERINGS (LF full; HF train subset) ----------------
    ordL = make_orderings(X_L0);
    ordH = make_orderings(X_H_tr0);

    best4 = {
        % name,         permL,              permH,               nn
        {'Time/Time',    ordL.time_major,    ordH.time_major,     nn_list_best4(1)}
        {'Rand/Rand',    ordL.rand,          ordH.rand,           nn_list_best4(2)}
        {'Orig/Orig',    ordL.orig,          ordH.orig,           nn_list_best4(3)}
        {'Space/Space',  ordL.space_major,   ordH.space_major,    nn_list_best4(4)}
    };

    %% ---------------- RESULTS TABLE ----------------
    R = table('Size',[0 10], ...
        'VariableTypes',{'string','double','double','double','double','double','double','double','double','double'}, ...
        'VariableNames',{'Config','nn','NLML_train','RMSE','MAE','NLPD','exitflag','iters','funcCount','nnzR'});

    %% ================= LOOP BEST4 =================
    for i = 1:numel(best4)
        name = string(best4{i}{1});
        pL   = best4{i}{2};
        pH   = best4{i}{3};
        nn   = best4{i}{4};

        fprintf('\n================ [%d/4] %s | nn=%d ================\n', i, name, nn);

        % Apply ordering
        X_L = X_L0(pL,:);      y_L = y_L0(pL);
        X_H = X_H_tr0(pH,:);   y_H = y_H_tr0(pH);

        % Build ModelInfo (kept compatible with your predictor expectations)
        M = struct();
        M.X_L = X_L; M.y_L = y_L;
        M.X_H = X_H; M.y_H = y_H;

        M.jitter  = 1e-8;
        M.nn_size = nn;
        M.conditioning = "Corr";
        M.kernel  = kernelName;
        M.MeanFunction = "zero";
        M.RhoFunction  = "constant";
        M.GLSType = "constant";
        M.cand_mult = 10;
        M.cov_type  = kernelName;  % your predictor checks this

        % ---- PRECOMPUTE NEIGHBORS ONCE (critical for fminunc stability) ----
        % Use metric scaled by hyp0 lengthscales (not dummy!)
        ell_t_L = exp(hyp0(2));  ell_s_L = exp(hyp0(9));
        ell_t_H = exp(hyp0(4));  ell_s_H = exp(hyp0(11));

        M.idxL_precomputed = causal_knn_indices(X_L, nn, ell_t_L, ell_s_L);
        M.idxH_precomputed = causal_knn_indices(X_H, nn, ell_t_H, ell_s_H);

        % Reset cached perm for this config
        M.perm_fixed = [];

        % ---- Choose hyp to use (opt or hyp0) ----
        if DO_OPT
            [hypUse, fval, exitflag, output] = run_fminunc(@(h)obj_safe(h, M), hyp0, maxIters);
            fprintf('fminunc: fval=%.6f exitflag=%d iters=%d funcCount=%d\n', ...
                fval, exitflag, output.iterations, output.funcCount);
        else
            hypUse = hyp0;
            exitflag = NaN; output.iterations = 0; output.funcCount = 0;
        end

        % ---- Fit once (populate caches for prediction) ----
        [NLML_train, M] = likelihoodVecchia_nonstat_GLS_v3_FAST(hypUse, M);

        % ---- Predict OOS ----
        [muH, s2H] = predict_calibratedCM3_fixed(X_H_test, M, optsPred);

        ytrue = y_H_test(:);
        yhat  = muH(:);
        e = yhat - ytrue;

        rmse = sqrt(mean(e.^2));
        mae  = mean(abs(e));
        s2 = max(s2H(:), 1e-12);
        nlpd = mean(0.5*log(2*pi*s2) + 0.5*(e.^2)./s2);

        nnzR = nnz(M.R);

        fprintf('Train NLML=%.2f | OOS RMSE=%.4f MAE=%.4f NLPD=%.4f | nnzR=%g\n', ...
            NLML_train, rmse, mae, nlpd, nnzR);

        R = [R; {name, nn, NLML_train, rmse, mae, nlpd, exitflag, output.iterations, output.funcCount, nnzR}]; %#ok<AGROW>

        % ---- PLOTS (OOS) ----
        plot_oos(name, nn, X_H_test, ytrue, yhat, s2, rmse);
    end

    fprintf('\n================ FINAL SUMMARY ================\n');
    disp(R);
end

%% ========================================================================
%% OPTIMIZER: fminunc wrapper with robust objective + soft penalties
%% ========================================================================
function [xopt, fopt, exitflag, output] = run_fminunc(obj, x0, maxIters)
    opts = optimoptions('fminunc', ...
        'Algorithm','quasi-newton', ...
        'Display','iter', ...
        'MaxIterations', maxIters, ...
        'MaxFunctionEvaluations', 6*maxIters, ...
        'StepTolerance', 1e-6, ...
        'OptimalityTolerance', 1e-6);

    [xopt, fopt, exitflag, output] = fminunc(obj, x0, opts);
end

function f = obj_safe(h, M)
    pen = soft_penalty(h);

    try
        [NLML, ~] = likelihoodVecchia_nonstat_GLS_v3_FAST(h, M); % do not mutate M in objective
        if ~isfinite(NLML), f = 1e20 + pen; else, f = NLML + pen; end
    catch
        f = 1e20 + pen;
    end
end

function p = soft_penalty(h)
    % Soft constraints help fminunc avoid crazy regions
    p = 0;

    % keep log params moderate
    ids = [1 2 3 4 8 9 10 11];
    for k = ids
        p = p + quad_outside(h(k), -10, 10, 1e2);
    end

    % rho
    p = p + quad_outside(h(5), -3, 3, 1e3);

    % log eps
    p = p + quad_outside(h(6), -20, 2, 1e4);
    p = p + quad_outside(h(7), -20, 2, 1e4);
end

function q = quad_outside(x, lo, hi, w)
    q = 0;
    if x < lo, q = w*(lo-x)^2; end
    if x > hi, q = w*(x-hi)^2; end
end

%% ========================================================================
%% OPTIMIZED LIKELIHOOD: fixed neighbors + cached AMD permutation
%% (Mathematically same as your v3; only performance/stability changes)
%% ========================================================================
function [NLML, M] = likelihoodVecchia_nonstat_GLS_v3_FAST(hyp, M)
    % Copy of your v3 logic with:
    %  - always uses idxL_precomputed/idxH_precomputed (neighbors fixed)
    %  - caches perm (symamd) once in M.perm_fixed
    %  - returns updated M (for prediction)

    X_L = M.X_L; X_H = M.X_H;
    y_L = M.y_L; y_H = M.y_H;
    y = [y_L; y_H];
    nL = size(X_L,1); nH = size(X_H,1);
    N  = nL + nH;

    nn_size = M.nn_size;
    jitter  = M.jitter;
    kernel  = M.kernel;

    % hyp mapping (same as your v3)
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    rho_const  = hyp(5);
    eps_LF     = exp(hyp(6));  eps_HF   = exp(hyp(7));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    % Vecchia LIGHT with precomputed indices (fixed neighbors)
    if ~isfield(M,'idxL_precomputed') || ~isfield(M,'idxH_precomputed')
        error('Missing idx*_precomputed. This FAST version requires fixed neighbors.');
    end

    result_LF = vecchia_approx_LIGHT_final(X_L, [s_sig_LF_s, s_ell_LF], [s_sig_LF_t, t_ell_LF], kernel, M.idxL_precomputed);
    result_HF = vecchia_approx_LIGHT_final(X_H, [s_sig_HF_s, s_ell_HF], [s_sig_HF_t, t_ell_HF], kernel, M.idxH_precomputed);

    Ki_L = result_LF.B' * result_LF.Di * result_LF.B;
    Ki_D = result_HF.B' * result_HF.Di * result_HF.B;

    log_det_W = -(sum(log(diag(result_LF.Di))) + sum(log(diag(result_HF.Di))));

    % rho model (only constant here)
    rho_input = rho_const;
    NonStat = "F";
    rho_H = rho_const * ones(nH,1);

    % build nested system
    [A, D, D_inv] = CM_nested1(X_L, X_H, rho_input, eps_LF, eps_HF, NonStat, y_L, y_H);

    H = blkdiag(Ki_L, Ki_D) + A' * D_inv * A + speye(size(A,2)) * jitter;

    % cached permutation
    if isfield(M,'perm_fixed') && ~isempty(M.perm_fixed)
        perm = M.perm_fixed;
    else
        perm = symamd(H);
        M.perm_fixed = perm;
    end

    [R, pchol] = chol(H(perm, perm));
    if pchol > 0, error('H not SPD'); end

    log_det_H = 2 * sum(log(diag(R)));

    % GLS design (constant: two intercepts)
    G_gls = [[ones(nL,1); zeros(nH,1)], [zeros(nL,1); ones(nH,1)]];

    % Apply K^{-1} to [y, G]
    Kinv_yG = apply_Kinv_local([y, G_gls], A, D_inv, R, perm);
    Kinv_y  = Kinv_yG(:,1);
    Kinv_G  = Kinv_yG(:,2:end);

    beta_gls = (G_gls' * Kinv_G) \ (G_gls' * Kinv_y);
    SIy_tilde = Kinv_y - Kinv_G * beta_gls;
    y_tilde = y - G_gls * beta_gls;

    % NLML
    term1 = 0.5 * (y_tilde' * SIy_tilde);
    term2 = 0.5 * (log_det_W + log_det_H + sum(log(diag(D))));
    term3 = 0.5 * N * log(2*pi);
    NLML = term1 + term2 + term3;

    % Cache for prediction (compatible with your predict_calibratedCM3_fixed)
    M.hyp = hyp;
    M.beta_gls = beta_gls;
    M.SIy = SIy_tilde;
    M.rho_H = rho_H;
    M.A = A;
    M.D_inv = D_inv;
    M.R = R;
    M.perm = perm;
    M.G_gls = G_gls;

    dbg.m_GLS = beta_gls;
    dbg.A = A;
    dbg.D_inv = D_inv;
    dbg.R = R;
    dbg.perm = perm;
    M.debug_vecchia = dbg;

    % (optional) store H if you want sparsity stats
    % M.H = H;
end

function X = apply_Kinv_local(V, A, Dinv, R, p)
    DY = Dinv * V;
    RHS = A' * DY;
    RHSP = RHS(p, :);
    ZP = R \ (R' \ RHSP);
    Z = zeros(size(RHS));
    Z(p,:) = ZP;
    X = DY - Dinv * (A * Z);
end

function result = vecchia_approx_LIGHT_final(locations, hyp_s, hyp_t, kernel, n_ind_mat)
    [n, ~] = size(locations);
    eps_val = 1e-7;
    B_rows_c = cell(n,1); B_cols_c = cell(n,1); B_vals_c = cell(n,1);
    Di_vals = zeros(n,1);

    var_self = max(k_space_time_v2(locations, [], hyp_s, hyp_t, kernel), eps_val);

    for i = 1:n
        n_ind = n_ind_mat(i, :);
        n_ind = n_ind(n_ind > 0);

        if ~isempty(n_ind)
            xi = locations(i,:);
            Xnbrs = locations(n_ind,:);
            K_nn = k_space_time_v2(Xnbrs, Xnbrs, hyp_s, hyp_t, kernel);
            K_nn = 0.5*(K_nn + K_nn') + eps_val*eye(length(n_ind));
            K_i_n = k_space_time_v2(Xnbrs, xi, hyp_s, hyp_t, kernel);

            Ai = K_nn \ K_i_n(:);
            cond_var = max(var_self(i) - K_i_n(:)' * Ai, eps_val);

            B_rows_c{i} = repmat(i, length(n_ind), 1);
            B_cols_c{i} = n_ind(:);
            B_vals_c{i} = -Ai;
            Di_vals(i) = 1 / cond_var;
        else
            Di_vals(i) = 1 / var_self(i);
        end
    end

    result.B  = sparse(vertcat(B_rows_c{:}), vertcat(B_cols_c{:}), vertcat(B_vals_c{:}), n, n) + speye(n);
    result.Di = spdiags(Di_vals, 0, n, n);
end

%% ========================================================================
%% PRECOMPUTE: causal KNN on previous points (fixed neighbors, smooth objective)
%% ========================================================================
function idx = causal_knn_indices(X, nn, ell_t, ell_s)
    % For each i, choose up to nn nearest among {1..i-1}
    % distance = (dt/ell_t)^2 + (dx/ell_s)^2 + (dy/ell_s)^2
    n = size(X,1);
    idx = zeros(n, nn);
    if n <= 1, return; end

    % scale coords
    Xt = X(:,1) / max(ell_t, 1e-12);
    Xx = X(:,2) / max(ell_s, 1e-12);
    Xy = X(:,3) / max(ell_s, 1e-12);

    for i = 2:n
        dt = Xt(1:i-1) - Xt(i);
        dx = Xx(1:i-1) - Xx(i);
        dy = Xy(1:i-1) - Xy(i);
        d2 = dt.^2 + dx.^2 + dy.^2;

        k = min(nn, i-1);
        [~, ord] = mink(d2, k);
        idx(i,1:k) = ord(:)';
    end
end

%% ========================================================================
%% ORDERINGS + PLOTS
%% ========================================================================
function ord = make_orderings(X)
    n = size(X,1);
    ord.orig = (1:n)';
    [~, ord.time_major]  = sortrows(X, [1 2 3]);
    [~, ord.space_major] = sortrows(X, [2 3 1]);
    ord.rand = randperm(n)';
end

function plot_oos(name, nn, Xtest, ytrue, yhat, s2, rmse)

    % Sort test points by time for line plot
    [t_sorted, ord] = sort(Xtest(:,1));
    ytrue_s = ytrue(ord);
    yhat_s  = yhat(ord);
    s2_s    = s2(ord);
    sigma_s = sqrt(max(s2_s, 0));

    % ===== 1) Line plot: Truth vs Prediction =====
    figure('Name', sprintf('OOS lines | %s nn=%d', name, nn));

    hold on;
    plot(t_sorted, ytrue_s, 'k-', 'LineWidth', 1.5);
    plot(t_sorted, yhat_s,  'r-', 'LineWidth', 1.5);

    % 95% interval
    upper = yhat_s + 2*sigma_s;
    lower = yhat_s - 2*sigma_s;

    fill([t_sorted; flipud(t_sorted)], ...
         [upper; flipud(lower)], ...
         [1 0.8 0.8], ...
         'EdgeColor','none', ...
         'FaceAlpha',0.4);

    plot(t_sorted, yhat_s,  'r-', 'LineWidth', 1.5); % redraw on top

    grid on;
    xlabel('t (HF test)');
    ylabel('Value');
    title(sprintf('%s | nn=%d | RMSE=%.4f', name, nn, rmse));
    legend('True','Predicted','95% CI','Location','best');

    hold off;

    % ===== 2) Residual line plot =====
    figure('Name', sprintf('OOS residual line | %s nn=%d', name, nn));

    residuals = yhat_s - ytrue_s;
    plot(t_sorted, residuals, 'b-', 'LineWidth', 1.5);
    yline(0,'k--');

    grid on;
    xlabel('t (HF test)');
    ylabel('Residual');
    title(sprintf('Residuals over time | %s | nn=%d', name, nn));
end
