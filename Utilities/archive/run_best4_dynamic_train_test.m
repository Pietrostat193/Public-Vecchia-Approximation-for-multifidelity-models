function run_best4_dynamic_train_test()
% RUN_BEST4_DYNAMIC_TRAIN_TEST
% One-click script:
% - uses simulate_data_dynamic() which already provides HF_train/HF_test split by station
% - evaluates 4 pre-chosen best configs (ordering, nn)
% - (optional) optimizes hyp with fminunc (stable because neighbors fixed)
% - predicts on HF_test and plots TIME-SERIES LINES per test station
%
% Requires in path:
%   simulate_data_dynamic
%   CM_nested1
%   k_space_time_v2
%   predict_calibratedCM3_fixed

    rng(123);
    kernelName = "RBF";

    %% ---------------- USER SETTINGS ----------------
    DO_OPT   = true;       % set false if you want no optimization
    maxIters = 60;         % fminunc iters per config

    % How many test stations to plot (to avoid 36 subplots if too many)
    maxStationsToPlot = 12;   % set Inf to plot all test stations

    % Data generation config
    cfg = struct();
    cfg.n_time  = 20;
    cfg.n_space = 6;
    cfg.target_corr_time   = 0.8;
    cfg.target_corr_spaceL = 0.72;
    cfg.target_corr_spaceD = 0.72;
    cfg.sigma_L2        = 2.0;
    cfg.sigma_d2        = 0.8;
    cfg.sigma_noise_L   = 0.3;
    cfg.sigma_noise_dd2 = 0.7;
    cfg.rho             = 0.6;

    seed = 1;
    train_fraction = 0.5;  % fraction of stations in HF_train

    %% ---------------- SIMULATE (already split) ----------------
    out = simulate_data_dynamic(seed, train_fraction, cfg);

    % LF FULL (all stations)
    LF_tbl = out.LF;

    % HF TRAIN / TEST already split by station
    HF_tr_tbl = out.HF_train;
    HF_te_tbl = out.HF_test;

    % Build matrices
    X_L0 = [LF_tbl.t, LF_tbl.s1, LF_tbl.s2];
    y_L0 = LF_tbl.fL(:);

    X_Htr0 = [HF_tr_tbl.t, HF_tr_tbl.s1, HF_tr_tbl.s2];
    y_Htr0 = HF_tr_tbl.fH(:);

    X_Hte  = [HF_te_tbl.t, HF_te_tbl.s1, HF_te_tbl.s2];
    y_Hte  = HF_te_tbl.fH(:);

    nL = size(X_L0,1);
    nHtr = size(X_Htr0,1);
    nHte = size(X_Hte,1);

    fprintf('LF rows=%d | HF_train rows=%d | HF_test rows=%d\n', nL, nHtr, nHte);

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

    %% ---------------- prediction opts (your function) ----------------
    optsPred = struct();
    optsPred.gamma_clip = [0.25, 4.0];
    optsPred.lambda_ridge = 1e-8;

    %% ---------------- ORDERINGS ----------------
    ordL  = make_orderings(X_L0);
    ordHtr = make_orderings(X_Htr0);

    %% ---------------- BEST 4 configs (from your earlier experiment) ----------------
    best4 = {
        % name,          permL,             permHtrain,           nn
        {'Time/Time',     ordL.time_major,   ordHtr.time_major,    40}
        {'Rand/Rand',     ordL.rand,         ordHtr.rand,          40}
        {'Orig/Orig',     ordL.orig,         ordHtr.orig,          20}
        {'Space/Space',   ordL.space_major,  ordHtr.space_major,   20}
    };

    %% ---------------- RESULTS TABLE ----------------
    R = table('Size',[0 9], ...
        'VariableTypes',{'string','double','double','double','double','double','double','double','double'}, ...
        'VariableNames',{'Config','nn','NLML_train','RMSE','MAE','NLPD','exitflag','iters','funcCount'});

    %% ================= LOOP BEST4 =================
    for i = 1:numel(best4)
        name = string(best4{i}{1});
        pL   = best4{i}{2};
        pH   = best4{i}{3};
        nn   = best4{i}{4};

        fprintf('\n================ [%d/4] %s | nn=%d ================\n', i, name, nn);

        % Apply ordering to TRAIN DATA
        X_L = X_L0(pL,:);     y_L = y_L0(pL);
        X_H = X_Htr0(pH,:);   y_H = y_Htr0(pH);

        % ModelInfo for this config (compatible with your predictor)
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
        M.cov_type  = kernelName;

        % Precompute neighbors ONCE (stabilizza fminunc e accelera)
        ell_t_L = exp(hyp0(2));  ell_s_L = exp(hyp0(9));
        ell_t_H = exp(hyp0(4));  ell_s_H = exp(hyp0(11));

        M.idxL_precomputed = causal_knn_indices(X_L, nn, ell_t_L, ell_s_L);
        M.idxH_precomputed = causal_knn_indices(X_H, nn, ell_t_H, ell_s_H);

        % Cache perm for AMD (reset per config)
        M.perm_fixed = [];

        % Choose hyp
        if DO_OPT
            [hypUse, fval, exitflag, output] = run_fminunc(@(h)obj_safe(h, M), hyp0, maxIters);
            fprintf('fminunc: fval=%.6f exitflag=%d iters=%d funcCount=%d\n', ...
                fval, exitflag, output.iterations, output.funcCount);
        else
            hypUse = hyp0;
            exitflag = NaN;
            output.iterations = 0;
            output.funcCount = 0;
        end

        % Fit once to populate caches in M
        [NLML_train, Mfit] = likelihoodVecchia_nonstat_GLS_v3_FAST(hypUse, M);

        % Predict on HF_test (true OOS stations)
        [muH, s2H] = predict_calibratedCM3_fixed(X_Hte, Mfit, optsPred);

        % Metrics
        ytrue = y_Hte(:);
        yhat  = muH(:);
        e = yhat - ytrue;

        rmse = sqrt(mean(e.^2));
        mae  = mean(abs(e));
        s2 = max(s2H(:), 1e-12);
        nlpd = mean(0.5*log(2*pi*s2) + 0.5*(e.^2)./s2);

        fprintf('Train NLML=%.2f | OOS RMSE=%.4f MAE=%.4f NLPD=%.4f\n', ...
            NLML_train, rmse, mae, nlpd);

        R = [R; {name, nn, NLML_train, rmse, mae, nlpd, exitflag, output.iterations, output.funcCount}]; %#ok<AGROW>

        % Plot LINES per station (use loc_id from HF_test table)
        plot_oos_lines_per_station(name, nn, HF_te_tbl, yhat, s2H, rmse, maxStationsToPlot);
    end

    fprintf('\n================ FINAL SUMMARY ================\n');
    disp(R);
end

%% ========================================================================
%% Optimizer wrappers
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
        [NLML, ~] = likelihoodVecchia_nonstat_GLS_v3_FAST(h, M);
        if ~isfinite(NLML), f = 1e20 + pen; else, f = NLML + pen; end
    catch
        f = 1e20 + pen;
    end
end

function p = soft_penalty(h)
    p = 0;
    ids = [1 2 3 4 8 9 10 11];
    for k = ids
        p = p + quad_outside(h(k), -10, 10, 1e2);
    end
    p = p + quad_outside(h(5), -3, 3, 1e3);
    p = p + quad_outside(h(6), -20, 2, 1e4);
    p = p + quad_outside(h(7), -20, 2, 1e4);
end

function q = quad_outside(x, lo, hi, w)
    q = 0;
    if x < lo, q = w*(lo-x)^2; end
    if x > hi, q = w*(x-hi)^2; end
end

%% ========================================================================
%% Optimized likelihood (fixed neighbors + cached AMD perm)
%% ========================================================================
function [NLML, M] = likelihoodVecchia_nonstat_GLS_v3_FAST(hyp, M)

    X_L = M.X_L; X_H = M.X_H;
    y_L = M.y_L; y_H = M.y_H;
    y = [y_L; y_H];
    nL = size(X_L,1); nH = size(X_H,1);
    N  = nL + nH;

    jitter  = M.jitter;
    kernel  = M.kernel;

    % hyp mapping (same as your v3)
    s_sig_LF_t = exp(hyp(1));  t_ell_LF = exp(hyp(2));
    s_sig_HF_t = exp(hyp(3));  t_ell_HF = exp(hyp(4));
    rho_const  = hyp(5);
    eps_LF     = exp(hyp(6));  eps_HF   = exp(hyp(7));
    s_sig_LF_s = exp(hyp(8));  s_ell_LF = exp(hyp(9));
    s_sig_HF_s = exp(hyp(10)); s_ell_HF = exp(hyp(11));

    % Vecchia LIGHT (neighbors fixed)
    result_LF = vecchia_approx_LIGHT_final(X_L, [s_sig_LF_s, s_ell_LF], [s_sig_LF_t, t_ell_LF], kernel, M.idxL_precomputed);
    result_HF = vecchia_approx_LIGHT_final(X_H, [s_sig_HF_s, s_ell_HF], [s_sig_HF_t, t_ell_HF], kernel, M.idxH_precomputed);

    Ki_L = result_LF.B' * result_LF.Di * result_LF.B;
    Ki_D = result_HF.B' * result_HF.Di * result_HF.B;

    log_det_W = -(sum(log(diag(result_LF.Di))) + sum(log(diag(result_HF.Di))));

    rho_input = rho_const;
    NonStat = "F";
    rho_H = rho_const * ones(nH,1);

    [A, D, D_inv] = CM_nested1(X_L, X_H, rho_input, eps_LF, eps_HF, NonStat, y_L, y_H);
    H = blkdiag(Ki_L, Ki_D) + A' * D_inv * A + speye(size(A,2)) * jitter;

    if isfield(M,'perm_fixed') && ~isempty(M.perm_fixed)
        perm = M.perm_fixed;
    else
        perm = symamd(H);
        M.perm_fixed = perm;
    end

    [R, pchol] = chol(H(perm, perm));
    if pchol > 0, error('H not SPD'); end

    log_det_H = 2 * sum(log(diag(R)));

    % GLS design: two intercepts
    G_gls = [[ones(nL,1); zeros(nH,1)], [zeros(nL,1); ones(nH,1)]];

    Kinv_yG = apply_Kinv_local([y, G_gls], A, D_inv, R, perm);
    Kinv_y  = Kinv_yG(:,1);
    Kinv_G  = Kinv_yG(:,2:end);

    beta_gls = (G_gls' * Kinv_G) \ (G_gls' * Kinv_y);
    SIy_tilde = Kinv_y - Kinv_G * beta_gls;
    y_tilde = y - G_gls * beta_gls;

    term1 = 0.5 * (y_tilde' * SIy_tilde);
    term2 = 0.5 * (log_det_W + log_det_H + sum(log(diag(D))));
    term3 = 0.5 * N * log(2*pi);
    NLML = term1 + term2 + term3;

    % Cache for prediction
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
%% Precompute neighbors: causal KNN among previous points
%% ========================================================================
function idx = causal_knn_indices(X, nn, ell_t, ell_s)
    n = size(X,1);
    idx = zeros(n, nn);
    if n <= 1, return; end

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
%% Orderings
%% ========================================================================
function ord = make_orderings(X)
    n = size(X,1);
    ord.orig = (1:n)';
    [~, ord.time_major]  = sortrows(X, [1 2 3]);
    [~, ord.space_major] = sortrows(X, [2 3 1]);
    ord.rand = randperm(n)';
end

%% ========================================================================
%% PLOTS: time-series lines per station (HF_test stations)
%% ========================================================================
function plot_oos_lines_per_station(name, nn, HF_te_tbl, yhat, s2H, rmse, maxStationsToPlot)

    locs = unique(HF_te_tbl.loc_id);
    if isfinite(maxStationsToPlot)
        locs = locs(1:min(numel(locs), maxStationsToPlot));
    end
    nLoc = numel(locs);

    % grid size for subplots
    ncols = ceil(sqrt(nLoc));
    nrows = ceil(nLoc / ncols);

    figure('Name', sprintf('OOS LINES per station | %s nn=%d', name, nn));
    sgtitle(sprintf('%s | nn=%d | RMSE=%.4f | (HF test stations)', name, nn, rmse));

    for k = 1:nLoc
        loc = locs(k);
        rows = (HF_te_tbl.loc_id == loc);

        t = HF_te_tbl.t(rows);
        ytrue = HF_te_tbl.fH(rows);
        ypred = yhat(rows);
        s2 = s2H(rows);

        % sort by time
        [t, ord] = sort(t);
        ytrue = ytrue(ord);
        ypred = ypred(ord);
        s2 = s2(ord);
        sig = sqrt(max(s2,0));

        upper = ypred + 2*sig;
        lower = ypred - 2*sig;

        subplot(nrows, ncols, k);
        hold on;

        % CI band
        fill([t; flipud(t)], [upper; flipud(lower)], [0.85 0.85 0.85], ...
             'EdgeColor','none', 'FaceAlpha',0.5);

        plot(t, ytrue, 'k-', 'LineWidth', 1.3);
        plot(t, ypred, 'r-', 'LineWidth', 1.3);

        grid on;
        title(sprintf('loc %d', loc));
        if k > (nLoc-ncols), xlabel('t'); end
        ylabel('fH');

        hold off;
    end

    % Also: global line plot (all test points sorted by time)
    figure('Name', sprintf('OOS GLOBAL line | %s nn=%d', name, nn));
    tAll = HF_te_tbl.t;
    [tAll, ord] = sort(tAll);
    ytrueAll = HF_te_tbl.fH(ord);
    ypredAll = yhat(ord);
    sigAll   = sqrt(max(s2H(ord), 0));

    upper = ypredAll + 2*sigAll;
    lower = ypredAll - 2*sigAll;

    hold on;
    fill([tAll; flipud(tAll)], [upper; flipud(lower)], [0.9 0.9 0.9], ...
         'EdgeColor','none', 'FaceAlpha',0.5);
    plot(tAll, ytrueAll, 'k-', 'LineWidth', 1.3);
    plot(tAll, ypredAll, 'r-', 'LineWidth', 1.3);
    grid on;
    xlabel('t');
    ylabel('fH');
    title(sprintf('GLOBAL (all test rows) | %s | nn=%d | RMSE=%.4f', name, nn, rmse));
    legend('95% CI','True','Pred','Location','best');
    hold off;
end
