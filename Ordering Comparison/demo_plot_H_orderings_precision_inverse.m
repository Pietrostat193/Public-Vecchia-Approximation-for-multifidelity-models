function T = demo_plot_H_orderings_precision_inverse(seed)
% DEMO_PLOT_H_ORDERINGS_PRECISION_INVERSE
% Fully self-contained Monte Carlo version (single-file).
%
% External dependencies REQUIRED on path:
%   simulate_data_dynamic
%   likelihood2Dsp_GLSmean
%   likelihoodVecchia_nonstat_GLS_v4
%   k_space_time
%   k_space_time_v2
%   vecchia_approx_space_time_corr_fast1

    if nargin < 1
        error('Seed must be provided.');
    end

    global ModelInfo
    kernelName = "RBF";

    %% ---------------- DATA ----------------
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

    train_fraction = 0.5;

    out = simulate_data_dynamic(seed, train_fraction, cfg);

    X_L0 = [out.LF.t, out.LF.s1, out.LF.s2];
    y_L0 = out.LF.fL(:);

    X_H0 = [out.HF.t, out.HF.s1, out.HF.s2];
    y_H0 = out.HF.fH(:);

    nL = size(X_L0,1);
    nH = size(X_H0,1);
    N  = nL + nH;

    G = ones(N,1);

    fprintf('Dataset: nL=%d, nH=%d, N=%d\n', nL, nH, N);

    %% ---------------- ModelInfo ----------------
    ModelInfo = struct();
    ModelInfo.jitter  = 1e-8;
    ModelInfo.conditioning = "Corr";
    ModelInfo.kernel       = kernelName;
    ModelInfo.MeanFunction = "zero";
    ModelInfo.RhoFunction  = "constant";
    ModelInfo.GLSType      = "constant";
    ModelInfo.cov_type     = "RBF";
    ModelInfo.combination  = "multiplicative";
    ModelInfo.cand_mult    = 10;
    ModelInfo.show_path_diag = false;

    %% ---------------- hyperparameters ----------------
    hyp = zeros(11,1);
    hyp(1)  = log(1.0);
    hyp(2)  = log(0.20);
    hyp(3)  = log(1.0);
    hyp(4)  = log(0.20);
    hyp(5)  = 0.6;
    hyp(6)  = log(0.10);
    hyp(7)  = log(0.10);
    hyp(8)  = log(1.0);
    hyp(9)  = log(1.00);
    hyp(10) = log(1.0);
    hyp(11) = log(1.00);

    %% ---------------- orderings ----------------
    ordL = make_orderings_full(X_L0, seed + 111);
    ordH = make_orderings_full(X_H0, seed + 222);

    combos = {
        {'Time-major / Time-major',                 ordL.time_major,          ordH.time_major}
        {'Time-causal+RandSpace / Time-causal+RandSpace', ordL.time_causal_randS,  ordH.time_causal_randS}
        {'Space-major / Space-major',               ordL.space_major,         ordH.space_major}
        {'Random / Random',                         ordL.rand,                ordH.rand}
    };

    nn_list = [10 15 20 30 40];

    %% ---------------- results table ----------------
    T = table('Size',[0 14], ...
        'VariableTypes',{'string','double','double','double','double','double', ...
                         'double','double','double','double','double','double','double','double'}, ...
        'VariableNames',{'Ordering','nn','NLML_Exact','NLML_Vecchia','DiffAbs','DiffRel', ...
                         'clipFrac_LF','clipFrac_HF','nnzH','densH','nnzR','fillR_ratio','condR_est','exactInvarAbs'});

    %% ================= main loop =================
    for k = 1:numel(combos)

        nm = string(combos{k}{1});
        pL = combos{k}{2};
        pH = combos{k}{3};

        X_L = X_L0(pL,:); y_L = y_L0(pL);
        X_H = X_H0(pH,:); y_H = y_H0(pH);

        ModelInfo.X_L = X_L; ModelInfo.y_L = y_L;
        ModelInfo.X_H = X_H; ModelInfo.y_H = y_H;

        NLML_E = likelihood2Dsp_GLSmean(hyp);

        % exact invariance sanity check (ordering shouldn't matter for exact)
        exactInvarAbs = exact_invariance_check(X_L, X_H, y_L, y_H, hyp, kernelName, G);

        for nn = nn_list

            ModelInfo.nn_size = nn;

            % ordering-dependent neighbor indices
            [idxL, idxH] = precompute_vecchia_indices_for_ordering(X_L, X_H, nn, kernelName);
            ModelInfo.idxL_precomputed = idxL;
            ModelInfo.idxH_precomputed = idxH;

            NLML_V = likelihoodVecchia_nonstat_GLS_v4(hyp);

            diffAbs = abs(NLML_V - NLML_E);
            diffRel = diffAbs / max(abs(NLML_E), 1e-12);

            % numerical degeneracy check (cond-var clipping)
            clipL = condvar_clip_fraction(X_L, idxL, hyp, "LF", kernelName);
            clipH = condvar_clip_fraction(X_H, idxH, hyp, "HF", kernelName);

            % sparsity/cost stats
            H = getfield_if_exists(ModelInfo,'H');
            Rchol = getfield_if_exists(ModelInfo,'R');
            [nnzH, densH, nnzR, fillRatio, condRest] = sparsity_stats(H, Rchol);

            T = [T; {nm, nn, NLML_E, NLML_V, diffAbs, diffRel, ...
                     clipL, clipH, nnzH, densH, nnzR, fillRatio, condRest, exactInvarAbs}]; %#ok<AGROW>
        end
    end
end

%% ========================================================================
%% ORDERINGS
%% ========================================================================
function ord = make_orderings_full(X, seed)
    % X = [t, s1, s2]
    n = size(X,1);
    ord.orig = (1:n)';

    % deterministic time-major: sort by (t, s1, s2)
    [~, ord.time_major]  = sortrows(X, [1 2 3]);

    % space-major: sort by (s1, s2, t)
    [~, ord.space_major] = sortrows(X, [2 3 1]);

    % fully random
    rng(seed);
    ord.rand = randperm(n)';

    % time-causal + random space within each time slice
    rng(seed);
    tvals = X(:,1);
    tuniq = unique(tvals,'stable');

    ord_tc = zeros(n,1);
    pos = 1;
    for it = 1:numel(tuniq)
        idx_t = find(tvals == tuniq(it));
        idx_t = idx_t(randperm(numel(idx_t)));
        ord_tc(pos:pos+numel(idx_t)-1) = idx_t(:);
        pos = pos + numel(idx_t);
    end
    ord.time_causal_randS = ord_tc;
end

%% ========================================================================
%% FIELD ACCESS + SPARSITY STATS
%% ========================================================================
function v = getfield_if_exists(S, f)
    if isfield(S,f), v = S.(f); else, v = []; end
end

function [nnzH, densH, nnzR, fillRatio, condRest] = sparsity_stats(H, R)
    if isempty(H)
        nnzH = NaN; densH = NaN;
    else
        nnzH  = nnz(H);
        densH = nnzH / numel(H);
    end

    if isempty(R)
        nnzR = NaN; fillRatio = NaN; condRest = NaN;
        return;
    end

    nnzR = nnz(R);
    fillRatio = nnzR / max(nnzH, 1);

    d = abs(diag(R));
    condRest = max(d) / max(min(d), 1e-30);
end

%% ========================================================================
%% EXACT INVARIANCE CHECK (uses exact reference NLML)
%% ========================================================================
function dAbs = exact_invariance_check(X_L, X_H, y_L, y_H, hyp, kernelName, G)
    nL = size(X_L,1); nH = size(X_H,1);

    NLML0 = exact_NLML_reference(X_L, X_H, y_L, y_H, hyp, kernelName, G);

    P = interleave_LF_HF(nL, nH);

    X_joint = [X_L; X_H];
    y_joint = [y_L; y_H];

    Xp = X_joint(P,:);
    yp = y_joint(P);

    XLp = Xp(1:nL,:); yLp = yp(1:nL);
    XHp = Xp(nL+1:end,:); yHp = yp(nL+1:end);

    NLML1 = exact_NLML_reference(XLp, XHp, yLp, yHp, hyp, kernelName, G);

    dAbs = abs(NLML1 - NLML0);
end

function P = interleave_LF_HF(nL, nH)
    n = nL + nH;
    P = zeros(n,1);
    m = min(nL,nH);
    idx = 1;

    for i = 1:m
        P(idx) = i;      idx = idx + 1;
        P(idx) = nL + i; idx = idx + 1;
    end
    if nL > m
        P(idx:end) = (m+1:nL)';
    elseif nH > m
        P(idx:end) = (nL + (m+1:nH))';
    end
end

function NLML = exact_NLML_reference(X_L, X_H, y_L, y_H, hyp, kernelName, G)
    y = [y_L; y_H];
    nL = size(X_L,1); nH = size(X_H,1); N = nL+nH;

    s2_tL = exp(hyp(1));  ell2_tL = exp(hyp(2));
    s2_tH = exp(hyp(3));  ell2_tH = exp(hyp(4));
    rho   = hyp(5);
    epsL  = exp(hyp(6));
    epsH  = exp(hyp(7));
    s2_sL = exp(hyp(8));  ell2_sL = exp(hyp(9))^2;
    s2_sH = exp(hyp(10)); ell2_sH = exp(hyp(11))^2;

    K_L    = k_space_time(X_L, X_L, [s2_sL, ell2_sL, ell2_sL], [s2_tL, ell2_tL], kernelName);
    K_D    = k_space_time(X_H, X_H, [s2_sH, ell2_sH, ell2_sH], [s2_tH, ell2_tH], kernelName);
    K_LH   = k_space_time(X_L, X_H, [s2_sL, ell2_sL, ell2_sL], [s2_tL, ell2_tL], kernelName);
    K_HH_L = k_space_time(X_H, X_H, [s2_sL, ell2_sL, ell2_sL], [s2_tL, ell2_tL], kernelName);

    Sigma = zeros(N,N);
    Sigma(1:nL,1:nL) = K_L + eye(nL)*epsL;
    Sigma(1:nL,nL+1:end) = rho*K_LH;
    Sigma(nL+1:end,1:nL) = rho*K_LH';
    Sigma(nL+1:end,nL+1:end) = rho^2*K_HH_L + K_D + eye(nH)*epsH;

    Sigma = 0.5*(Sigma+Sigma') + eye(N)*1e-8;
    L = chol(Sigma,'lower');

    invL_G = L \ G;
    invL_y = L \ y;
    beta = (invL_G'*invL_G) \ (invL_G'*invL_y);

    res = y - G*beta;
    invL_res = L \ res;

    NLML = 0.5*(invL_res'*invL_res + 2*sum(log(diag(L))) + N*log(2*pi));
end

%% ========================================================================
%% COND-VAR CLIP FRACTION (numerical degeneracy proxy)
%% ========================================================================
function fracClip = condvar_clip_fraction(X, idxMat, hyp, which, kernelName)
    switch which
        case "LF"
            sig_t = exp(hyp(1)); ell_t = exp(hyp(2));
            sig_s = exp(hyp(8)); ell_s = exp(hyp(9));
        case "HF"
            sig_t = exp(hyp(3)); ell_t = exp(hyp(4));
            sig_s = exp(hyp(10)); ell_s = exp(hyp(11));
        otherwise
            error('which must be "LF" or "HF"');
    end

    eps_val = 1e-7;
    n = size(X,1);

    % diag variance vector (your k_space_time_v2 supports X,[] to return diag)
    var_self = max(k_space_time_v2(X, [], [sig_s, ell_s], [sig_t, ell_t], kernelName), eps_val);

    clipped = false(n,1);
    for i = 1:n
        nbrs = idxMat(i,:); 
        nbrs = nbrs(nbrs>0);
        if isempty(nbrs), continue; end

        xi = X(i,:);
        Xnbr = X(nbrs,:);

        K_nn = k_space_time_v2(Xnbr, Xnbr, [sig_s, ell_s], [sig_t, ell_t], kernelName);
        K_nn = 0.5*(K_nn+K_nn') + eps_val*eye(numel(nbrs));

        K_i_n = k_space_time_v2(Xnbr, xi, [sig_s, ell_s], [sig_t, ell_t], kernelName);

        Ai = K_nn \ K_i_n(:);
        cv_raw = var_self(i) - K_i_n(:)' * Ai;

        if cv_raw < eps_val
            clipped(i) = true;
        end
    end

    fracClip = mean(clipped);
end

%% ========================================================================
%% NEIGHBOR PRECOMPUTE FOR VECCHIA (ordering-dependent)
%% ========================================================================
function [idxL, idxH] = precompute_vecchia_indices_for_ordering(X_L, X_H, nn, kernelName)
    hyp_dummy_s = [1, 1];
    hyp_dummy_t = [1, 1];

    % This function is external; we assume it returns struct with field B (as in your code)
    resL = vecchia_approx_space_time_corr_fast1(X_L, hyp_dummy_s, hyp_dummy_t, nn, 1e-6, kernelName, 10, 1, 1, []);
    resH = vecchia_approx_space_time_corr_fast1(X_H, hyp_dummy_s, hyp_dummy_t, nn, 1e-6, kernelName, 10, 1, 1, []);

    idxL = extract_indices(resL.B, nn);
    idxH = extract_indices(resH.B, nn);
end

function idx_mat = extract_indices(B, nn)
    n = size(B,1);
    idx_mat = zeros(n, nn);
    for i = 2:n
        cols = find(B(i, 1:i-1));
        if ~isempty(cols)
            len = min(numel(cols), nn);
            idx_mat(i, 1:len) = cols(end-len+1:end);
        end
    end
end