function T = demo_compare_dense_vs_vecchia_v4()
% DEMO_COMPARE_DENSE_VS_VECCHIA_V4
% Compare "exact" dense likelihood (likelihood2Dsp_GLSmean) vs
% Vecchia REML likelihood (likelihoodVecchia_nonstat_GLS_v4).
%
% Key points:
% 1) We use likelihood2Dsp_GLSmean(hyp) as the dense reference (Exact).
% 2) We use likelihoodVecchia_nonstat_GLS_v4(hyp) as the Vecchia approx.
% 3) We FORCE the same GLS design matrix structure in BOTH by using
%    ModelInfo.GLSType consistently, and we also build ModelInfo.Phi
%    to match Vecchia's G_gls *before calling the dense likelihood*,
%    in case your likelihood2Dsp_GLSmean expects ModelInfo.Phi.
%
% Metrics:
%   - NLML_Exact, NLML_Vecchia, DiffAbs, DiffRel
%   - clipFrac_LF / clipFrac_HF (degeneracy indicator)
%   - nnzR / condR_est (cost + numeric proxy from Vecchia's R factor)
%
% Requires on path:
%   simulate_data_dynamic (or replace with your own data loader)
%   likelihood2Dsp_GLSmean
%   likelihoodVecchia_nonstat_GLS_v4
%   vecchia_approx_space_time_corr_fast1 (only for neighbor indices precompute)
%   k_space_time_v2 (for clip fraction)
%   k_space_time / k_space_time_v2 kernels

    global ModelInfo
    rng(123);
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

    seed = 1;
    train_fraction = 0.5;

    out = simulate_data_dynamic(seed, train_fraction, cfg);

    X_L0 = [out.LF.t, out.LF.s1, out.LF.s2];
    y_L0 = out.LF.fL(:);

    X_H0 = [out.HF.t, out.HF.s1, out.HF.s2];
    y_H0 = out.HF.fH(:);

    nL = size(X_L0,1);
    nH = size(X_H0,1);
    N  = nL + nH;

    fprintf('Dataset: nL=%d, nH=%d, N=%d\n', nL, nH, N);

    %% ---------------- ModelInfo base ----------------
    ModelInfo = struct();
    ModelInfo.jitter  = 1e-8;
    ModelInfo.conditioning = "Corr";
    ModelInfo.kernel       = kernelName;
    ModelInfo.MeanFunction = "zero";
    ModelInfo.RhoFunction  = "constant";

    % IMPORTANT: use the same mean design choice that Vecchia uses
    % Options:
    %   ModelInfo.GLSType = "constant";  % two intercepts (LF/HF)
    %   ModelInfo.GLSType = "adaptive";  % blkdiag([1,lat,lon]_L, [1,lat,lon]_H)
    ModelInfo.GLSType      = "constant";

    ModelInfo.cov_type     = "RBF";
    ModelInfo.combination  = "multiplicative";
    ModelInfo.cand_mult    = 10;
    ModelInfo.show_path_diag = false;

    % If your dense likelihood supports ML/REML switch:
    % - v4 Vecchia is REML by construction (because we added the extra term),
    %   so set this to true to match objectives if your dense likelihood uses it.
    ModelInfo.use_reml = true;

    %% ---------------- hyp (fixed for comparison) ----------------
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
    ordL = make_orderings_full(X_L0, 111);
    ordH = make_orderings_full(X_H0, 222);

    combos = {
        {'Time-major / Time-major',                      ordL.time_major,           ordH.time_major}
        {'Time-causal+RandSpace / Time-causal+RandSpace',ordL.time_causal_randS,    ordH.time_causal_randS}
        {'Space-major / Space-major',                    ordL.space_major,          ordH.space_major}
        {'Random / Random',                              ordL.rand,                 ordH.rand}
    };

    nn_list = [10 15 20 30 40];

    %% ---------------- results table ----------------
    T = table('Size',[0 12], ...
        'VariableTypes',{'string','double','double','double','double','double', ...
                         'double','double','double','double','double','double'}, ...
        'VariableNames',{'Ordering','nn','NLML_Exact','NLML_Vecchia','DiffAbs','DiffRel', ...
                         'clipFrac_LF','clipFrac_HF','nnzR','condR_est','betaNorm','betaDiffNorm'});

    %% ================= main loop =================
    for k = 1:numel(combos)
        nm = string(combos{k}{1});
        pL = combos{k}{2};
        pH = combos{k}{3};

        X_L = X_L0(pL,:); y_L = y_L0(pL);
        X_H = X_H0(pH,:); y_H = y_H0(pH);

        % Set current data
        ModelInfo.X_L = X_L; ModelInfo.y_L = y_L;
        ModelInfo.X_H = X_H; ModelInfo.y_H = y_H;

        % --- Build Phi (design) to match Vecchia's G_gls (important!)
        ModelInfo.Phi = build_gls_design_like_vecchia(X_L, X_H);

        % Dense reference
        NLML_E = likelihood2Dsp_GLSmean(hyp);
        betaE  = getfield_if_exists(ModelInfo,'beta_hat');
        if isempty(betaE), betaE = getfield_if_exists(ModelInfo,'beta_gls'); end

        fprintf('\n--- %s --- Dense(Exact) NLML=%.6f\n', nm, NLML_E);

        for nn = nn_list
            ModelInfo.nn_size = nn;

            % neighbor indices for LIGHT (ordering-dependent)
            [idxL, idxH] = precompute_vecchia_indices_for_ordering(X_L, X_H, nn, kernelName);
            ModelInfo.idxL_precomputed = idxL;
            ModelInfo.idxH_precomputed = idxH;

            % Vecchia v4
            NLML_V = likelihoodVecchia_nonstat_GLS_v4(hyp);
            betaV  = getfield_if_exists(ModelInfo,'beta_gls');

            diffAbs = abs(NLML_V - NLML_E);
            diffRel = diffAbs / max(abs(NLML_E), 1e-12);

            % clip fractions (numerical degeneracy)
            clipL = condvar_clip_fraction(X_L, idxL, hyp, "LF", kernelName);
            clipH = condvar_clip_fraction(X_H, idxH, hyp, "HF", kernelName);

            % sparsity/cost stats from Vecchia's R
            Rchol = getfield_if_exists(ModelInfo,'R');
            [nnzR, condRest] = chol_stats(Rchol);

            % Compare beta norms if both exist
            betaNorm = NaN; betaDiffNorm = NaN;
            if ~isempty(betaV)
                betaNorm = norm(betaV);
            end
            if ~isempty(betaE) && ~isempty(betaV) && numel(betaE)==numel(betaV)
                betaDiffNorm = norm(betaE - betaV);
            end

            T = [T; {nm, nn, NLML_E, NLML_V, diffAbs, diffRel, ...
                     clipL, clipH, nnzR, condRest, betaNorm, betaDiffNorm}]; %#ok<AGROW>

            fprintf('nn=%-3d | Vecchia=%.2f | diff=%.2f (%.3g rel) | nnzR=%g | clip(L/H)=%.2f/%.2f\n', ...
                nn, NLML_V, diffAbs, diffRel, nnzR, clipL, clipH);
        end
    end

    fprintf('\n==================== SORT BY DiffAbs (accuracy) ====================\n');
    disp(sortrows(T,'DiffAbs'));

    fprintf('\n==================== SORT BY nnzR (cost proxy) ====================\n');
    disp(sortrows(T,'nnzR'));
end

%% ========================================================================
%% orderings (FULL)
%% ========================================================================
function ord = make_orderings_full(X, seed)
    % X = [t, s1, s2]
    if nargin < 2, seed = 1; end
    n = size(X,1);
    ord.orig = (1:n)';

    [~, ord.time_major]  = sortrows(X, [1 2 3]); % (t,s1,s2)
    [~, ord.space_major] = sortrows(X, [2 3 1]); % (s1,s2,t)

    rng(seed);
    ord.rand = randperm(n)';

    rng(seed);
    tvals = X(:,1);
    [tuniq, ~] = unique(tvals, 'stable');

    ord_tc = zeros(n,1);
    pos = 1;
    for it = 1:numel(tuniq)
        idx_t = find(tvals == tuniq(it));
        idx_t = idx_t(randperm(numel(idx_t))); % random space within each time
        ord_tc(pos:pos+numel(idx_t)-1) = idx_t(:);
        pos = pos + numel(idx_t);
    end
    ord.time_causal_randS = ord_tc;
end

%% ========================================================================
%% Build GLS design matrix like Vecchia (G_gls)
%% ========================================================================
function Phi = build_gls_design_like_vecchia(X_L, X_H)
    global ModelInfo
    nL = size(X_L,1);
    nH = size(X_H,1);

    if isfield(ModelInfo,'GLSType') && strcmp(ModelInfo.GLSType, "adaptive")
        G_L = [ones(nL,1), X_L(:,2), X_L(:,3)];
        G_H = [ones(nH,1), X_H(:,2), X_H(:,3)];
        Phi = blkdiag(G_L, G_H);
    else
        Phi = [[ones(nL,1); zeros(nH,1)], ...
               [zeros(nL,1); ones(nH,1)]];
    end
end

%% ========================================================================
%% helpers
%% ========================================================================
function v = getfield_if_exists(S, f)
    if isfield(S,f), v = S.(f); else, v = []; end
end

function [nnzR, condRest] = chol_stats(R)
    if isempty(R)
        nnzR = NaN; condRest = NaN; return;
    end
    nnzR = nnz(R);
    d = abs(diag(R));
    condRest = max(d) / max(min(d), 1e-30);
end

%% ========================================================================
%% clip fraction
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

    var_self = max(k_space_time_v2(X, [], [sig_s, ell_s], [sig_t, ell_t], kernelName), eps_val);
    clipped = false(n,1);

    for i=1:n
        nbrs = idxMat(i,:); nbrs = nbrs(nbrs>0);
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
%% neighbor precompute (ordering-dependent)
%% ========================================================================
function [idxL, idxH] = precompute_vecchia_indices_for_ordering(X_L, X_H, nn, kernelName)
    hyp_dummy_s = [1, 1]; hyp_dummy_t = [1, 1];
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
