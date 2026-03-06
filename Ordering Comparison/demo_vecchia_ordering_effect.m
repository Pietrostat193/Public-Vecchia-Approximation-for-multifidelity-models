function demo_plot_H_orderings()
    global ModelInfo;

    %% 1) Simulate data
    seed = 1;
    train_fraction = 1.0;  % <-- set to 1 so LF/HF grids align perfectly (easier interleaving)
    out = simulate_data(seed, train_fraction);

    % Use FULL LF and FULL HF on identical coordinates
    LF = out.LF;
    HF = out.HF;

    X_L0 = [LF.t, LF.s1, LF.s2];
    y_L0 = LF.fL;

    X_H0 = [HF.t, HF.s1, HF.s2];
    y_H0 = HF.fH;

    nL = size(X_L0,1);
    nH = size(X_H0,1);
    fprintf('nL=%d, nH=%d\n', nL, nH);

    %% 2) Base ModelInfo (fields your likelihood expects)
    ModelInfo = struct();
    ModelInfo.jitter  = 1e-8;
    ModelInfo.nn_size = 15;
    ModelInfo.conditioning = "Corr";  % you said Corr-conditioning
    ModelInfo.kernel       = "RBF";
    ModelInfo.MeanFunction = "zero";
    ModelInfo.RhoFunction  = "constant";

    % IMPORTANT: If you cache fixed neighbors, they are ORDERING-DEPENDENT.
    % For this demo we turn off caches so each ordering is "honest".
    if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
    if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

    ModelInfo.cand_mult = 10;
    ModelInfo.show_path_diag = false;

    %% 3) Hyperparameters hyp (Mean=zero, Rho=constant => length 11 in your function)
    hyp = zeros(11,1);
    hyp(1)  = log(1.0);   % s_sig_LF_t
    hyp(2)  = log(0.20);  % t_ell_LF
    hyp(3)  = log(1.0);   % s_sig_HF_t
    hyp(4)  = log(0.20);  % t_ell_HF
    hyp(5)  = 0.6;        % rho
    hyp(6)  = log(0.10);  % eps_LF
    hyp(7)  = log(0.10);  % eps_HF
    hyp(8)  = log(1.0);   % s_sig_LF_s
    hyp(9)  = log(1.00);  % s_ell_LF
    hyp(10) = log(1.0);   % s_sig_HF_s
    hyp(11) = log(1.00);  % s_ell_HF

    %% 4) Define orderings (WITHIN each fidelity block)
    % These change Vecchia predecessor sets and thus change H itself.
    ordL = make_orderings(X_L0);
    ordH = make_orderings(X_H0);

    %% 5) Loop over a few combinations
    combos = {
        % name,   permL,                  permH
        {'Orig / Orig',               ordL.orig,          ordH.orig}
        {'Time-major / Time-major',   ordL.time_major,    ordH.time_major}
        {'Space-major / Space-major', ordL.space_major,   ordH.space_major}
        {'Random / Random',           ordL.rand,          ordH.rand}
    };

    results = struct([]);

    for k = 1:numel(combos)
        nm   = combos{k}{1};
        pL   = combos{k}{2};
        pH   = combos{k}{3};

        % Apply within-block orderings
        X_L = X_L0(pL,:); y_L = y_L0(pL);
        X_H = X_H0(pH,:); y_H = y_H0(pH);

        % Put into ModelInfo
        ModelInfo.X_L = X_L; ModelInfo.y_L = y_L;
        ModelInfo.X_H = X_H; ModelInfo.y_H = y_H;

        % IMPORTANT: clear neighbor caches again so you don't accidentally
        % reuse idxFixed computed for a different ordering.
        if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
        if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

        % Run once: builds ModelInfo.H inside likelihood
        NLML = likelihoodVecchia_nonstat_GLS(hyp); %#ok<NASGU>
        H = ModelInfo.H;

        % Stats
        nnzH  = nnz(H);
        densH = nnzH / numel(H);

        % Factorization fill proxy (after AMD)
        p = symamd(H);
        Hp = H(p,p);
        R  = chol(Hp);
        nnzR = nnz(R);

        fprintf('\n=== %s ===\n', nm);
        fprintf('H: %dx%d | nnz(H)=%d | dens=%.3g | nnz(chol(H(p,p)))=%d\n', ...
            size(H,1), size(H,2), nnzH, densH, nnzR);

        % Plots: H in native ordering
        figure('Name',['H spy: ' nm]);
        spy(H);
        title(sprintf('spy(H) | %s | nnz=%d | dens=%.3g', nm, nnzH, densH));

        % Plots: H after symamd (what matters for sparse Cholesky)
        figure('Name',['H spy after symamd: ' nm]);
        spy(Hp);
        title(sprintf('spy(H(p,p)) symamd | %s | nnz=%d', nm, nnz(Hp)));

        results(k).name = nm;
        results(k).nnzH = nnzH;
        results(k).dens = densH;
        results(k).nnzR = nnzR;
    end

    %% 6) (Optional) Demonstrate *joint* LF/HF reordering as a pure permutation of H
    % This does NOT change Vecchia neighbors (since Vecchia was built separately),
    % but it shows the referee's point that "bandedness" depends on how you order
    % the joint vector y even if the underlying graph is the same.
    %
    % Take the last H and permute to interleave LF/HF indices:
    Hlast = ModelInfo.H;
    P = interleave_LF_HF(size(X_L0,1), size(X_H0,1)); % assumes equal is ideal
    Hinter = Hlast(P,P);

    figure('Name','H after interleaving LF/HF (pure permutation)');
    spy(Hinter);
    title('spy(PHP'') where P interleaves LF/HF (pure permutation)');

end

%% ---------- helpers ----------

function ord = make_orderings(X)
    n = size(X,1);
    ord.orig = (1:n)';

    % time-major then space: sort by (t, x, y)
    [~, ord.time_major] = sortrows(X, [1 2 3]);

    % space-major then time: sort by (x, y, t)
    [~, ord.space_major] = sortrows(X, [2 3 1]);

    % random
    ord.rand = randperm(n)';
end

function P = interleave_LF_HF(nL, nH)
    % Build a permutation of 1:(nL+nH) that interleaves LF(1),HF(1),LF(2),HF(2),...
    % If nL ~= nH, the remainder is appended at the end.
    n = nL + nH;
    P = zeros(n,1);

    m = min(nL,nH);
    idx = 1;

    for i = 1:m
        P(idx)   = i;         idx = idx + 1;       % LF i  ->  i
        P(idx)   = nL + i;    idx = idx + 1;       % HF i  ->  nL+i
    end

    if nL > m
        P(idx:end) = (m+1:nL)';
    elseif nH > m
        P(idx:end) = (nL + (m+1:nH))';
    end
end
