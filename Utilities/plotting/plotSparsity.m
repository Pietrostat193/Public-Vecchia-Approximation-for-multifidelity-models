function plotSparsity()
    clear; clc; close all;
    global ModelInfo;

    %% 1) Simulate data
    seed = 1;
    train_fraction = 1.0;
    out = simulate_data(seed, train_fraction);

    LF = out.LF;
    HF = out.HF;

    X_L0 = [LF.t, LF.s1, LF.s2];
    y_L0 = LF.fL;

    X_H0 = [HF.t, HF.s1, HF.s2];
    y_H0 = HF.fH;

    nL = size(X_L0,1);
    nH = size(X_H0,1);
    fprintf('nL=%d, nH=%d\n', nL, nH);

    %% 2) Build ordering structs (THIS WAS MISSING)
    ordL = make_orderings(X_L0);
    ordH = make_orderings(X_H0);

    %% 3) ModelInfo base fields
    ModelInfo = struct();
    ModelInfo.jitter  = 1e-8;
    ModelInfo.nn_size = 15;
    ModelInfo.conditioning = "Corr";
    ModelInfo.kernel       = "RBF";
    ModelInfo.MeanFunction = "zero";
    ModelInfo.RhoFunction  = "constant";
    ModelInfo.cand_mult    = 10;

    % IMPORTANT: disable cached neighbors (ordering-dependent!)
    if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
    if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

    %% 4) Hyperparameters
    hyp = zeros(11,1);
    hyp(1)  = log(1.0);
    hyp(2)  = log(0.20);
    hyp(3)  = log(1.0);
    hyp(4)  = log(0.20);
    hyp(5)  = 0.6;
    hyp(6)  = log(0.10);
    hyp(7)  = log(0.10);
    hyp(8)  = log(1.0);
    hyp(9)  = log(1.0);
    hyp(10) = log(1.0);
    hyp(11) = log(1.0);

    %% 5) 3x3 sparsity plot
    figure('Position',[100 100 1200 1000]);

    cases = {
        'Station-major', ordL.orig,        ordH.orig;
        'Time-major',    ordL.time_major,  ordH.time_major;
        'Random',        ordL.rand,        ordH.rand
    };

    for i = 1:3
        name = cases{i,1};
        pL   = cases{i,2};
        pH   = cases{i,3};

        % reorder data
        ModelInfo.X_L = X_L0(pL,:); ModelInfo.y_L = y_L0(pL);
        ModelInfo.X_H = X_H0(pH,:); ModelInfo.y_H = y_H0(pH);

        % clear ordering-dependent caches again
        if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
        if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

        likelihoodVecchia_nonstat_GLS(hyp);
        H = ModelInfo.H;

        p  = symamd(H);
        Hp = H(p,p);
        R  = chol(Hp);

        % ---- column 1: H ----
        subplot(3,3,3*(i-1)+1)
        spy(H)
        title([name ': H'])
        ylabel('row')

        % ---- column 2: H(p,p) ----
        subplot(3,3,3*(i-1)+2)
        spy(Hp)
        title('H after AMD')

        % ---- column 3: chol(H) ----
        subplot(3,3,3*(i-1)+3)
        spy(R)
        title('chol(H)')
    end

    sgtitle('Effect of ordering on sparsity of H');

end

%% -------- helper: build orderings ----------
function ord = make_orderings(X)
    n = size(X,1);
    ord.orig = (1:n)';

    % time-major: sort by (t, x, y)
    [~, ord.time_major] = sortrows(X, [1 2 3]);

    % space-major: sort by (x, y, t)
    [~, ord.space_major] = sortrows(X, [2 3 1]);

    % random
    ord.rand = randperm(n)';
end
