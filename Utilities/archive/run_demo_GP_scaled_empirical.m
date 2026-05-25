function run_demo_GP_scaled_empirical()
    clear; clc; close all;
    global ModelInfo;

    %% -------------------- 1) Simula dati --------------------
    seed = 1;
    train_fraction = 0.6;  % HF_train / HF_test split per stazioni
    out = simulate_data(seed, train_fraction);

    HF_train = out.HF_train;
    HF_test  = out.HF_test;
    LF_tbl   = out.LF;

    % X = [t, lat, lon] coerente con il tuo likelihood
    X_L = [LF_tbl.t,      LF_tbl.s1,      LF_tbl.s2];
    y_L =  LF_tbl.fL;

    X_H = [HF_train.t,    HF_train.s1,    HF_train.s2];
    y_H =  HF_train.fH;

    X_H_test = [HF_test.t, HF_test.s1, HF_test.s2];
    y_H_test =  HF_test.fH;

    fprintf('nL=%d, nH_train=%d, nH_test=%d\n', size(X_L,1), size(X_H,1), size(X_H_test,1));

    %% -------------------- 2) Set ModelInfo --------------------
    ModelInfo = struct();
    ModelInfo.X_L = X_L;
    ModelInfo.X_H = X_H;
    ModelInfo.y_L = y_L;
    ModelInfo.y_H = y_H;

    ModelInfo.jitter = 1e-8;
    ModelInfo.nn_size = 15;

    ModelInfo.kernel = "RBF";
    ModelInfo.conditioning = "Corr";

    ModelInfo.MeanFunction = "zero";      % tienilo semplice per debug
    ModelInfo.RhoFunction  = "GP_scaled_empirical";

    ModelInfo.cand_mult = 10;
    ModelInfo.show_path_diag = false;

    % importante: reset cache vicini (ordering-dependent)
    if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
    if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

    %% -------------------- 3) Hyperparameters --------------------
    % ATTENZIONE: con RhoFunction="GP_scaled_empirical" il ramo usa hyp(end-2:end)
    % quindi qui mettiamo (minimo) 11 + 3 = 14 parametri.
    hyp = zeros(14,1);

    % (1:11) come nel tuo modello base
    hyp(1)  = log(1.0);   hyp(2)  = log(0.20);  % LF: sig_t, ell_t
    hyp(3)  = log(1.0);   hyp(4)  = log(0.20);  % HF: sig_t, ell_t
    hyp(5)  = 0.6;                             % rho (usato solo se constant)
    hyp(6)  = log(0.10); hyp(7)  = log(0.10);   % eps_LF, eps_HF
    hyp(8)  = log(1.0);   hyp(9)  = log(1.0);   % LF: sig_s, ell_s
    hyp(10) = log(1.0);   hyp(11) = log(1.0);   % HF: sig_s, ell_s

    % (12:14) per GP_scaled_empirical: log_sigma, log_ell1, log_ell2
    hyp(12) = log(0.5);   % sigma_rho
    hyp(13) = log(1.0);   % ell1_rho
    hyp(14) = log(1.0);   % ell2_rho

    %% -------------------- 4) Fit: chiama likelihood (costruisce rho_H, H, fattori) --------------------
    NLML = likelihoodVecchia_nonstat_GLS(hyp);
    fprintf('NLML = %.4f\n', NLML);

    % ora ModelInfo contiene:
    % - ModelInfo.debug_vecchia (A, D_inv, perm, R chol, etc.)
    % - ModelInfo.rho_H (valori stimati a HF_train)
    dbg = ModelInfo.debug_vecchia;

    %% -------------------- 5) Predizione su HF_test --------------------
    % Predizione "plug-in" per y_H_test usando:
    % E[y_H* | data] = m_H* + Cov(y_H*, y)^T * K^{-1}(y - m)
    %
    % Qui MeanFunction="zero" quindi m=0.
    % Serve:
    %   a) calcolare K^{-1} y  (già in dbg.SIy)
    %   b) calcolare k_* = Cov(y_H*, y)  (usiamo forma del modello)

    Kinv_y = dbg.SIy;   % = K^{-1} y (raw y), da likelihood

    % Costruisci cov cross tra test HF e training (y_L + y_H_train)
    % NOTA: qui implemento il modello classico:
    %   y_L = w_L + eps_L
    %   y_H = rh_
