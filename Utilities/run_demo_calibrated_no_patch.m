function compare_rho_fixed_vs_moving_varrho()
    clear; clc; close all;
    global ModelInfo;

    %% ---------------- 1) Simula dati con rho(s) variabile ----------------
    seed = 1;
    train_fraction = 0.6;
    out = simulate_data_varrho(seed, train_fraction);

    HF_train = out.HF_train;
    HF_test  = out.HF_test;
    LF_tbl   = out.LF;

    X_L = [LF_tbl.t, LF_tbl.s1, LF_tbl.s2];
    y_L =  LF_tbl.fL;

    X_H = [HF_train.t, HF_train.s1, HF_train.s2];
    y_H =  HF_train.fH;

    X_H_test = [HF_test.t, HF_test.s1, HF_test.s2];
    y_H_test =  HF_test.fH;

    fprintf('nL=%d, nH_train=%d, nH_test=%d\n', size(X_L,1), size(X_H,1), size(X_H_test,1));

    %% ---------------- 2) Setup comune ModelInfo ----------------
    base = struct();
    base.X_L = X_L; base.X_H = X_H;
    base.y_L = y_L; base.y_H = y_H;

    base.jitter = 1e-8;
    base.nn_size = 15;
    base.kernel = "RBF";
    base.conditioning = "Corr";
    base.cand_mult = 10;
    base.show_path_diag = false;

    base.MeanFunction = "zero";   % confronto pulito

    % se il predictor li richiede
    base.cov_type = "RBF";
    base.combination = "multiplicative";

    %% ---------------- 3) Definisci i due casi ----------------
    cases = { ...
        struct('name',"rho constant",        'RhoFunction',"constant"), ...
        struct('name',"rho GP_scaled_emp",   'RhoFunction',"GP_scaled_empirical") ...
    };

    results = struct([]);

    %% ---------------- 4) Loop: fit + predict per ogni caso ----------------
    for c = 1:numel(cases)
        ModelInfo = base;
        ModelInfo.RhoFunction = cases{c}.RhoFunction;

        % reset cache vicini (importante)
        if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
        if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

        % hyp iniziale coerente col caso
        hyp0 = default_hyp_init(ModelInfo.RhoFunction);

        % ottimizza hyp (veloce)
        obj = @(h) likelihoodVecchia_nonstat_GLS(h);

        if license('test','optimization_toolbox')
            opts = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton', ...
                'MaxIterations', 40, 'MaxFunctionEvaluations', 150);
            [hyp_hat, nlml] = fminunc(obj, hyp0, opts);
        else
            opts = optimset('Display','iter','MaxIter',40,'MaxFunEvals',150);
            [hyp_hat, nlml] = fminsearch(obj, hyp0, opts);
        end

        % salva hyp e richiama likelihood per popolare debug_vecchia / gprModel_rho
        ModelInfo.hyp = hyp_hat;
        nlml = likelihoodVecchia_nonstat_GLS(hyp_hat);

        % predizione calibrata
        optsPred = struct();
        optsPred.calib_mode   = 'global_affine';
        optsPred.gamma_clip   = [0.25, 4.0];
        optsPred.lambda_ridge = 1e-8;
        optsPred.gamma_subset = [];
        optsPred.seed         = 42;

        mu_pred = predictVecchia_CM_calibrated2(X_H_test, optsPred);

        rmse = sqrt(mean((mu_pred - y_H_test).^2));

        results(c).name = cases{c}.name;
        results(c).RhoFunction = cases{c}.RhoFunction;
        results(c).NLML = nlml;
        results(c).RMSE = rmse;
        results(c).mu_pred = mu_pred;
        results(c).hyp_hat = hyp_hat;

        fprintf('\n[%s] NLML=%.4f | RMSE=%.4f\n', results(c).name, results(c).NLML, results(c).RMSE);
    end

    %% ---------------- 5) Tabella confronto ----------------
    fprintf('\n=== Confronto (rho vera varia nello spazio) ===\n');
    fprintf('%-18s  %-22s  %-12s  %-12s\n', 'Case', 'RhoFunction', 'NLML', 'RMSE');
    for c = 1:numel(results)
        fprintf('%-18s  %-22s  %-12.4f  %-12.4f\n', ...
            results(c).name, string(results(c).RhoFunction), results(c).NLML, results(c).RMSE);
    end

    %% ---------------- 6) Plot predizioni ----------------
    figure;
    plot(y_H_test,'k.-','DisplayName','HF test vero'); hold on;
    plot(results(1).mu_pred,'o-','DisplayName',results(1).name);
    plot(results(2).mu_pred,'x-','DisplayName',results(2).name);
    legend('Location','best');
    title('Confronto HF prediction: rho fisso vs rho(s) adattivo');
    grid on;

    figure;
    subplot(1,2,1);
    scatter(y_H_test, results(1).mu_pred, 20, 'filled'); grid on;
    xlabel('vero'); ylabel('predetto'); title(results(1).name);

    subplot(1,2,2);
    scatter(y_H_test, results(2).mu_pred, 20, 'filled'); grid on;
    xlabel('vero'); ylabel('predetto'); title(results(2).name);

    %% ---------------- 7) Plot rho(s) vera (se disponibile) ----------------
    sc = out.station_coords;   % sempre ha loc_id,s1,s2 (dalla tua stampa)
    rho_true_station = [];

    % (A) preferisci la colonna rho_loc se esiste
    if istable(sc) && any(strcmp(sc.Properties.VariableNames,'rho_loc'))
        rho_true_station = sc.rho_loc;
    end

    % (B) altrimenti usa out.rho_loc se esiste
    if isempty(rho_true_station) && isfield(out,'rho_loc') && ~isempty(out.rho_loc)
        rho_true_station = out.rho_loc(:);
    end

    if ~isempty(rho_true_station)
        figure;
        scatter(sc.s1, sc.s2, 140, rho_true_station, 'filled'); colorbar;
        title('True \rho(s) used in simulation (per station)');
        xlabel('s1'); ylabel('s2'); grid on;
    else
        warning('rho true non disponibile: aggiungi rho_loc a simulate_data_varrho output (station_coords o out.rho_loc).');
    end

    %% ---------------- 8) Diagnostica: rho_true vs rho_local stimata (se disponibile) ----------------
    % Questa diagnostica ha senso solo per GP_scaled_empirical (che salva rho_local e X_H_unique)
    if isfield(ModelInfo,'rho_local') && isfield(ModelInfo,'X_H_unique') ...
            && ~isempty(ModelInfo.rho_local) && ~isempty(ModelInfo.X_H_unique) ...
            && ~isempty(rho_true_station)

        rho_local = ModelInfo.rho_local(:);      % n_unique x 1
        Xuniq     = ModelInfo.X_H_unique;        % n_unique x 2 = (s1,s2)

        [tf, loc] = ismember(Xuniq, [sc.s1 sc.s2], 'rows');
        rho_true = nan(size(rho_local));
        rho_true(tf) = rho_true_station(loc(tf));

        figure;
        scatter(rho_true, rho_local, 60, 'filled'); grid on;
        xlabel('\rho true (simulation)'); ylabel('\rho local estimated');
        title('Check: \rho true vs \rho\_local (empirical)');

        ok = isfinite(rho_true) & isfinite(rho_local);
        if any(ok)
            r = corr(rho_true(ok), rho_local(ok));
            fprintf('Corr(rho_true, rho_local) = %.3f (n=%d)\n', r, sum(ok));
        end
    end
end

%% ================= helper: hyp init coerente =================
function hyp0 = default_hyp_init(RhoFunction)
    RF = string(RhoFunction);

    switch RF
        case "constant"
            hyp0 = zeros(11,1);
        otherwise
            % GP_scaled_empirical usa hyp(end-2:end) => almeno 14
            hyp0 = zeros(14,1);
    end

    % base 11
    hyp0(1)  = log(1.0);  hyp0(2)  = log(0.20);  % LF time
    hyp0(3)  = log(1.0);  hyp0(4)  = log(0.20);  % HF time
    hyp0(5)  = 0.6;                              % rho (constant)
    hyp0(6)  = log(0.10); hyp0(7)  = log(0.10);  % eps_LF, eps_HF
    hyp0(8)  = log(1.0);  hyp0(9)  = log(1.0);   % LF space
    hyp0(10) = log(1.0);  hyp0(11) = log(1.0);   % HF space

    if numel(hyp0) >= 14
        hyp0(12) = log(0.5);  % rho-GP sigma
        hyp0(13) = log(1.0);  % rho-GP ell1
        hyp0(14) = log(1.0);  % rho-GP ell2
    end
end
