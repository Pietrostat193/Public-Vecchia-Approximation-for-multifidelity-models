function results = compare_rho_LOSO_realdata(data, opts)
% Compare rho approaches on real dataset with Leave-One-Station-Out (LOSO).
% LF = Wind_speed, HF = ws
%
% Required columns in data table:
%   Wind_speed, ws, Lat_LF, Lon_LF, Lat_HF, Lon_HF, Time, IDStation
%
% Dependencies (must be on path):
%   likelihoodVecchia_nonstat_GLS
%   predictVecchia_CM_calibrated2
%
% Example:
%   results = compare_rho_LOSO_realdata(data.sorted_data, struct('holdout_station',525));
%
% Options (opts):
%   holdout_station      : scalar or vector, station IDs to hold out
%   do_all_stations      : true/false
%   nn_size              : default 50
%   cand_mult            : default 10
%   max_iter             : default 200
%   max_fun              : default 2000
%   n_starts             : default 3 (multi-start)
%   time_cap_per_station : default 400 (set [] to use all)
%   seed                 : rng seed for subsampling and multistart

    if nargin < 2, opts = struct(); end
    if ~isfield(opts,'holdout_station'), opts.holdout_station = []; end
    if ~isfield(opts,'do_all_stations'), opts.do_all_stations = false; end

    % --- Vecchia / model defaults
    if ~isfield(opts,'nn_size'),  opts.nn_size  = 50; end
    if ~isfield(opts,'cand_mult'),opts.cand_mult= 10; end
    if ~isfield(opts,'conditioning'), opts.conditioning = "Corr"; end
    if ~isfield(opts,'kernel'), opts.kernel = "RBF"; end
    if ~isfield(opts,'MeanFunction'), opts.MeanFunction = "zero"; end
    if ~isfield(opts,'cov_type'), opts.cov_type = "RBF"; end
    if ~isfield(opts,'combination'), opts.combination = "multiplicative"; end

    % --- prediction defaults
    if ~isfield(opts,'calib_mode'), opts.calib_mode = "global_affine"; end
    if ~isfield(opts,'gamma_clip'), opts.gamma_clip = [0.25, 4.0]; end
    if ~isfield(opts,'lambda_ridge'), opts.lambda_ridge = 1e-8; end

    % --- optimization defaults (improved)
    if ~isfield(opts,'max_iter'), opts.max_iter = 200; end
    if ~isfield(opts,'max_fun'),  opts.max_fun  = 2000; end
    if ~isfield(opts,'n_starts'), opts.n_starts = 3; end
    if ~isfield(opts,'seed'), opts.seed = 42; end

    % --- dataset size control
    if ~isfield(opts,'time_cap_per_station'), opts.time_cap_per_station = 400; end

    requiredVars = ["Wind_speed","ws","Lat_LF","Lon_LF","Lat_HF","Lon_HF","Time","IDStation"];
    missing = setdiff(requiredVars, string(data.Properties.VariableNames));
    if ~isempty(missing)
        error("Missing required columns: %s", strjoin(missing,", "));
    end

    % Ensure numeric
    if ~isnumeric(data.Time), data.Time = double(data.Time); end
    if ~isnumeric(data.IDStation), data.IDStation = double(data.IDStation); end

    stations = unique(data.IDStation);
    stations = stations(:);

    if isempty(opts.holdout_station)
        holdouts = stations;
        if ~opts.do_all_stations
            holdouts = stations(1);
        end
    else
        holdouts = opts.holdout_station(:);
    end

    cases = { ...
        struct('name',"rho constant",      'RhoFunction',"constant"), ...
        struct('name',"rho GP_scaled_emp", 'RhoFunction',"GP_scaled_empirical") ...
    };

    results_rows = [];
    row = 0;

    for h = 1:numel(holdouts)
        hold_id = holdouts(h);
        fprintf('\n================ LOSO holdout station %g ================\n', hold_id);

        is_hold  = (data.IDStation == hold_id);
        train_tbl = data(~is_hold,:);
        test_tbl  = data(is_hold,:);

        % -------- optional: cap to 400 time points per station (train & test)
        if ~isempty(opts.time_cap_per_station)
            rng(opts.seed);

            train_tbl = cap_times_per_station(train_tbl, opts.time_cap_per_station);
            test_tbl  = cap_times_per_station(test_tbl,  opts.time_cap_per_station);
        end

        % Build training arrays
        X_L = [double(train_tbl.Time), double(train_tbl.Lat_LF), double(train_tbl.Lon_LF)];
        y_L = double(train_tbl.Wind_speed);

        X_H = [double(train_tbl.Time), double(train_tbl.Lat_HF), double(train_tbl.Lon_HF)];
        y_H = double(train_tbl.ws);

        % Test points: predict HF at held-out station
        Xstar  = [double(test_tbl.Time), double(test_tbl.Lat_HF), double(test_tbl.Lon_HF)];
        y_true = double(test_tbl.ws);

        fprintf('Train: nL=%d, nH=%d | Test(HF)=%d\n', size(X_L,1), size(X_H,1), size(Xstar,1));

        % Sorting (stabilizes Vecchia ordering)
        [~, pL] = sortrows(X_L, [2 3 1]);  % space-major (lat,lon) then time
        X_L = X_L(pL,:); y_L = y_L(pL);

        [~, pH] = sortrows(X_H, [2 3 1]);
        X_H = X_H(pH,:); y_H = y_H(pH);

        % ---------------- Fit: constant first (for warm-start) ----------------
        fit = struct();
        for c = 1:numel(cases)
            global ModelInfo;
            ModelInfo = struct();
            ModelInfo.X_L = X_L;  ModelInfo.y_L = y_L;
            ModelInfo.X_H = X_H;  ModelInfo.y_H = y_H;

            ModelInfo.jitter = 1e-8;
            ModelInfo.nn_size = opts.nn_size;
            ModelInfo.kernel = opts.kernel;
            ModelInfo.conditioning = opts.conditioning;
            ModelInfo.cand_mult = opts.cand_mult;
            ModelInfo.show_path_diag = false;

            ModelInfo.MeanFunction = opts.MeanFunction;
            ModelInfo.cov_type = opts.cov_type;
            ModelInfo.combination = opts.combination;
            ModelInfo.RhoFunction = cases{c}.RhoFunction;

            % Reset neighbor caches
            if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
            if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

            obj = @(hh) likelihoodVecchia_nonstat_GLS(hh);

            % ---- initialization
            if c == 1
                hyp_base0 = default_hyp_init("constant");
                hyp0 = hyp_base0;
            else
                % warm start: base 11 from constant optimum, then add GP-rho params
                hyp0 = default_hyp_init("GP_scaled_empirical");
                if isfield(fit,'hyp_hat_const') && ~isempty(fit.hyp_hat_const)
                    hyp0(1:11) = fit.hyp_hat_const(1:11);
                end
            end

            % ---- multistart around hyp0
            rng(opts.seed + 100*c);
            [hyp_hat, nlml] = run_multistart(obj, hyp0, opts);

            % finalize: cache all ModelInfo fields (debug_vecchia, gprModel_rho, etc.)
            ModelInfo.hyp = hyp_hat;
            nlml = likelihoodVecchia_nonstat_GLS(hyp_hat);

            % predict
            predOpts = struct();
            predOpts.calib_mode   = char(opts.calib_mode);
            predOpts.gamma_clip   = opts.gamma_clip;
            predOpts.lambda_ridge = opts.lambda_ridge;
            predOpts.gamma_subset = [];
            predOpts.seed         = opts.seed;


            mu_pred = predictVecchia_CM_calibrated2(Xstar, predOpts);
            rmse = sqrt(mean((mu_pred - y_true).^2));
            mae  = mean(abs(mu_pred - y_true));

            % store
            row = row + 1;
            results_rows(row).holdout_station = hold_id;
            results_rows(row).case_name = cases{c}.name;
            results_rows(row).RhoFunction = string(cases{c}.RhoFunction);
            results_rows(row).NLML = nlml;
            results_rows(row).RMSE = rmse;
            results_rows(row).MAE  = mae;
            results_rows(row).nTrainL = size(X_L,1);
            results_rows(row).nTrainH = size(X_H,1);
            results_rows(row).nTestH  = size(Xstar,1);
            results_rows(row).ModelInfo=ModelInfo
            results_rows(row).mu_pred = mu_pred; %#ok<NASGU>
            results_rows(row).y_true = y_true;   %#ok<NASGU>
            results_rows(row).Xstar  = Xstar;    %#ok<NASGU>

            fprintf('[%s] NLML=%.3f | RMSE=%.3f | MAE=%.3f\n', ...
                cases{c}.name, nlml, rmse, mae);

            % cache constant hyp for warm-start
            if c == 1
                fit.hyp_hat_const = hyp_hat;
            end

            % keep predictions for combined plots
            fit.(matlab.lang.makeValidName(string(cases{c}.RhoFunction))).mu_pred = mu_pred;
            fit.(matlab.lang.makeValidName(string(cases{c}.RhoFunction))).nlml = nlml;
            fit.(matlab.lang.makeValidName(string(cases{c}.RhoFunction))).rmse = rmse;
        end

        % ---------------- Combined plots (same figures) ----------------
        % 1) HF true vs both predictions
        figure;
        plot(y_true,'k.-','DisplayName','HF true'); hold on;
        plot(fit.constant.mu_pred,'o-','DisplayName','rho constant');
        plot(fit.GP_scaled_empirical.mu_pred,'x-','DisplayName','rho GP\_scaled\_emp');
        grid on; legend('Location','best');
        title(sprintf('Holdout station %g | nn=%d | HF predictions', hold_id, opts.nn_size));
        xlabel('test index'); ylabel('ws (HF)');

        % 2) Parity plot (both on same axes)
        figure; hold on; grid on;
        scatter(y_true, fit.constant.mu_pred, 22, 'filled', 'DisplayName','rho constant');
        scatter(y_true, fit.GP_scaled_empirical.mu_pred, 22, 'DisplayName','rho GP\_scaled\_emp');
        xlabel('HF true'); ylabel('HF predicted');
        legend('Location','best');
        title(sprintf('Holdout station %g | Parity plot', hold_id));

        % 3) Residual comparison
        figure; hold on; grid on;
        r1 = fit.constant.mu_pred - y_true;
        r2 = fit.GP_scaled_empirical.mu_pred - y_true;
        plot(r1,'o-','DisplayName','residuals: rho constant');
        plot(r2,'x-','DisplayName','residuals: rho GP\_scaled\_emp');
        yline(0,'k-');
        legend('Location','best');
        title(sprintf('Holdout station %g | Residuals', hold_id));
        xlabel('test index'); ylabel('pred - true');
    end

    results = struct2table(results_rows);

    if height(results) > 2
        fprintf('\n=========== SUMMARY (mean over holdouts) ===========\n');
        G = groupsummary(results, "RhoFunction", "mean", ["RMSE","MAE","NLML"]);
        disp(G);
    end
end

% ============================ helpers ============================

function tbl = cap_times_per_station(tbl, capN)
% Keep at most capN rows per station, sampling uniformly over time.
    if capN <= 0 || height(tbl) <= capN, return; end
    ids = unique(tbl.IDStation);
    keep = false(height(tbl),1);
    for i = 1:numel(ids)
        idx = find(tbl.IDStation == ids(i));
        if numel(idx) <= capN
            keep(idx) = true;
        else
            % sample evenly across sorted Time
            [~, ord] = sort(tbl.Time(idx));
            idx = idx(ord);
            pick = round(linspace(1, numel(idx), capN));
            keep(idx(pick)) = true;
        end
    end
    tbl = tbl(keep,:);
end

function [best_x, best_f] = run_multistart(obj, x0, opts)
% Multi-start optimization with warm-start + jittered restarts.
    n_starts = max(1, opts.n_starts);
    starts = cell(n_starts,1);
    starts{1} = x0;

    % small perturbations (log-params tolerate 0.05-0.2)
    for k = 2:n_starts
        jitter = 0.10 * randn(size(x0));
        starts{k} = x0 + jitter;
    end

    best_f = inf;
    best_x = x0;

    use_fminunc = license('test','optimization_toolbox');
    if use_fminunc
        o = optimoptions('fminunc', ...
            'Display','iter', ...
            'Algorithm','quasi-newton', ...
            'MaxIterations', opts.max_iter, ...
            'MaxFunctionEvaluations', opts.max_fun, ...
            'OptimalityTolerance', 1e-3, ...
            'StepTolerance', 1e-8);
    else
        o = optimset('Display','iter', ...
            'MaxIter', opts.max_iter, ...
            'MaxFunEvals', opts.max_fun);
    end

    for k = 1:n_starts
        xk0 = starts{k};
        try
            if use_fminunc
                [xk, fk] = fminunc(obj, xk0, o);
            else
                [xk, fk] = fminsearch(obj, xk0, o);
            end
        catch ME
            warning('Start %d failed: %s', k, ME.message);
            continue;
        end

        if fk < best_f
            best_f = fk;
            best_x = xk;
        end
    end
end

function hyp0 = default_hyp_init(RhoFunction)
    RF = string(RhoFunction);

    switch RF
        case "constant"
            hyp0 = zeros(11,1);
        otherwise
            hyp0 = zeros(14,1);
    end

    % base 11
    hyp0(1)  = log(1.0);  hyp0(2)  = log(0.20);  % LF time
    hyp0(3)  = log(1.0);  hyp0(4)  = log(0.20);  % HF time
    hyp0(5)  = 0.6;                              % rho
    hyp0(6)  = log(0.10); hyp0(7)  = log(0.10);  % eps_LF, eps_HF
    hyp0(8)  = log(1.0);  hyp0(9)  = log(1.0);   % LF space
    hyp0(10) = log(1.0);  hyp0(11) = log(1.0);   % HF space

    if numel(hyp0) >= 14
        hyp0(12) = log(0.5);  % rho GP sigma
        hyp0(13) = log(1.0);  % rho GP ell1
        hyp0(14) = log(1.0);  % rho GP ell2
    end
end
