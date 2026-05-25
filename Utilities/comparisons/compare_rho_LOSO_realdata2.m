function results = compare_rho_LOSO_realdata2(data, opts)
% Compare rho approaches on real dataset with Leave-One-Station-Out (LOSO).
% LF = Wind_speed, HF = ws
%
% Required columns in data table:
%   Wind_speed, ws, Lat_LF, Lon_LF, Lat_HF, Lon_HF, Time, IDStation
%
% Models evaluated:
%   1) rho constant
%   2) rho GP_scaled_empirical
%   3) warpMFGP (warping + constant rho, MeanFunction='zero')
%
% Dependencies (must be on path):
%   likelihoodVecchia_nonstat_GLS
%   predict_calibratedCM3
%   compute_rho_star
%   KCDF_Estim, Gen_Lookup, Kernel_invNS
%
% Example:
%   results = compare_rho_LOSO_realdata(data.sorted_data, struct());
%
% Options (opts):
%   holdout_station      : scalar or vector; if empty -> all stations (default)
%   do_all_stations      : if holdout_station empty, run all (default true)
%   nn_size              : default 50
%   cand_mult            : default 10
%   conditioning         : default "Corr"
%   kernel               : default "RBF"
%   MeanFunction         : default "zero"
%   cov_type             : default "RBF"
%   combination          : default "multiplicative"
%
%   calib_mode           : default "global_affine"
%   gamma_clip           : default [0.25, 4.0]
%   lambda_ridge         : default 1e-8
%
%   max_iter             : default 200
%   max_fun              : default 2000
%   n_starts             : default 3
%   seed                 : default 42
%
%   time_cap_per_station : default 400 (set [] to use all)
%
%   ci_alpha             : default 0.05 (95% CI)
%
%   % warp options
%   warp_kernel          : default opts.kernel
%   warp_max_iter        : default 100
%   HypInit              : optional initial hyp vector (if empty -> default_hyp_init)

    % -------- WMFGP PATH + kernel default --------
if ~isfield(opts,'wmfgp_path')
    opts.wmfgp_path = 'C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\5_Warping\WMFGP-main (1)\WMFGP-main';
end
if exist(opts.wmfgp_path,'dir')
    addpath(genpath(opts.wmfgp_path));
else
    warning('WMFGP path not found: %s', opts.wmfgp_path);
end

% default kernel requested
kernel = 'Tria';
if ~isfield(opts,'kernel'),       opts.kernel = kernel; end
if ~isfield(opts,'warp_kernel'),  opts.warp_kernel = kernel; end


     

    if nargin < 2, opts = struct(); end
    % --- which stations to hold out
    if ~isfield(opts,'holdout_station'), opts.holdout_station = []; end
    if ~isfield(opts,'do_all_stations'), opts.do_all_stations = true; end

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

    % --- optimization defaults
    if ~isfield(opts,'max_iter'), opts.max_iter = 200; end
    if ~isfield(opts,'max_fun'),  opts.max_fun  = 2000; end
    if ~isfield(opts,'n_starts'), opts.n_starts = 3; end
    if ~isfield(opts,'seed'), opts.seed = 42; end

    % --- dataset size control
    if ~isfield(opts,'time_cap_per_station'), opts.time_cap_per_station = 400; end

    % --- CI
    if ~isfield(opts,'ci_alpha'), opts.ci_alpha = 0.05; end

    % --- warp options
    if ~isfield(opts,'warp_kernel'),   opts.warp_kernel = opts.kernel; end
    if ~isfield(opts,'warp_max_iter'), opts.warp_max_iter = 100; end
    if ~isfield(opts,'HypInit'), opts.HypInit = []; end

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
        if opts.do_all_stations
            holdouts = stations;
        else
            holdouts = stations(1);
        end
    else
        holdouts = opts.holdout_station(:);
    end

    % ---- CASES (3 models) ----
    cases = { ...
        struct('name',"rho constant",        'type',"plain", 'RhoFunction',"constant"), ...
        struct('name',"rho GP_scaled_emp",   'type',"plain", 'RhoFunction',"GP_scaled_empirical"), ...
        struct('name',"warpMFGP",            'type',"warp",  'RhoFunction',"constant") ...
    };

    results_rows = [];
    row = 0;

    for h = 1:numel(holdouts)
        hold_id = holdouts(h);
        fprintf('\n================ LOSO holdout station %g (%d/%d) ================\n', hold_id, h, numel(holdouts));

        is_hold   = (data.IDStation == hold_id);
        train_tbl = data(~is_hold,:);
        test_tbl  = data(is_hold,:);

        % ---- optional: cap to time_cap_per_station rows per station
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

        % ---------------- Fit/Predict each case ----------------
        fit = struct();

        for c = 1:numel(cases)
            caseC = cases{c};
            fprintf('\n--- Case: %s ---\n', caseC.name);

            % ---- set up ModelInfo fresh each run
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
            ModelInfo.RhoFunction = caseC.RhoFunction;

            % Reset neighbor caches
            if isfield(ModelInfo,'vecchia_idxL'), ModelInfo = rmfield(ModelInfo,'vecchia_idxL'); end
            if isfield(ModelInfo,'vecchia_idxH'), ModelInfo = rmfield(ModelInfo,'vecchia_idxH'); end

            % ---- objective
            obj = @(hh) likelihoodVecchia_nonstat_GLS(hh);

            % ---- initialization / warm start
            if c == 1
                hyp0 = default_hyp_init("constant");
            elseif c == 2
                hyp0 = default_hyp_init("GP_scaled_empirical");
                if isfield(fit,'hyp_hat_const') && ~isempty(fit.hyp_hat_const)
                    hyp0(1:11) = fit.hyp_hat_const(1:11);
                end
            else
                % warpMFGP uses constant rho init (or opts.HypInit if provided)
                if ~isempty(opts.HypInit)
                    hyp0 = opts.HypInit(:);
                else
                    hyp0 = default_hyp_init("constant");
                end
            end

            % ---------------- special: WARPING MODEL ----------------
            warp_meta = struct();
            if caseC.type == "warp"
                % Enforce requested settings
                ModelInfo.RhoFunction = "constant";
                ModelInfo.MeanFunction = "zero";

                % Save originals
                yH_orig = ModelInfo.y_H;
                yL_orig = ModelInfo.y_L;

                kernelW = string(opts.warp_kernel);

                % Warping step (as you specified)
                [~, y_H_normdata, bgk_band_y_H, ~] = KCDF_Estim(ModelInfo.y_H, kernelW);
                lookup_y_H = Gen_Lookup(ModelInfo.y_L, bgk_band_y_H, kernelW);

                [~, y_L_normdata, bgk_band_y_L, ~] = KCDF_Estim(ModelInfo.y_L, kernelW);
                lookup_y_L = Gen_Lookup(ModelInfo.y_L, bgk_band_y_L, kernelW);

                % Replace data with warped/normalized for fitting/prediction
                ModelInfo.y_H = y_H_normdata;
                ModelInfo.y_L = y_L_normdata;

                warp_meta.kernel = kernelW;
                warp_meta.bgk_band_y_H = bgk_band_y_H;
                warp_meta.bgk_band_y_L = bgk_band_y_L;
                warp_meta.lookup_y_H = lookup_y_H;
                warp_meta.lookup_y_L = lookup_y_L;
                warp_meta.yH_orig = yH_orig;
                warp_meta.yL_orig = yL_orig;
            end

            % ---- multistart optimization
            rng(opts.seed + 100*c);
            [hyp_hat, nlml] = run_multistart(obj, hyp0, opts, caseC);

            % finalize (cache in ModelInfo)
            ModelInfo.hyp = hyp_hat;
            nlml = likelihoodVecchia_nonstat_GLS(hyp_hat); %#ok<NASGU>

            % ---- prediction with variance (predict_calibratedCM3)
            predOpts = struct();
            predOpts.calib_mode   = char(opts.calib_mode);
            predOpts.gamma_clip   = opts.gamma_clip;
            predOpts.lambda_ridge = opts.lambda_ridge;
            predOpts.gamma_subset = [];
            predOpts.seed         = opts.seed;

            [mu_pred, s2_pred] = predict_calibratedCM3(Xstar, predOpts);

            % ---- CI in model scale
            alpha = opts.ci_alpha;
            z = norminv(1 - alpha/2);
            sd = sqrt(max(s2_pred,0));
            lo = mu_pred - z*sd;
            hi = mu_pred + z*sd;

            % ---- if warp model: inverse map predictions to original HF scale
            mu_pred_inv = []; lo_inv = []; hi_inv = [];
            if caseC.type == "warp"
                mu_pred_inv = Kernel_invNS(mu_pred, warp_meta.lookup_y_H);

                % approximate CI via inverse mapping of bounds (simple + common)
                lo_inv = Kernel_invNS(lo, warp_meta.lookup_y_H);
                hi_inv = Kernel_invNS(hi, warp_meta.lookup_y_H);

                % For error metrics, compare on original scale:
                y_ref = y_true;
                rmse = sqrt(mean((mu_pred_inv - y_ref).^2));
                mae  = mean(abs(mu_pred_inv - y_ref));

            else
                rmse = sqrt(mean((mu_pred - y_true).^2));
                mae  = mean(abs(mu_pred - y_true));
            end

            % ---- store run
            row = row + 1;
            results_rows(row).holdout_station = hold_id;
            results_rows(row).case_name = caseC.name;
            results_rows(row).case_type = string(caseC.type);
            results_rows(row).RhoFunction = string(caseC.RhoFunction);

            % optimization
            results_rows(row).hyp_hat = hyp_hat; %#ok<NASGU>
            results_rows(row).NLML = ModelInfo.nlml; % if your likelihood sets it; else we keep empty
            if ~isfield(results_rows(row),'NLML') || isempty(results_rows(row).NLML)
                % fallback: store last computed objective if not in ModelInfo
                results_rows(row).NLML = likelihoodVecchia_nonstat_GLS(hyp_hat);
            end

            % metrics
            results_rows(row).RMSE = rmse;
            results_rows(row).MAE  = mae;

            % sizes
            results_rows(row).nTrainL = size(X_L,1);
            results_rows(row).nTrainH = size(X_H,1);
            results_rows(row).nTestH  = size(Xstar,1);

            % data & preds (save everything for each run)
            results_rows(row).Xstar = Xstar; %#ok<NASGU>
            results_rows(row).y_true = y_true; %#ok<NASGU>
            results_rows(row).mu_pred = mu_pred; %#ok<NASGU>
            results_rows(row).s2_pred = s2_pred; %#ok<NASGU>
            results_rows(row).ci_lo = lo; %#ok<NASGU>
            results_rows(row).ci_hi = hi; %#ok<NASGU>

            % warp outputs (optional)
            results_rows(row).mu_pred_inv = mu_pred_inv; %#ok<NASGU>
            results_rows(row).ci_lo_inv = lo_inv; %#ok<NASGU>
            results_rows(row).ci_hi_inv = hi_inv; %#ok<NASGU>
            results_rows(row).warp_meta = warp_meta; %#ok<NASGU>

            % keep ModelInfo snapshot (heavy but requested)
            results_rows(row).ModelInfo = ModelInfo; %#ok<NASGU>

            fprintf('[%s] RMSE=%.3f | MAE=%.3f | NLML=%.3f\n', ...
                caseC.name, rmse, mae, results_rows(row).NLML);

            % cache constant hyp for warm-start
            if c == 1
                fit.hyp_hat_const = hyp_hat;
            end

            % keep predictions for quick plots
            nm = matlab.lang.makeValidName(string(caseC.name));
            fit.(nm).mu_pred = mu_pred;
            fit.(nm).s2_pred = s2_pred;
            fit.(nm).ci_lo = lo;
            fit.(nm).ci_hi = hi;
            fit.(nm).rmse = rmse;
        end

        % ---------------- quick plots for this holdout ----------------
        % Plot on ORIGINAL scale:
        % - plain models: mu_pred directly
        % - warpMFGP: use mu_pred_inv if available
        figure; hold on; grid on;
        plot(y_true,'k.-','DisplayName','HF true');

        for c = 1:numel(cases)
            nm = matlab.lang.makeValidName(string(cases{c}.name));
            if cases{c}.type == "warp" && ~isempty(results_rows(row - (numel(cases)-c)).mu_pred_inv)
                mu_plot = results_rows(row - (numel(cases)-c)).mu_pred_inv;
                plot(mu_plot,'-','DisplayName',cases{c}.name);
            else
                plot(fit.(nm).mu_pred,'-','DisplayName',cases{c}.name);
            end
        end

        title(sprintf('Holdout station %g | HF predictions (all models)', hold_id));
        xlabel('test index'); ylabel('ws (HF)');
        legend('Location','best');

        % Example: plot CI for the first model (optional)
        % (you can replicate for the others)
    end

    results = struct2table(results_rows);

    if height(results) > 1
        fprintf('\n=========== SUMMARY (mean over holdouts) ===========\n');
        G = groupsummary(results, "case_name", "mean", ["RMSE","MAE","NLML"]);
        disp(G);
    end
end

% ============================ helpers ============================

function tbl = cap_times_per_station(tbl, capN)
% Keep at most capN rows per station, sampling evenly over sorted time.
    if capN <= 0 || height(tbl) <= capN, return; end
    ids = unique(tbl.IDStation);
    keep = false(height(tbl),1);
    for i = 1:numel(ids)
        idx = find(tbl.IDStation == ids(i));
        if numel(idx) <= capN
            keep(idx) = true;
        else
            [~, ord] = sort(tbl.Time(idx));
            idx = idx(ord);
            pick = round(linspace(1, numel(idx), capN));
            keep(idx(pick)) = true;
        end
    end
    tbl = tbl(keep,:);
end

function [best_x, best_f] = run_multistart(obj, x0, opts, caseC)
% Multi-start optimization with warm-start + jittered restarts.
    n_starts = max(1, opts.n_starts);
    starts = cell(n_starts,1);
    starts{1} = x0;

    for k = 2:n_starts
        jitter = 0.10 * randn(size(x0));
        starts{k} = x0 + jitter;
    end

    best_f = inf;
    best_x = x0;

    use_fminunc = license('test','optimization_toolbox');
    if use_fminunc
        maxit = opts.max_iter;
        if isfield(opts,'warp_max_iter') && caseC.type=="warp"
            maxit = opts.warp_max_iter;
        end

        o = optimoptions('fminunc', ...
            'Algorithm','quasi-newton', ...
            'Display','iter', ...
            'MaxIterations', maxit, ...
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
% Same idea as your init, kept compatible.
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
