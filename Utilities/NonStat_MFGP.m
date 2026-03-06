function [m, v, ModelInfo, ci_lower_all, ci_upper_all] = NonStat_MFGP(X_L, X_H, y_L, y_H, varargin)
% NonStat_MFGP  Non-stationary multi-fidelity GP with bounded hyperparameter optimization.
%
% New opt_method='tighten': staged Vecchia with frozen neighbors (nn grid = [20 40 60 100]).
% Keeps only three methods: 'local' (fmincon), 'quasi-newton' (fminunc), 'tighten' (staged fmincon).
%
% --- signature & defaults unchanged except opt_method ---

    % -------- Parse inputs --------
    p = inputParser;
    addParameter(p, 'HypInit', zeros(11,1));
    addParameter(p, 'Kernel', 'Matern');
    addParameter(p, 'NNSize', 50);
    addParameter(p, 'MeanFunction', 'zero');
    addParameter(p, 'RhoFunction', 'linear');
    addParameter(p, 'Conditioning', 'Corr');      % <- default Corr
    addParameter(p, 'PredictionX', X_L);
    addParameter(p, 'opt_method', 'tighten');     % <- default tighten
    addParameter(p, 'Bounds', struct([]));
    addParameter(p, 'UseDataDrivenBounds', true);
    addParameter(p, 'TimeColumn', 1);
    addParameter(p, 'SpaceColumns', []);
    addParameter(p, 'BlockSize', 10000);
    addParameter(p, 'RecordTrajectory', true);
    parse(p, varargin{:});

    kernel         = p.Results.Kernel;
    nn_size        = p.Results.NNSize;
    mean_function  = p.Results.MeanFunction;
    rho_function   = p.Results.RhoFunction;
    conditioning   = p.Results.Conditioning;
    prediction_X   = p.Results.PredictionX;
    hyp_init       = p.Results.HypInit(:);
    opt_method     = lower(p.Results.opt_method);
    Bounds         = p.Results.Bounds;
    useDataDriven  = p.Results.UseDataDrivenBounds;
    time_col       = p.Results.TimeColumn;
    space_cols     = p.Results.SpaceColumns;
    block_size     = p.Results.BlockSize;
    record_traj    = p.Results.RecordTrajectory;

    if isempty(space_cols)
        space_cols = setdiff(1:size(X_L,2), time_col);
    end

    % -------- Setup ModelInfo --------
    global ModelInfo;
    ModelInfo = struct();
    ModelInfo.MeanFunction = mean_function;
    ModelInfo.RhoFunction  = rho_function;
    ModelInfo.kernel       = kernel;
    ModelInfo.nn_size      = nn_size;
    ModelInfo.jitter       = 1e-6;
    ModelInfo.conditioning = conditioning;
    ModelInfo.X_H = X_H;
    ModelInfo.X_L = X_L;
    ModelInfo.y_H = y_H;
    ModelInfo.y_L = y_L;

    % -------- Hyperparameter bounds --------
    n_params = numel(hyp_init);
    assert(n_params==11, 'HypInit must be 11x1 (log-params except rho raw).');

    if ~isempty(Bounds) && isfield(Bounds,'lb') && isfield(Bounds,'ub') ...
            && ~isempty(Bounds.lb) && ~isempty(Bounds.ub)
        lb = Bounds.lb(:);  ub = Bounds.ub(:);
        names = default_param_names();
    else
        if useDataDriven
            [lb, ub, names] = make_bounds_from_data(X_L, X_H, y_L, y_H, time_col, space_cols);
        else
            [lb, ub, names] = manual_wide_bounds();
        end
    end
    assert(numel(lb)==n_params && numel(ub)==n_params, 'Bounds size mismatch.');
    ModelInfo.Bounds.lb    = lb;
    ModelInfo.Bounds.ub    = ub;
    ModelInfo.Bounds.Names = names;

    % Ensure init is inside the box
    hyp_init = min(max(hyp_init, lb), ub);

    % -------- Likelihood handle (boxed) --------
    safe_like = @(x) catch_eval(@() likelihoodVecchia_nonstat(x), 1e10);

    % -------- Optimize --------
    all_hyp  = [];
    all_fval = [];

    switch opt_method
        case 'local'   % single bounded fmincon (no staging)
            options = stage_fmincon_options(record_traj, @outfun_collect);
            ModelInfo.nn_size      = nn_size;
            ModelInfo.idxFixed_L   = [];    % no freezing
            ModelInfo.idxFixed_H   = [];
            [hyp_opt, ~] = fmincon(@likelihoodVecchia_nonstat, hyp_init, ...
                                   [],[],[],[], lb, ub, [], options);

        case 'quasi-newton'   % unconstrained (kept for parity)
            options = optimoptions('fminunc', ...
                'Algorithm','quasi-newton', ...
                'SpecifyObjectiveGradient', false, ...
                'FiniteDifferenceType','central', ...
                'FiniteDifferenceStepSize',1e-4, ...
                'TypicalX', abs(hyp_init)+1, ...
                'Display','iter', ...
                'MaxIterations', 200, ...
                'MaxFunctionEvaluations', 5000, ...
                'FunctionTolerance',1e-8, ...
                'StepTolerance',1e-8);
            try, options.OptimalityTolerance = 1e-10; end
            try, options.TolFun              = 1e-12; end
            try, options.TolX                = 1e-12; end
            if record_traj, options.OutputFcn = @outfun_collect; end
            % (No bounds here)
            [hyp_opt, ~] = fminunc(@likelihoodVecchia_nonstat, hyp_init, options);

        case 'tighten'   % <<< staged Vecchia with frozen neighbors
            % Fixed grid requested
            nn_grid = [20 40 60 100];

            hyp_curr = hyp_init;
            for k = 1:numel(nn_grid)
                ModelInfo.nn_size = nn_grid(k);

                % --- compute current ell_t / ell_s from hyp_curr ---
                %    (use the average of LF/HF length-scales for freezing)
                t_ell_LF = exp(hyp_curr(2));
                t_ell_HF = exp(hyp_curr(4));
                s_ell_LF = exp(hyp_curr(9));
                s_ell_HF = exp(hyp_curr(11));
                ell_t = mean([t_ell_LF, t_ell_HF]);
                ell_s = mean([s_ell_LF, s_ell_HF]);

                % --- precompute frozen neighbors (Corr only) ---
                if strcmpi(conditioning,'Corr')
                    idxL = precompute_corr_fixed_idx(X_L(:,[time_col space_cols]), nn_grid(k), ell_t, ell_s);
                    idxH = precompute_corr_fixed_idx(X_H(:,[time_col space_cols]), nn_grid(k), ell_t, ell_s);
                    ModelInfo.idxFixed_L = idxL;
                    ModelInfo.idxFixed_H = idxH;
                else
                    ModelInfo.idxFixed_L = [];
                    ModelInfo.idxFixed_H = [];
                end

                % --- bounded fmincon at this stage ---
                options = stage_fmincon_options(record_traj, @outfun_collect);
                [hyp_curr, ~] = fmincon(@likelihoodVecchia_nonstat, hyp_curr, ...
                                        [],[],[],[], lb, ub, [], options);
            end

            hyp_opt = hyp_curr;

        otherwise
            error("Unsupported opt_method. Use 'tighten', 'local', or 'quasi-newton'.");
    end

    ModelInfo.hyp = hyp_opt;
    if record_traj
        ModelInfo.all_hyp  = all_hyp;
        ModelInfo.all_fval = all_fval;
    else
        ModelInfo.all_hyp  = [];
        ModelInfo.all_fval = [];
    end

    % -------- Prediction (single block) --------
    n_total = size(prediction_X, 1);
    ModelInfo.X_L = prediction_X;  % (for predict code that uses ModelInfo.X_L)
    ModelInfo.y_L = y_L;
    likelihoodVecchia_nonstat(ModelInfo.hyp); % refresh caches if any
    [m, v, ci_lower_all, ci_upper_all] = predictVecchia_nonstat2(prediction_X);

    % -------- Nested helpers --------
    function stop = outfun_collect(x, optimValues, state)
        stop = false;
        if strcmp(state,'iter')
            all_hyp  = [all_hyp;  x(:)'];
            all_fval = [all_fval; optimValues.fval];
        end
    end

    function val = catch_eval(fun_handle, penalty)
        try
            val = fun_handle();
            if ~isfinite(val), val = penalty; end
        catch
            val = penalty;
        end
    end
end

% ---------- subfunctions (same file) ----------

function options = stage_fmincon_options(record_traj, outfun)
    options = optimoptions('fmincon', ...
        'Algorithm','interior-point', ...
        'Display','iter', ...
        'SpecifyObjectiveGradient', false, ...
        'MaxFunctionEvaluations', 5000);
    try, options.OptimalityTolerance = 1e-8;  end
    try, options.StepTolerance       = 1e-8;  end
    try, options.TolFun              = 1e-8;  end
    if record_traj, options.OutputFcn = outfun; end
end

function names = default_param_names()
    names = {'log s_sig_LF_t','log t_ell_LF', ...
             'log s_sig_HF_t','log t_ell_HF', ...
             'rho', ...
             'log eps_LF','log eps_HF', ...
             'log s_sig_LF_s','log s_ell_LF', ...
             'log s_sig_HF_s','log s_ell_HF'};
end

function [lb, ub, names] = manual_wide_bounds()
    names = default_param_names();
    % wide but reasonable
    lb = [-6; -7; -6; -7;  -5; -10; -10;  -6; -7;  -6; -7];
    ub = [ 6;  6;  6;  6;   5;   3;   3;   6;  6;   6;  6];
end

function [lb, ub, names] = make_bounds_from_data(X_L, X_H, y_L, y_H, tcol, scol)
    names = default_param_names();
    y_all = [y_L; y_H];
    yr = max(y_all)-min(y_all)+eps;
    % crude spans => log-length bounds
    t_span  = max([X_L(:,tcol); X_H(:,tcol)]) - min([X_L(:,tcol); X_H(:,tcol)]) + eps;
    s_span1 = max([X_L(:,scol(1)); X_H(:,scol(1))]) - min([X_L(:,scol(1)); X_H(:,scol(1))]) + eps;
    s_span2 = max([X_L(:,scol(2)); X_H(:,scol(2))]) - min([X_L(:,scol(2)); X_H(:,scol(2))]) + eps;
    % very safe ranges
    lb = [ log(yr)-6; log(t_span)-7;  log(yr)-6; log(t_span)-7;  -5;  log(var(y_L)+1e-6)-10; log(var(y_H)+1e-6)-10;  log(yr)-6; log(min(s_span1,s_span2))-7;  log(yr)-6; log(min(s_span1,s_span2))-7 ];
    ub = [ log(yr)+6; log(t_span)+6;  log(yr)+6; log(t_span)+6;   5;  log(var(y_L)+1e-6)+3;   log(var(y_H)+1e-6)+3;   log(yr)+6; log(max(s_span1,s_span2))+6; log(yr)+6; log(max(s_span1,s_span2))+6 ];
end

function idxFixed = precompute_corr_fixed_idx(X_ts, nn, ell_t, ell_s)
% X_ts: [n x 3] with [time, x, y]; produce (n x nn) indices of previous rows
    n = size(X_ts,1);
    idxFixed = zeros(n, nn);
    inv_ell_t2 = 1/(ell_t^2 + eps);
    inv_ell_s2 = 1/(ell_s^2 + eps);
    for i = 2:n
        J = 1:(i-1);                     % only previous
        dt = (X_ts(J,1)-X_ts(i,1)).^2 * inv_ell_t2;
        dx = (X_ts(J,2)-X_ts(i,2)).^2 * inv_ell_s2;
        dy = (X_ts(J,3)-X_ts(i,3)).^2 * inv_ell_s2;
        r2 = dt + dx + dy;
        k = min(nn, numel(J));
        [~,ord] = mink(r2, k);
        sel = J(ord);
        if numel(sel) < nn
            sel = [sel(:); zeros(nn-numel(sel),1)];
        end
        idxFixed(i,:) = sel(1:nn);
    end
end
