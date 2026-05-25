% ========================================================================
% LOSO Approximate GP baselines (FIC) for holdout-station prediction
%
% Response (HF): y = ws
% Additional covariate: y_L = Wind_speed (LF)
%
% Models:
%   (1) GP-ST      : X = [Time, Lat_HF, Lon_HF]
%   (2) GP-yL      : X = [Wind_speed]
%   (3) GP-yL+ST   : X = [Wind_speed, Time, Lat_HF, Lon_HF]
%
% Approx GP:
%   FitMethod='fic', PredictMethod='fic', ActiveSetSize=m
%
% Requires: Statistics and Machine Learning Toolbox (fitrgp)
% Data file: South_Lombardy_sorted_data.mat containing table sorted_data
% ========================================================================

clear; clc;

% ---------- 1) Load data ----------
dataPath = ...
 'C:\Users\2692812C\OneDrive - University of Glasgow\Desktop\Projects\6.VecchiaApproximation\3d_SpaceTimeVecchia\1_South_lombardy';

dataFile = fullfile(dataPath, 'South_Lombardy_sorted_data.mat');
S = load(dataFile);

disp("Loaded variables:");
disp(fieldnames(S));

% Change if your variable name differs
data = S.sorted_data;

% ---------- 2) Checks ----------
assert(istable(data), 'Loaded object is not a table.');

requiredVars = ["ws","Wind_speed","Lat_HF","Lon_HF","Time","IDStation"];
missing = setdiff(requiredVars, string(data.Properties.VariableNames));
assert(isempty(missing), "Missing required columns: %s", strjoin(missing,", "));

% ---------- 3) Settings ----------
seed = 42;
rng(seed);

do_all_stations = true;
holdout_station = [];          % e.g. 525 to run one station
time_cap_per_station = 100;    % set [] or 0 to disable
standardize_X = true;

% Approx GP settings
approxMethod = 'fic';          % 'fic' or 'sr'
mActive = 300;                 % inducing/active set size
kernelFun = 'ardsquaredexponential'; % try: 'ardmatern32', 'ardmatern52'

% ---------- 4) Holdout list ----------
stations = unique(double(data.IDStation));
stations = stations(:);

if ~isempty(holdout_station)
    holdouts = double(holdout_station(:));
else
    holdouts = stations;
    if ~do_all_stations
        holdouts = stations(1);
    end
end

% ---------- 5) Run LOSO ----------
rows = struct([]);
r = 0;

modelNames = ["GP-ST", "GP-yL", "GP-yL+ST"];

for h = 1:numel(holdouts)
    hold_id = holdouts(h);

    fprintf('\n================ LOSO holdout station %g ================\n', hold_id);

    is_hold  = (double(data.IDStation) == hold_id);
    train_tbl = data(~is_hold,:);
    test_tbl  = data(is_hold,:);

    % ---- optional cap per station (train & test)
    if ~isempty(time_cap_per_station) && time_cap_per_station > 0
        train_tbl = cap_times_per_station_simple(train_tbl, time_cap_per_station, seed);
        test_tbl  = cap_times_per_station_simple(test_tbl,  time_cap_per_station, seed);
    end

    % HF target (response)
    ytrain = double(train_tbl.ws);    % <-- YES, correct response
    ytrue  = double(test_tbl.ws);

    % LF covariate
    yL_train = double(train_tbl.Wind_speed);
    yL_test  = double(test_tbl.Wind_speed);

    % Spatio-temporal coords (HF coords)
    Xst_train = [double(train_tbl.Time), double(train_tbl.Lat_HF), double(train_tbl.Lon_HF)];
    Xst_test  = [double(test_tbl.Time),  double(test_tbl.Lat_HF),  double(test_tbl.Lon_HF)];

    fprintf('Train n=%d | Test n=%d\n', size(Xst_train,1), size(Xst_test,1));

    % For each model type
    for mi = 1:numel(modelNames)
        modelTag = modelNames(mi);

        % --- Build Xtrain / Xtest depending on model
        switch modelTag
            case "GP-ST"
                Xtrain = Xst_train;
                Xtest  = Xst_test;

            case "GP-yL"
                Xtrain = yL_train(:);
                Xtest  = yL_test(:);

            case "GP-yL+ST"
                Xtrain = [yL_train(:), Xst_train];
                Xtest  = [yL_test(:),  Xst_test];

            otherwise
                error("Unknown modelTag: %s", modelTag);
        end

        % --- Standardize inputs (recommended)
        if standardize_X
            mu = mean(Xtrain,1);
            sd = std(Xtrain,0,1);
            sd(sd==0) = 1;
            Xtrain_s = (Xtrain - mu) ./ sd;
            Xtest_s  = (Xtest  - mu) ./ sd;
        else
            Xtrain_s = Xtrain;
            Xtest_s  = Xtest;
        end

        % --- Fit approximate GP (FIC or SR)
        % Use ActiveSetMethod='random' and ActiveSetSize=mActive
        n = size(Xtrain_s,1);
        m = min(mActive, n);

        try
            gprMdl = fitrgp( ...
                Xtrain_s, ytrain, ...
                'KernelFunction', kernelFun, ...
                'FitMethod', approxMethod, ...
                'PredictMethod', approxMethod, ...
                'ActiveSetMethod', 'random', ...
                'ActiveSetSize', m, ...
                'Standardize', false);
        catch ME
            warning("fitrgp failed for station %g, model %s: %s", hold_id, modelTag, ME.message);
            continue;
        end

        % --- Predict
        ypred = predict(gprMdl, Xtest_s);

        % --- Metrics
        rmse = sqrt(mean((ypred - ytrue).^2));
        mae  = mean(abs(ypred - ytrue));

        ok = isfinite(ypred) & isfinite(ytrue);
        if sum(ok) >= 2 && std(ytrue(ok)) > 0 && std(ypred(ok)) > 0
            corr_val = corr(ytrue(ok), ypred(ok));
        else
            corr_val = NaN;
        end

        fprintf('%s | RMSE=%.5f | MAE=%.5f | CORR=%.5f | m=%d\n', ...
            modelTag, rmse, mae, corr_val, m);

        % --- store row
        r = r + 1;
        rows(r).holdout_station = hold_id;
        rows(r).Model = sprintf('%s (%s, m=%d)', modelTag, approxMethod, m);
        rows(r).RMSE = rmse;
        rows(r).MAE  = mae;
        rows(r).CORR = corr_val;
        rows(r).nTrain = n;
        rows(r).nTest  = size(Xtest_s,1);
    end
end

results_gp3 = struct2table(rows);
results_gp3 = sortrows(results_gp3, {'holdout_station','Model'});
disp(results_gp3);

% Optional: mean by model (nice summary)
results_gp3.ModelShort = categorical(regexprep(string(results_gp3.Model), '\s*\(.*\)$', ''));
G = groupsummary(results_gp3, "ModelShort", "mean", ["RMSE","MAE","CORR"]);
disp(G);

% ========================================================================
% Helper: cap times per station (simple)
% ========================================================================
function tbl = cap_times_per_station_simple(tbl, capN, seed)
    if capN <= 0 || height(tbl) <= capN, return; end
    rng(seed);

    ids = unique(double(tbl.IDStation));
    keep = false(height(tbl),1);

    for i = 1:numel(ids)
        idx = find(double(tbl.IDStation) == ids(i));
        if numel(idx) <= capN
            keep(idx) = true;
        else
            [~, ord] = sort(double(tbl.Time(idx)));
            idx = idx(ord);
            pick = round(linspace(1, numel(idx), capN));
            keep(idx(pick)) = true;
        end
    end
    tbl = tbl(keep,:);
end
