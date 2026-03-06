% ========================================================================
% COMPARE 4 MODELS (station-wise, LOSO) + OVERALL AVERAGES + WINNER COUNTS
%
% Needs in workspace:
%   results       (36x13) from load('Results_real__south_L.mat')
%   results_warp  (18x20)
%   results_gp3   OR results_gp (18x...)
% ========================================================================

clc;

assert(exist('results','var')==1 && istable(results), 'Need table "results".');
assert(exist('results_warp','var')==1 && istable(results_warp), 'Need table "results_warp".');

% ------------------------ GP table: prefer results_gp3, else results_gp, else load results_gp.mat
if ~exist('results_gp3','var')
    if exist('results_gp','var') && istable(results_gp)
        results_gp3 = results_gp;
        fprintf('Using results_gp as results_gp3.\n');
    elseif exist('results_gp.mat','file')
        Sgp = load('results_gp.mat');
        assert(isfield(Sgp,'results_gp') && istable(Sgp.results_gp), ...
            'results_gp.mat found but does not contain table "results_gp".');
        results_gp3 = Sgp.results_gp;
        fprintf('Loaded results_gp from results_gp.mat.\n');
    else
        error('Cannot find GP table (results_gp3 / results_gp / results_gp.mat).');
    end
end

% ------------------------ 1) Ensure CORR exists for all tables
results      = addCorrIfMissing(results);
results_warp = addCorrIfMissing(results_warp);
results_gp3  = addCorrIfMissing(results_gp3);

% ------------------------ 2) Extract the four model tables (Station, RMSE, MAE, CORR, Model)

A = results(strcmp(string(results.RhoFunction), "constant"), {'holdout_station','RMSE','MAE','CORR'});
A.Model = repmat("MFGP (rho constant)", height(A), 1);
A.Properties.VariableNames = {'Station','RMSE','MAE','CORR','Model'};

B = results(strcmp(string(results.RhoFunction), "GP_scaled_empirical"), {'holdout_station','RMSE','MAE','CORR'});
B.Model = repmat("MFGP (rho GP_scaled_emp)", height(B), 1);
B.Properties.VariableNames = {'Station','RMSE','MAE','CORR','Model'};

C = results_warp(:, {'holdout_station','RMSE','MAE','CORR'});
C.Model = repmat("MFGP (rho constant, warped)", height(C), 1);
C.Properties.VariableNames = {'Station','RMSE','MAE','CORR','Model'};

% GP baseline: select GP-ST if present
if ismember("Model", string(results_gp3.Properties.VariableNames)) && any(contains(string(results_gp3.Model),"GP-ST"))
    idxGP = contains(string(results_gp3.Model),"GP-ST");
else
    idxGP = true(height(results_gp3),1);
end
D = results_gp3(idxGP, {'holdout_station','RMSE','MAE','CORR'});
D.Model = repmat("GP-ST (approx)", height(D), 1);
D.Properties.VariableNames = {'Station','RMSE','MAE','CORR','Model'};

% ------------------------ 3) Long table
T_long = [A; C; B; D];                 % order: constant, warped, rhoGP, GP baseline
T_long = sortrows(T_long, {'Station','Model'});

disp('=== LONG table: Station x Model (4 models) ===');
disp(T_long);

% ------------------------ 4) Wide table (one row per station)
W = unique(T_long(:,{'Station'}));
W = sortrows(W,'Station');

W = joinOneModel(W, T_long, "MFGP (rho constant)",         "mf_const");
W = joinOneModel(W, T_long, "MFGP (rho constant, warped)", "mf_warp");
W = joinOneModel(W, T_long, "MFGP (rho GP_scaled_emp)",    "mf_gp");
W = joinOneModel(W, T_long, "GP-ST (approx)",              "gp_st");

disp('=== WIDE table: 4 models side-by-side (per station) ===');
disp(W);

% ------------------------ 5) Overall averages per model (mean and std)
Gmean = groupsummary(T_long, "Model", "mean", ["RMSE","MAE","CORR"]);
Gstd  = groupsummary(T_long, "Model", "std",  ["RMSE","MAE","CORR"]);

disp('=== Overall mean metrics per model ===');
disp(Gmean);

disp('=== Overall std metrics per model ===');
disp(Gstd);

% ------------------------ 6) Winner counts by station (RMSE)
models = ["MFGP (rho constant)", "MFGP (rho constant, warped)", ...
          "MFGP (rho GP_scaled_emp)", "GP-ST (approx)"];

rmseMat = [W.RMSE_mf_const, W.RMSE_mf_warp, W.RMSE_mf_gp, W.RMSE_gp_st];

winner = strings(height(W),1);
for i = 1:height(W)
    v = rmseMat(i,:);
    if all(isnan(v))
        winner(i) = "NA";
    else
        [~, j] = min(v, [], 'omitnan');
        winner(i) = models(j);
    end
end
W.RMSE_Winner = winner;

disp('=== RMSE winner per station ===');
disp(W(:,{'Station','RMSE_Winner'}));

% ---- Manual winner counts (version-proof)
uniqW = unique(W.RMSE_Winner);
cnt = zeros(numel(uniqW),1);
for i = 1:numel(uniqW)
    cnt(i) = sum(W.RMSE_Winner == uniqW(i));
end
winCounts = table(uniqW, cnt, 'VariableNames', {'Model','WinCount'});
winCounts = sortrows(winCounts, 'WinCount', 'descend');

disp('=== Winner counts (RMSE) ===');
disp(winCounts);

% ========================================================================
% Local helper: compute CORR if missing (requires mu_pred + y_true)
% ========================================================================
function T = addCorrIfMissing(T)
    if ~istable(T), return; end
    if ismember("CORR", string(T.Properties.VariableNames)), return; end

    if ~all(ismember(["mu_pred","y_true"], string(T.Properties.VariableNames)))
        T.CORR = nan(height(T),1);
        return;
    end

    T.CORR = nan(height(T),1);
    for i = 1:height(T)
        y = double(T.y_true{i}(:));
        yhat = double(T.mu_pred{i}(:));
        ok = isfinite(y) & isfinite(yhat);
        y = y(ok); yhat = yhat(ok);
        if numel(y) >= 2 && std(y) > 0 && std(yhat) > 0
            T.CORR(i) = corr(y, yhat);
        else
            T.CORR(i) = NaN;
        end
    end
end

% ========================================================================
% Local helper: join one model into wide table with clean VariableNames
% ========================================================================
function W = joinOneModel(W, T_long, modelName, tag)
    % Force to char to avoid string issues in VariableNames
    modelName = char(modelName);
    tag = char(tag);

    Tm = T_long(T_long.Model == string(modelName), {'Station','RMSE','MAE','CORR'});

    newNames = { ...
        'Station', ...
        sprintf('RMSE_%s', tag), ...
        sprintf('MAE_%s',  tag), ...
        sprintf('CORR_%s', tag) };

    % Only rename if table is non-empty (outerjoin handles empties, but renaming needs consistent width)
    if ~isempty(Tm)
        Tm.Properties.VariableNames = newNames;
    else
        % Create an empty table with the right variables for a safe join
        Tm = table('Size',[0 4], ...
                   'VariableTypes',{'double','double','double','double'}, ...
                   'VariableNames',newNames(1:4));
        % Add Station column explicitly as double for join consistency
        Tm.Station = double(Tm.Station);
    end

    W = outerjoin(W, Tm, 'Keys','Station', 'MergeKeys',true);
end
