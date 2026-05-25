%% ============================================================
%  Extract fitted hyperparameter vectors from TWO MAT files:
%    - results_warp.mat           -> contains `results_warp`
%    - Results_real__south_L.mat  -> contains `results`
%
%  Output:
%    - paramTable : table with one row per (station x model)
%    - paramBank  : struct for quick lookup, e.g. paramBank.constant.s_100
% =============================================================

clear; clc;

%% ---------------- USER SETTINGS ------------------------------
warpFile    = 'results_warp.mat';
mainFile    = 'Results_real__south_L.mat';

saveOut     = true;
outFile     = 'paramBank_realdata_twoFiles.mat';

%% ---------------- LOAD FILES ---------------------------------
if ~isfile(warpFile), error('Cannot find: %s', warpFile); end
if ~isfile(mainFile), error('Cannot find: %s', mainFile); end

Swarp = load(warpFile);
Sm    = load(mainFile);

fprintf('Loaded: %s\n', warpFile);
disp('Vars in warp file:'); disp(fieldnames(Swarp));

fprintf('Loaded: %s\n', mainFile);
disp('Vars in main file:'); disp(fieldnames(Sm));

% Expect these variable names (edit here if yours differ)
if isfield(Sm,'results')
    results = Sm.results;
else
    error('`results` not found in %s. Rename in code if your table has a different name.', mainFile);
end

if isfield(Swarp,'results_warp')
    results_warp = Swarp.results_warp;
else
    error('`results_warp` not found in %s. Rename in code if your table has a different name.', warpFile);
end

%% ---------------- CANONICAL MODEL MAPPING ---------------------
% Adjust mapping rules if your strings differ.
canon = struct();

% constant (non-warp)
canon.constant = @(cn,rf) contains(cn,"rho constant") & rf=="constant" & ~contains(cn,"warped");

% warped constant
canon.warped   = @(cn,rf) contains(cn,"warped") & rf=="constant";

% empirical GP-scaled rho
canon.gp_scaled_emp = @(cn,rf) contains(cn,"GP_scaled_emp") | rf=="GP_scaled_empirical";

% GP baseline (if present somewhere)
canon.gp_st = @(cn,rf) contains(cn,"GP-ST") | contains(cn,"GP3D") | contains(cn,"GP-3D") | contains(cn,"GP-ST (approx)");

%% ---------------- EXTRACT BOTH TABLES -------------------------
rows_main = localExtract(results,      "results_main", canon);
rows_warp = localExtract(results_warp, "results_warp", canon);

paramTable = [rows_main; rows_warp];

if isempty(paramTable) || height(paramTable)==0
    error('No rows extracted. Check your strings in case_name/RhoFunction and the canon rules.');
end

%% ---------------- BUILD PARAM BANK ----------------------------
paramBank = struct();

for i = 1:height(paramTable)
    mk  = paramTable.modelKey{i};
    sid = paramTable.holdout_station(i);
    hyp = paramTable.hyp{i};

    if ~isfield(paramBank, mk)
        paramBank.(mk) = struct();
    end

    fStation = sprintf('s_%g', sid);
    paramBank.(mk).(fStation) = hyp;
end

%% ---------------- QUICK CHECKS --------------------------------
fprintf('\nExtracted counts per modelKey:\n');
disp(groupcounts(categorical(string(paramTable.modelKey))));

fprintf('\nExample lookup (station 100):\n');
if isfield(paramBank,'constant') && isfield(paramBank.constant,'s_100')
    fprintf('constant hyp length: %d\n', numel(paramBank.constant.s_100));
end
if isfield(paramBank,'warped') && isfield(paramBank.warped,'s_100')
    fprintf('warped hyp length:   %d\n', numel(paramBank.warped.s_100));
end
if isfield(paramBank,'gp_scaled_emp') && isfield(paramBank.gp_scaled_emp,'s_100')
    fprintf('gp_scaled_emp hyp length: %d\n', numel(paramBank.gp_scaled_emp.s_100));
end

%% ---------------- SAVE ----------------------------------------
if saveOut
    save(outFile, 'paramBank', 'paramTable');
    fprintf('\nSaved: %s\n', outFile);
end

%% ============================================================
%  Local function: extract ModelInfo.hyp (and optionally caches)
%% ============================================================
function out = localExtract(T, tag, canon)

    if ~istable(T)
        error('Expected %s to be a table.', tag);
    end

    requiredCols = ["holdout_station","case_name","RhoFunction","ModelInfo"];
    missing = setdiff(requiredCols, string(T.Properties.VariableNames));
    if ~isempty(missing)
        error('%s is missing required columns: %s', tag, strjoin(missing,", "));
    end

    cn = string(T.case_name);
    rf = string(T.RhoFunction);

    modelKey = strings(height(T),1);
    modelKey(:) = "";

    for i = 1:height(T)
        if canon.warped(cn(i),rf(i))
            modelKey(i) = "warped";
        elseif canon.constant(cn(i),rf(i))
            modelKey(i) = "constant";
        elseif canon.gp_scaled_emp(cn(i),rf(i))
            modelKey(i) = "gp_scaled_emp";
        elseif canon.gp_st(cn(i),rf(i))
            modelKey(i) = "gp_st";
        else
            modelKey(i) = ""; % ignore unknown rows
        end
    end

    keep = modelKey ~= "";
    T = T(keep,:);
    modelKey = modelKey(keep);

    if height(T)==0
        out = table();
        return;
    end

    hypCell   = cell(height(T),1);
    debugCell = cell(height(T),1);

    for i = 1:height(T)

        % ---- robust access: works for both cell and struct columns
        MI = getRowModelInfo(T.ModelInfo, i);

        if isempty(MI) || ~isstruct(MI)
            error('Row %d in %s has empty/non-struct ModelInfo.', i, tag);
        end
        if ~isfield(MI,'hyp') || isempty(MI.hyp)
            error('Row %d in %s: ModelInfo.hyp missing/empty.', i, tag);
        end

        hypCell{i} = MI.hyp(:);

        % Optional cached pieces (only if present)
        dbg = struct();
        if isfield(MI,'debug_vecchia'), dbg.debug_vecchia = MI.debug_vecchia; end
        if isfield(MI,'perm'),         dbg.perm = MI.perm; end
        if isfield(MI,'R'),            dbg.R = MI.R; end
        if isfield(MI,'H'),            dbg.H = MI.H; end
        debugCell{i} = dbg;
    end

    out = table();
    out.source_table    = repmat(string(tag), height(T), 1);
    out.holdout_station = double(T.holdout_station);
    out.case_name       = string(T.case_name);
    out.RhoFunction     = string(T.RhoFunction);
    out.modelKey        = cellstr(modelKey);

    out.hyp             = hypCell;
    out.debug_cache     = debugCell;

    % carry metrics if present
    metricCols = intersect(["NLML","RMSE","MAE","CORR"], string(T.Properties.VariableNames));
    for m = 1:numel(metricCols)
        out.(metricCols(m)) = T.(metricCols(m));
    end
end


function MI = getRowModelInfo(modelInfoCol, i)
% Robustly returns the ModelInfo struct for row i, regardless of storage type:
% - cell column: ModelInfo{i}
% - struct column: ModelInfo(i)
% - struct scalar repeated (rare): ModelInfo

    if iscell(modelInfoCol)
        MI = modelInfoCol{i};
        return;
    end

    if isstruct(modelInfoCol)
        % struct array: pick i-th element
        if numel(modelInfoCol) >= i
            MI = modelInfoCol(i);
        else
            % struct scalar (unlikely but safe fallback)
            MI = modelInfoCol;
        end
        return;
    end

    error('ModelInfo column type not supported: %s', class(modelInfoCol));
end
